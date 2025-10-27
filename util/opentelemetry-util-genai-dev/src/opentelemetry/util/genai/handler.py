# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Telemetry handler for GenAI invocations.

This module exposes the `TelemetryHandler` class, which manages the lifecycle of
GenAI (Generative AI) invocations and emits telemetry data (spans and related attributes).
It supports starting, stopping, and failing LLM invocations.

Classes:
    - TelemetryHandler: Manages GenAI invocation lifecycles and emits telemetry.

Functions:
    - get_telemetry_handler: Returns a singleton `TelemetryHandler` instance.

Usage:
    handler = get_telemetry_handler()

    # Create an invocation object with your request data
    invocation = LLMInvocation(
        request_model="my-model",
        input_messages=[...],
        provider="my-provider",
        attributes={"custom": "attr"},
    )

    # Start the invocation (opens a span)
    handler.start_llm(invocation)

    # Populate outputs and any additional attributes, then stop (closes the span)
    invocation.output_messages = [...]
    invocation.attributes.update({"more": "attrs"})
    handler.stop_llm(invocation)

    # Or, in case of error
    # handler.fail_llm(invocation, Error(type="...", message="..."))
"""

import logging
import os
import time
from typing import Any, Optional

from opentelemetry.util._importlib_metadata import (
    entry_points,  # pyright: ignore[reportUnknownVariableType]
)

try:
    from opentelemetry.util.genai.debug import genai_debug_log
except Exception:  # pragma: no cover - fallback if debug module missing

    def genai_debug_log(*_args: Any, **_kwargs: Any) -> None:  # type: ignore
        return None


from opentelemetry import _events as _otel_events
from opentelemetry import _logs
from opentelemetry import metrics as _metrics
from opentelemetry import trace as _trace_mod
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import get_tracer
from opentelemetry.util.genai.emitters.configuration import (
    build_emitter_pipeline,
)
from opentelemetry.util.genai.span_context import (
    extract_span_context,
    span_context_hex_ids,
    store_span_context,
)
from opentelemetry.util.genai.types import (
    AgentCreation,
    AgentInvocation,
    ContentCapturingMode,
    EmbeddingInvocation,
    Error,
    EvaluationResult,
    GenAI,
    LLMInvocation,
    Step,
    ToolCall,
    Workflow,
)
from opentelemetry.util.genai.utils import get_content_capturing_mode
from opentelemetry.util.genai.version import __version__

from .callbacks import CompletionCallback
from .config import parse_env
from .environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS,
    OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS,
)

_LOGGER = logging.getLogger(__name__)

_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _coerce_completion_callback(
    provider: object, name: str
) -> CompletionCallback | None:
    if provider is None:
        return None
    if hasattr(provider, "on_completion"):
        if isinstance(provider, type):
            try:
                instance = provider()
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning(
                    "Completion callback class '%s' failed to instantiate: %s",
                    name,
                    exc,
                )
                return None
            return instance  # type: ignore[return-value]
        return provider  # type: ignore[return-value]
    if callable(provider):
        try:
            instance = provider()
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning(
                "Completion callback factory '%s' raised an exception: %s",
                name,
                exc,
            )
            return None
        if hasattr(instance, "on_completion"):
            return instance  # type: ignore[return-value]
        _LOGGER.warning(
            "Completion callback factory '%s' returned an object without on_completion",
            name,
        )
        return None
    _LOGGER.warning(
        "Completion callback entry point '%s' is not callable or instance",
        name,
    )
    return None


def _load_completion_callbacks(
    selected: set[str] | None,
) -> tuple[list[tuple[str, CompletionCallback]], set[str]]:
    callbacks: list[tuple[str, CompletionCallback]] = []
    seen: set[str] = set()
    try:
        entries = entry_points(
            group="opentelemetry_util_genai_completion_callbacks"
        )
    except Exception:  # pragma: no cover - defensive
        _LOGGER.debug("Completion callback entry point group not available")
        return callbacks, seen
    for ep in entries:  # type: ignore[assignment]
        name = getattr(ep, "name", "")
        lowered = name.lower()
        seen.add(lowered)
        if selected and lowered not in selected:
            continue
        try:
            provider = ep.load()
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.debug(
                "Failed to load completion callback '%s': %s",
                name,
                exc,
                exc_info=True,
            )
            continue
        instance = _coerce_completion_callback(provider, name)
        if instance is None:
            continue
        callbacks.append((name, instance))
    return callbacks, seen


def _is_truthy_env(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_VALUES


def _parse_callback_filter(value: Optional[str]) -> set[str] | None:
    if value is None:
        return None
    selected = {
        item.strip().lower() for item in value.split(",") if item.strip()
    }
    return selected or None


class TelemetryHandler:
    """
    High-level handler managing GenAI invocation lifecycles and emitting
    them as spans, metrics, and events. Evaluation execution & emission is
    delegated to EvaluationManager for extensibility (mirrors emitter design).
    """

    def __init__(self, **kwargs: Any):
        tracer_provider = kwargs.get("tracer_provider")
        # Store provider reference for later identity comparison (test isolation)
        # Use already imported _trace_mod for provider reference; avoid re-import for lint.
        self._tracer_provider_ref = (
            tracer_provider or _trace_mod.get_tracer_provider()
        )
        self._tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_36_0.value,
        )
        self._event_logger = _otel_events.get_event_logger(__name__)
        # Logger for content events (uses Logs API, not Events API)
        self._content_logger = _logs.get_logger(__name__)
        meter_provider = kwargs.get("meter_provider")
        self._meter_provider = meter_provider  # store for flushing in tests
        if meter_provider is not None:
            meter = meter_provider.get_meter(__name__)
        else:
            meter = _metrics.get_meter(__name__)
        # Fixed canonical evaluation histograms (no longer dynamic):
        # gen_ai.evaluation.(relevance|hallucination|sentiment|toxicity|bias)
        self._evaluation_histograms: dict[str, Any] = {}

        _CANONICAL_METRICS = {
            "relevance",
            "hallucination",
            "sentiment",
            "toxicity",
            "bias",
        }

        def _get_eval_histogram(canonical_name: str):
            name = canonical_name.strip().lower()
            if name not in _CANONICAL_METRICS:
                return None  # ignore unknown metrics (no emission)
            full_name = f"gen_ai.evaluation.{name}"
            hist = self._evaluation_histograms.get(full_name)
            if hist is not None:
                return hist
            try:
                hist = meter.create_histogram(
                    name=full_name,
                    unit="1",
                    description=f"GenAI evaluation metric '{name}' (0-1 score where applicable)",
                )
                self._evaluation_histograms[full_name] = hist
            except Exception:  # pragma: no cover - defensive
                return None
            return hist

        self._get_eval_histogram = _get_eval_histogram  # type: ignore[attr-defined]

        settings = parse_env()
        self._completion_callbacks: list[CompletionCallback] = []
        composite, capture_control = build_emitter_pipeline(
            tracer=self._tracer,
            meter=meter,
            event_logger=self._event_logger,
            content_logger=self._content_logger,
            evaluation_histogram=self._get_eval_histogram,
            settings=settings,
        )
        self._emitter = composite
        self._capture_control = capture_control
        self._evaluation_manager = None
        # Active agent identity stack (name, id) for implicit propagation to nested operations
        self._agent_context_stack: list[tuple[str, str]] = []
        # Span registry (run_id -> Span) to allow parenting even after original invocation ended.
        # We intentionally retain ended parent spans to preserve trace linkage for late children
        # (e.g., final LLM call after agent/workflow termination). A lightweight size cap can be
        # added later if memory pressure surfaces.
        self._span_registry: dict[str, _trace_mod.Span] = {}
        # Generic entity registry (run_id -> entity object) allowing instrumentation
        # layers to avoid storing lifecycle objects. This supports simplified
        # instrumentations that only pass run_id on end/error callbacks.
        self._entity_registry: dict[str, GenAI] = {}
        self._initialize_default_callbacks()

    def _refresh_capture_content(
        self,
    ):  # re-evaluate env each start in case singleton created before patching
        try:
            mode = get_content_capturing_mode()
            emitters = list(
                self._emitter.iter_emitters(("span", "content_events"))
            )
            # Determine new values for span-like emitters
            new_value_span = mode in (
                ContentCapturingMode.SPAN_ONLY,
                ContentCapturingMode.SPAN_AND_EVENT,
            )
            control = getattr(self, "_capture_control", None)
            span_capture_allowed = True
            if control is not None:
                span_capture_allowed = control.span_allowed
            if _is_truthy_env(
                os.environ.get(
                    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
                )
            ):
                span_capture_allowed = True
            # Respect the content capture mode for all generator kinds
            new_value_events = mode in (
                ContentCapturingMode.EVENT_ONLY,
                ContentCapturingMode.SPAN_AND_EVENT,
            )
            for em in emitters:
                role = getattr(em, "role", None)
                if role == "content_event" and hasattr(em, "_capture_content"):
                    try:
                        em._capture_content = new_value_events  # type: ignore[attr-defined]
                    except Exception:
                        pass
                elif role in ("span", "traceloop_compat") and hasattr(
                    em, "set_capture_content"
                ):
                    try:
                        desired_span = new_value_span and span_capture_allowed
                        if role == "traceloop_compat":
                            desired = desired_span or new_value_events
                        else:
                            desired = desired_span
                        em.set_capture_content(desired)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception:
            pass

    def start_llm(
        self,
        invocation: LLMInvocation,
    ) -> LLMInvocation:
        """Start an LLM invocation and create a pending span entry."""
        # Ensure capture content settings are current
        self._refresh_capture_content()
        genai_debug_log("handler.start_llm.begin", invocation)
        # Implicit agent inheritance
        if (
            not invocation.agent_name or not invocation.agent_id
        ) and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not invocation.agent_name:
                invocation.agent_name = top_name
            if not invocation.agent_id:
                invocation.agent_id = top_id
        # Start invocation span; tracer context propagation handles parent/child links
        self._emitter.on_start(invocation)
        # Register span if created
        span = getattr(invocation, "span", None)
        if span is not None:
            self._span_registry[str(invocation.run_id)] = span
        # Register entity for later stop/fail by run_id
        self._entity_registry[str(invocation.run_id)] = invocation
        try:
            span_context = invocation.span_context
            if span_context is None and invocation.span is not None:
                span_context = extract_span_context(invocation.span)
                store_span_context(invocation, span_context)
            trace_hex, span_hex = span_context_hex_ids(span_context)
            if trace_hex and span_hex:
                genai_debug_log(
                    "handler.start_llm.span_created",
                    invocation,
                    trace_id=trace_hex,
                    span_id=span_hex,
                )
            else:
                genai_debug_log("handler.start_llm.no_span", invocation)
        except Exception:  # pragma: no cover
            pass
        return invocation

    def stop_llm(self, invocation: LLMInvocation) -> LLMInvocation:
        """Finalize an LLM invocation successfully and end its span."""
        invocation.end_time = time.time()
        self._emitter.on_end(invocation)
        self._notify_completion(invocation)
        self._entity_registry.pop(str(invocation.run_id), None)
        try:
            span_context = invocation.span_context
            if span_context is None and invocation.span is not None:
                span_context = extract_span_context(invocation.span)
                store_span_context(invocation, span_context)
            trace_hex, span_hex = span_context_hex_ids(span_context)
            genai_debug_log(
                "handler.stop_llm.complete",
                invocation,
                duration_ms=round(
                    (invocation.end_time - invocation.start_time) * 1000, 3
                )
                if invocation.end_time
                else None,
                trace_id=trace_hex,
                span_id=span_hex,
            )
        except Exception:  # pragma: no cover
            pass
        # Force flush metrics if a custom provider with force_flush is present
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover - defensive
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return invocation

    def fail_llm(
        self, invocation: LLMInvocation, error: Error
    ) -> LLMInvocation:
        """Fail an LLM invocation and end its span with error status."""
        invocation.end_time = time.time()
        self._emitter.on_error(error, invocation)
        self._notify_completion(invocation)
        self._entity_registry.pop(str(invocation.run_id), None)
        try:
            span_context = invocation.span_context
            if span_context is None and invocation.span is not None:
                span_context = extract_span_context(invocation.span)
                store_span_context(invocation, span_context)
            trace_hex, span_hex = span_context_hex_ids(span_context)
            genai_debug_log(
                "handler.fail_llm.error",
                invocation,
                error_type=getattr(error, "type", None),
                error_message=getattr(error, "message", None),
                trace_id=trace_hex,
                span_id=span_hex,
            )
        except Exception:  # pragma: no cover
            pass
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return invocation

    def start_embedding(
        self, invocation: EmbeddingInvocation
    ) -> EmbeddingInvocation:
        """Start an embedding invocation and create a pending span entry."""
        self._refresh_capture_content()
        if (
            not invocation.agent_name or not invocation.agent_id
        ) and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not invocation.agent_name:
                invocation.agent_name = top_name
            if not invocation.agent_id:
                invocation.agent_id = top_id
        invocation.start_time = time.time()
        self._emitter.on_start(invocation)
        span = getattr(invocation, "span", None)
        if span is not None:
            self._span_registry[str(invocation.run_id)] = span
        self._entity_registry[str(invocation.run_id)] = invocation
        return invocation

    def stop_embedding(
        self, invocation: EmbeddingInvocation
    ) -> EmbeddingInvocation:
        """Finalize an embedding invocation successfully and end its span."""
        invocation.end_time = time.time()
        self._emitter.on_end(invocation)
        self._notify_completion(invocation)
        self._entity_registry.pop(str(invocation.run_id), None)
        # Force flush metrics if a custom provider with force_flush is present
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return invocation

    def fail_embedding(
        self, invocation: EmbeddingInvocation, error: Error
    ) -> EmbeddingInvocation:
        """Fail an embedding invocation and end its span with error status."""
        invocation.end_time = time.time()
        self._emitter.on_error(error, invocation)
        self._notify_completion(invocation)
        self._entity_registry.pop(str(invocation.run_id), None)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return invocation

    # ToolCall lifecycle --------------------------------------------------
    def start_tool_call(self, invocation: ToolCall) -> ToolCall:
        """Start a tool call invocation and create a pending span entry."""
        if (
            not invocation.agent_name or not invocation.agent_id
        ) and self._agent_context_stack:
            top_name, top_id = self._agent_context_stack[-1]
            if not invocation.agent_name:
                invocation.agent_name = top_name
            if not invocation.agent_id:
                invocation.agent_id = top_id
        self._emitter.on_start(invocation)
        span = getattr(invocation, "span", None)
        if span is not None:
            self._span_registry[str(invocation.run_id)] = span
        self._entity_registry[str(invocation.run_id)] = invocation
        return invocation

    def stop_tool_call(self, invocation: ToolCall) -> ToolCall:
        """Finalize a tool call invocation successfully and end its span."""
        invocation.end_time = time.time()
        self._emitter.on_end(invocation)
        self._notify_completion(invocation)
        self._entity_registry.pop(str(invocation.run_id), None)
        return invocation

    def fail_tool_call(self, invocation: ToolCall, error: Error) -> ToolCall:
        """Fail a tool call invocation and end its span with error status."""
        invocation.end_time = time.time()
        self._emitter.on_error(error, invocation)
        self._notify_completion(invocation)
        self._entity_registry.pop(str(invocation.run_id), None)
        return invocation

    # Workflow lifecycle --------------------------------------------------
    def start_workflow(self, workflow: Workflow) -> Workflow:
        """Start a workflow and create a pending span entry."""
        self._refresh_capture_content()
        self._emitter.on_start(workflow)
        span = getattr(workflow, "span", None)
        if span is not None:
            self._span_registry[str(workflow.run_id)] = span
        self._entity_registry[str(workflow.run_id)] = workflow
        return workflow

    def _handle_evaluation_results(
        self, invocation: GenAI, results: list[EvaluationResult]
    ) -> None:
        if not results:
            return
        try:
            self._emitter.on_evaluation_results(results, invocation)
        except Exception:  # pragma: no cover - defensive
            pass

    def evaluation_results(
        self, invocation: GenAI, results: list[EvaluationResult]
    ) -> None:
        """Public hook for completion callbacks to report evaluation output."""

        try:
            genai_debug_log(
                "handler.evaluation_results.begin",
                invocation,
                result_count=len(results),
            )
        except Exception:  # pragma: no cover - defensive
            pass
        self._handle_evaluation_results(invocation, results)
        try:
            genai_debug_log(
                "handler.evaluation_results.end",
                invocation,
                result_count=len(results),
            )
        except Exception:  # pragma: no cover - defensive
            pass

    def register_completion_callback(
        self, callback: CompletionCallback
    ) -> None:
        if callback in self._completion_callbacks:
            return
        self._completion_callbacks.append(callback)

    def unregister_completion_callback(
        self, callback: CompletionCallback
    ) -> None:
        try:
            self._completion_callbacks.remove(callback)
        except ValueError:
            pass

    def _notify_completion(self, invocation: GenAI) -> None:
        if not self._completion_callbacks:
            return
        callbacks = list(self._completion_callbacks)
        for callback in callbacks:
            try:
                callback.on_completion(invocation)
            except Exception:  # pragma: no cover - defensive
                continue

    def _initialize_default_callbacks(self) -> None:
        disable_defaults = _is_truthy_env(
            os.getenv(
                OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS
            )
        )
        if disable_defaults:
            _LOGGER.debug(
                "Default completion callbacks disabled via %s",
                OTEL_INSTRUMENTATION_GENAI_DISABLE_DEFAULT_COMPLETION_CALLBACKS,
            )
            return

        selected = _parse_callback_filter(
            os.getenv(OTEL_INSTRUMENTATION_GENAI_COMPLETION_CALLBACKS)
        )
        callbacks, seen = _load_completion_callbacks(selected)
        if selected:
            missing = selected - seen
            for name in missing:
                _LOGGER.debug(
                    "Completion callback '%s' not found in entry points",
                    name,
                )
        if not callbacks:
            return

        for name, callback in callbacks:
            bound_ok = True
            binder = getattr(callback, "bind_handler", None)
            if callable(binder):
                try:
                    bound_ok = bool(binder(self))
                except Exception as exc:  # pragma: no cover - defensive
                    _LOGGER.warning(
                        "Completion callback '%s' failed to bind: %s",
                        name,
                        exc,
                    )
                    shutdown = getattr(callback, "shutdown", None)
                    if callable(shutdown):
                        try:
                            shutdown()
                        except Exception:  # pragma: no cover - defensive
                            pass
                    continue
            if not bound_ok:
                shutdown = getattr(callback, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception:  # pragma: no cover - defensive
                        pass
                continue
            manager = getattr(callback, "manager", None)
            if manager is not None:
                self._evaluation_manager = manager
            self.register_completion_callback(callback)

    def stop_workflow(self, workflow: Workflow) -> Workflow:
        """Finalize a workflow successfully and end its span."""
        workflow.end_time = time.time()
        self._emitter.on_end(workflow)
        self._notify_completion(workflow)
        self._entity_registry.pop(str(workflow.run_id), None)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return workflow

    def fail_workflow(self, workflow: Workflow, error: Error) -> Workflow:
        """Fail a workflow and end its span with error status."""
        workflow.end_time = time.time()
        self._emitter.on_error(error, workflow)
        self._notify_completion(workflow)
        self._entity_registry.pop(str(workflow.run_id), None)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return workflow

    # Agent lifecycle -----------------------------------------------------
    def start_agent(
        self, agent: AgentCreation | AgentInvocation
    ) -> AgentCreation | AgentInvocation:
        """Start an agent operation (create or invoke) and create a pending span entry."""
        self._refresh_capture_content()
        self._emitter.on_start(agent)
        span = getattr(agent, "span", None)
        if span is not None:
            self._span_registry[str(agent.run_id)] = span
        self._entity_registry[str(agent.run_id)] = agent
        # Push agent identity context (use run_id as canonical id)
        if isinstance(agent, AgentInvocation):
            try:
                if agent.name:
                    self._agent_context_stack.append(
                        (agent.name, str(agent.run_id))
                    )
            except Exception:  # pragma: no cover - defensive
                pass
        return agent

    def stop_agent(
        self, agent: AgentCreation | AgentInvocation
    ) -> AgentCreation | AgentInvocation:
        """Finalize an agent operation successfully and end its span."""
        agent.end_time = time.time()
        self._emitter.on_end(agent)
        self._notify_completion(agent)
        self._entity_registry.pop(str(agent.run_id), None)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        # Pop context if matches top
        if isinstance(agent, AgentInvocation):
            try:
                if self._agent_context_stack:
                    top_name, top_id = self._agent_context_stack[-1]
                    if top_name == agent.name and top_id == str(agent.run_id):
                        self._agent_context_stack.pop()
            except Exception:
                pass
        return agent

    def fail_agent(
        self, agent: AgentCreation | AgentInvocation, error: Error
    ) -> AgentCreation | AgentInvocation:
        """Fail an agent operation and end its span with error status."""
        agent.end_time = time.time()
        self._emitter.on_error(error, agent)
        self._notify_completion(agent)
        self._entity_registry.pop(str(agent.run_id), None)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        # Pop context if this agent is active
        if isinstance(agent, AgentInvocation):
            try:
                if self._agent_context_stack:
                    top_name, top_id = self._agent_context_stack[-1]
                    if top_name == agent.name and top_id == str(agent.run_id):
                        self._agent_context_stack.pop()
            except Exception:
                pass
        return agent

    # Step lifecycle ------------------------------------------------------
    def start_step(self, step: Step) -> Step:
        """Start a step and create a pending span entry."""
        self._refresh_capture_content()
        self._emitter.on_start(step)
        span = getattr(step, "span", None)
        if span is not None:
            self._span_registry[str(step.run_id)] = span
        self._entity_registry[str(step.run_id)] = step
        return step

    def stop_step(self, step: Step) -> Step:
        """Finalize a step successfully and end its span."""
        step.end_time = time.time()
        self._emitter.on_end(step)
        self._notify_completion(step)
        self._entity_registry.pop(str(step.run_id), None)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return step

    def fail_step(self, step: Step, error: Error) -> Step:
        """Fail a step and end its span with error status."""
        step.end_time = time.time()
        self._emitter.on_error(error, step)
        self._notify_completion(step)
        self._entity_registry.pop(str(step.run_id), None)
        if (
            hasattr(self, "_meter_provider")
            and self._meter_provider is not None
        ):
            try:  # pragma: no cover
                self._meter_provider.force_flush()  # type: ignore[attr-defined]
            except Exception:
                pass
        return step

    def evaluate_llm(
        self,
        invocation: LLMInvocation,
        evaluators: Optional[list[str]] = None,
    ) -> list[EvaluationResult]:
        """Proxy to EvaluationManager for running evaluators.

        Retained public signature for backward compatibility. The underlying
        implementation has been refactored into EvaluationManager to allow
        pluggable emission similar to emitters.
        """
        manager = getattr(self, "_evaluation_manager", None)
        if manager is None or not manager.has_evaluators:
            return []
        if evaluators:
            _LOGGER.warning(
                "Direct evaluator overrides are ignored; using configured evaluators"
            )
        return manager.evaluate_now(invocation)  # type: ignore[attr-defined]

    def evaluate_agent(
        self,
        agent: AgentInvocation,
        evaluators: Optional[list[str]] = None,
    ) -> list[EvaluationResult]:
        """Run evaluators against an AgentInvocation.

        Mirrors evaluate_llm to allow explicit agent evaluation triggering.
        """
        if not isinstance(agent, AgentInvocation):
            _LOGGER.debug(
                "Skipping agent evaluation for non-invocation type: %s",
                type(agent).__name__,
            )
            return []
        manager = getattr(self, "_evaluation_manager", None)
        if manager is None or not manager.has_evaluators:
            return []
        if evaluators:
            _LOGGER.warning(
                "Direct evaluator overrides are ignored; using configured evaluators"
            )
        return manager.evaluate_now(agent)  # type: ignore[attr-defined]

    def wait_for_evaluations(self, timeout: Optional[float] = None) -> None:
        """Wait for all pending evaluations to complete, up to the specified timeout.

        This is primarily intended for use in test scenarios to ensure that
        all asynchronous evaluation steps have finished before assertions are made.
        """
        manager = getattr(self, "_evaluation_manager", None)
        if manager is None or not manager.has_evaluators:
            return
        manager.wait_for_all(timeout)  # type: ignore[attr-defined]

    # Generic lifecycle API ------------------------------------------------
    def start(self, obj: Any) -> Any:
        """Generic start method for any invocation type."""
        if isinstance(obj, Workflow):
            return self.start_workflow(obj)
        if isinstance(obj, (AgentCreation, AgentInvocation)):
            return self.start_agent(obj)
        if isinstance(obj, Step):
            return self.start_step(obj)
        if isinstance(obj, LLMInvocation):
            return self.start_llm(obj)
        if isinstance(obj, EmbeddingInvocation):
            return self.start_embedding(obj)
        if isinstance(obj, ToolCall):
            return self.start_tool_call(obj)
        return obj

    # ---- registry helpers -----------------------------------------------
    def get_span_by_run_id(
        self, run_id: Any
    ) -> Optional[_trace_mod.Span]:  # run_id may be UUID or str
        try:
            key = str(run_id)
        except Exception:
            return None
        return self._span_registry.get(key)

    def has_span(self, run_id: Any) -> bool:
        try:
            return str(run_id) in self._span_registry
        except Exception:
            return False

    # ---- entity registry helpers ---------------------------------------
    def get_entity(self, run_id: Any) -> Optional[GenAI]:
        try:
            return self._entity_registry.get(str(run_id))
        except Exception:
            return None

    def finish_by_run_id(self, run_id: Any) -> None:
        entity = self.get_entity(run_id)
        if entity is None:
            return
        if isinstance(entity, Workflow):
            self.stop_workflow(entity)
        elif isinstance(entity, (AgentCreation, AgentInvocation)):
            self.stop_agent(entity)
        elif isinstance(entity, Step):
            self.stop_step(entity)
        elif isinstance(entity, LLMInvocation):
            self.stop_llm(entity)
        elif isinstance(entity, EmbeddingInvocation):
            self.stop_embedding(entity)
        elif isinstance(entity, ToolCall):
            self.stop_tool_call(entity)

    def fail_by_run_id(self, run_id: Any, error: Error) -> None:
        entity = self.get_entity(run_id)
        if entity is None:
            return
        if isinstance(entity, Workflow):
            self.fail_workflow(entity, error)
        elif isinstance(entity, (AgentCreation, AgentInvocation)):
            self.fail_agent(entity, error)
        elif isinstance(entity, Step):
            self.fail_step(entity, error)
        elif isinstance(entity, LLMInvocation):
            self.fail_llm(entity, error)
        elif isinstance(entity, EmbeddingInvocation):
            self.fail_embedding(entity, error)
        elif isinstance(entity, ToolCall):
            self.fail_tool_call(entity, error)

    def finish(self, obj: Any) -> Any:
        """Generic finish method for any invocation type."""
        if isinstance(obj, Workflow):
            return self.stop_workflow(obj)
        if isinstance(obj, (AgentCreation, AgentInvocation)):
            return self.stop_agent(obj)
        if isinstance(obj, Step):
            return self.stop_step(obj)
        if isinstance(obj, LLMInvocation):
            return self.stop_llm(obj)
        if isinstance(obj, EmbeddingInvocation):
            return self.stop_embedding(obj)
        if isinstance(obj, ToolCall):
            return self.stop_tool_call(obj)
        return obj

    def fail(self, obj: Any, error: Error) -> Any:
        """Generic fail method for any invocation type."""
        if isinstance(obj, Workflow):
            return self.fail_workflow(obj, error)
        if isinstance(obj, (AgentCreation, AgentInvocation)):
            return self.fail_agent(obj, error)
        if isinstance(obj, Step):
            return self.fail_step(obj, error)
        if isinstance(obj, LLMInvocation):
            return self.fail_llm(obj, error)
        if isinstance(obj, EmbeddingInvocation):
            return self.fail_embedding(obj, error)
        if isinstance(obj, ToolCall):
            return self.fail_tool_call(obj, error)
        return obj


def get_telemetry_handler(**kwargs: Any) -> TelemetryHandler:
    """
    Returns a singleton TelemetryHandler instance. If the global tracer provider
    has changed since the handler was created, a new handler is instantiated so that
    spans are recorded with the active provider (important for test isolation).
    """
    handler: Optional[TelemetryHandler] = getattr(
        get_telemetry_handler, "_default_handler", None
    )
    current_provider = _trace_mod.get_tracer_provider()
    requested_provider = kwargs.get("tracer_provider")
    target_provider = requested_provider or current_provider
    recreate = False
    if handler is not None:
        # Recreate if provider changed or handler lacks provider reference (older instance)
        if not hasattr(handler, "_tracer_provider_ref"):
            recreate = True
        elif handler._tracer_provider_ref is not target_provider:  # type: ignore[attr-defined]
            recreate = True
    if handler is None or recreate:
        handler = TelemetryHandler(**kwargs)
        setattr(get_telemetry_handler, "_default_handler", handler)
    return handler
