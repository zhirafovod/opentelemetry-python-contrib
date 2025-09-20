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

import os
import time
from typing import Any, Optional

from opentelemetry import _events as _otel_events
from opentelemetry import metrics as _metrics
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import get_tracer
from opentelemetry.util.genai import (
    evaluators as _genai_evaluators,  # noqa: F401  # trigger builtin registration
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_ENABLE,
    OTEL_INSTRUMENTATION_GENAI_EVALUATORS,
)
from opentelemetry.util.genai.evaluators.registry import get_evaluator
from opentelemetry.util.genai.generators import SpanGenerator
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    Error,
    EvaluationResult,
    LLMInvocation,
)
from opentelemetry.util.genai.utils import get_content_capturing_mode
from opentelemetry.util.genai.version import __version__


class TelemetryHandler:
    """
    High-level handler managing GenAI invocation lifecycles and emitting
    them as spans, metrics, and events.
    """

    def __init__(self, **kwargs: Any):
        tracer_provider = kwargs.get("tracer_provider")
        self._tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_36_0.value,
        )
        self._event_logger = _otel_events.get_event_logger(__name__)
        meter = _metrics.get_meter(__name__)
        # Single histogram for all evaluation scores (name stable across metrics)
        self._evaluation_histogram = meter.create_histogram(
            name="gen_ai.evaluation.score",
            unit="1",
            description="Scores produced by GenAI evaluators in [0,1] when applicable",
        )

        capture_content = False
        try:
            mode = get_content_capturing_mode()
            capture_content = mode in (
                ContentCapturingMode.SPAN_ONLY,
                ContentCapturingMode.SPAN_AND_EVENT,
            )
        except (
            Exception
        ):  # ValueError for default stability or other issues; ignore silently
            capture_content = False
        self._generator = SpanGenerator(
            tracer=self._tracer, capture_content=capture_content
        )

    def _refresh_capture_content(
        self,
    ):  # re-evaluate env each start in case singleton created before patching
        try:
            mode = get_content_capturing_mode()
            self._generator._capture_content = mode in (
                ContentCapturingMode.SPAN_ONLY,
                ContentCapturingMode.SPAN_AND_EVENT,
            )
        except Exception:
            # Leave existing setting unchanged if stability mode default or invalid
            pass

    def start_llm(
        self,
        invocation: LLMInvocation,
    ) -> LLMInvocation:
        """Start an LLM invocation and create a pending span entry."""
        self._refresh_capture_content()
        self._generator.start(invocation)
        return invocation

    def stop_llm(self, invocation: LLMInvocation) -> LLMInvocation:
        """Finalize an LLM invocation successfully and end its span."""
        invocation.end_time = time.time()
        self._generator.finish(invocation)
        return invocation

    def fail_llm(
        self, invocation: LLMInvocation, error: Error
    ) -> LLMInvocation:
        """Fail an LLM invocation and end its span with error status."""
        invocation.end_time = time.time()
        self._generator.error(error, invocation)
        return invocation

    def evaluate_llm(
        self,
        invocation: LLMInvocation,
        evaluators: Optional[list[str]] = None,
    ) -> list[EvaluationResult]:
        """Run registered evaluators against a completed LLMInvocation.

        Phase 2: Executes evaluator backends, records scores to a unified histogram
        (gen_ai.evaluation.score) and emits a gen_ai.evaluations event containing all
        metric results. Evaluation spans are not yet implemented (planned for Phase 3).

        Evaluation enablement is controlled by the environment variable
        OTEL_INSTRUMENTATION_GENAI_EVALUATION_ENABLE. If not enabled, this
        returns an empty list.

        Args:
            invocation: The LLMInvocation that has been finished (stop_llm or fail_llm).
            evaluators: Optional explicit list of evaluator names. If None, falls back
                to OTEL_INSTRUMENTATION_GENAI_EVALUATORS (comma-separated). If still
                empty, returns [] immediately.

        Returns:
            A list of EvaluationResult objects (possibly empty).
        """
        enabled_val = os.environ.get(
            OTEL_INSTRUMENTATION_GENAI_EVALUATION_ENABLE, "false"
        ).lower()
        if enabled_val not in ("true", "1", "yes"):  # disabled
            return []

        if evaluators is None:
            env_names = os.environ.get(
                OTEL_INSTRUMENTATION_GENAI_EVALUATORS, ""
            ).strip()
            if env_names:
                evaluators = [
                    n.strip() for n in env_names.split(",") if n.strip()
                ]
            else:
                evaluators = []
        if not evaluators:
            return []

        results: list[EvaluationResult] = []
        # Ensure invocation end_time is set (user might have forgotten to call stop_llm)
        if invocation.end_time is None:
            invocation.end_time = time.time()

        for name in evaluators:
            try:
                evaluator = get_evaluator(name)
            except Exception as exc:  # unknown evaluator or construction error
                results.append(
                    EvaluationResult(
                        metric_name=name,
                        error=Error(message=str(exc), type=type(exc)),
                    )
                )
                continue
            try:
                eval_out = evaluator.evaluate(invocation)
                # Normalise: allow evaluator to return single or list
                if isinstance(eval_out, EvaluationResult):
                    results.append(eval_out)
                elif isinstance(eval_out, list):
                    results.extend(eval_out)
                else:
                    # Unsupported return type -> mark as error
                    results.append(
                        EvaluationResult(
                            metric_name=name,
                            error=Error(
                                message=f"Unsupported evaluation return type: {type(eval_out)}",
                                type=TypeError,
                            ),
                        )
                    )
            except Exception as exc:  # evaluator runtime error
                results.append(
                    EvaluationResult(
                        metric_name=name,
                        error=Error(message=str(exc), type=type(exc)),
                    )
                )
        # Phase 2: emit metrics & event
        if results:
            evaluation_items = []
            for res in results:
                attrs = {
                    "gen_ai.operation.name": "evaluation",
                    "gen_ai.evaluation.name": res.metric_name,
                    "gen_ai.request.model": invocation.request_model,
                }
                if invocation.provider:
                    attrs["gen_ai.provider.name"] = invocation.provider
                if res.label is not None:
                    attrs["gen_ai.evaluation.score.label"] = res.label
                if res.error is not None:
                    attrs["error.type"] = res.error.type.__qualname__
                # Record metric if score present and numeric
                if isinstance(res.score, (int, float)):
                    self._evaluation_histogram.record(
                        res.score,
                        attributes={
                            k: v for k, v in attrs.items() if v is not None
                        },
                    )
                # Build event body item
                item = {
                    "gen_ai.evaluation.name": res.metric_name,
                }
                if isinstance(res.score, (int, float)):
                    item["gen_ai.evaluation.score.value"] = res.score
                if res.label is not None:
                    item["gen_ai.evaluation.score.label"] = res.label
                if res.explanation:
                    item["gen_ai.evaluation.explanation"] = res.explanation
                if res.error is not None:
                    item["error.type"] = res.error.type.__qualname__
                    item["error.message"] = res.error.message
                # include custom attributes from evaluator result
                for k, v in res.attributes.items():
                    item[k] = v
                evaluation_items.append(item)
            if evaluation_items:
                event_attrs = {
                    "gen_ai.operation.name": "evaluation",
                    "gen_ai.request.model": invocation.request_model,
                }
                if invocation.provider:
                    event_attrs["gen_ai.provider.name"] = invocation.provider
                if invocation.response_id:
                    event_attrs["gen_ai.response.id"] = invocation.response_id
                event_body = {"evaluations": evaluation_items}
                try:
                    self._event_logger.emit(
                        _otel_events.Event(
                            name="gen_ai.evaluations",
                            attributes=event_attrs,
                            body=event_body,
                            # Link to invocation span if available
                            span_id=invocation.span.get_span_context().span_id
                            if invocation.span
                            else None,
                            trace_id=invocation.span.get_span_context().trace_id
                            if invocation.span
                            else None,
                        )
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
        return results


def get_telemetry_handler(**kwargs: Any) -> TelemetryHandler:
    """
    Returns a singleton TelemetryHandler instance.
    """
    handler: Optional[TelemetryHandler] = getattr(
        get_telemetry_handler, "_default_handler", None
    )
    if handler is None:
        handler = TelemetryHandler(**kwargs)
        setattr(get_telemetry_handler, "_default_handler", handler)
    return handler
