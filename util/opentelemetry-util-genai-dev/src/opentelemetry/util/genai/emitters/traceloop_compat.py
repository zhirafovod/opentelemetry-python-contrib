# Traceloop compatibility emitter
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_RESPONSE_ID,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from ..types import Error, LLMInvocation


class TraceloopCompatEmitter:
    """Emitter that recreates (a subset of) the original Traceloop LangChain span format.

    Phase 1 scope:
      * One span per LLMInvocation (no workflow/task/tool hierarchy yet)
      * Span name: ``<callback_name>.chat`` (fallback to ``chat <model>``)
      * Attributes prefixed with ``traceloop.`` copied from invocation.attributes
      * Emits semantic convention attributes from named fields and request_functions
      * Optional content capture (inputs/outputs) if enabled via util-genai content mode
    """

    role = "traceloop_compat"
    name = "traceloop_compat_span"

    def __init__(
        self, tracer: Optional[Tracer] = None, capture_content: bool = False
    ):
        self._tracer: Tracer = tracer or trace.get_tracer(__name__)
        self._capture_content = capture_content

    def set_capture_content(
        self, value: bool
    ):  # pragma: no cover - trivial mutator
        self._capture_content = value

    # Lifecycle -----------------------------------------------------------
    def handles(self, obj: object) -> bool:
        return isinstance(obj, LLMInvocation)

    def _apply_semconv_start(self, invocation: LLMInvocation, span):
        """Apply semantic convention attributes at start."""
        try:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", invocation.request_model)
            if invocation.provider:
                span.set_attribute("gen_ai.provider.name", invocation.provider)
            if invocation.framework:
                span.set_attribute("gen_ai.framework", invocation.framework)
            # function definitions
            if invocation.request_functions:
                for idx, fn in enumerate(invocation.request_functions):
                    name = fn.get("name")
                    if name:
                        span.set_attribute(f"gen_ai.request.function.{idx}.name", name)
                    desc = fn.get("description")
                    if desc:
                        span.set_attribute(f"gen_ai.request.function.{idx}.description", desc)
                    params = fn.get("parameters")
                    if params is not None:
                        span.set_attribute(f"gen_ai.request.function.{idx}.parameters", str(params))
        except Exception:  # pragma: no cover
            pass

    def _apply_semconv_finish(self, invocation: LLMInvocation, span):
        try:
            if invocation.response_model_name:
                span.set_attribute("gen_ai.response.model", invocation.response_model_name)
            if invocation.response_id:
                span.set_attribute(GEN_AI_RESPONSE_ID, invocation.response_id)
            if invocation.input_tokens is not None:
                span.set_attribute("gen_ai.usage.input_tokens", invocation.input_tokens)
            if invocation.output_tokens is not None:
                span.set_attribute("gen_ai.usage.output_tokens", invocation.output_tokens)
            # Reapply function definitions if any added later
            if invocation.request_functions:
                for idx, fn in enumerate(invocation.request_functions):
                    name = fn.get("name")
                    if name:
                        span.set_attribute(f"gen_ai.request.function.{idx}.name", name)
                    desc = fn.get("description")
                    if desc:
                        span.set_attribute(f"gen_ai.request.function.{idx}.description", desc)
                    params = fn.get("parameters")
                    if params is not None:
                        span.set_attribute(f"gen_ai.request.function.{idx}.parameters", str(params))
        except Exception:  # pragma: no cover
            pass

    def start(self, invocation: LLMInvocation) -> None:  # noqa: D401
        if not isinstance(invocation, LLMInvocation):  # defensive
            return
        cb_name = invocation.attributes.get("traceloop.callback_name")
        if cb_name:
            span_name = f"{cb_name}.chat"
        else:
            # Fallback similar but distinct from semconv span naming to avoid collision
            span_name = f"chat {invocation.request_model}"
        cm = self._tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, end_on_exit=False
        )
        span = cm.__enter__()
        # Persist references for finish/error
        invocation.attributes.setdefault("traceloop.span.kind", "llm")
        invocation.__dict__["traceloop_span"] = span
        invocation.__dict__["traceloop_cm"] = cm
        # Copy traceloop.* attributes
        for k, v in invocation.attributes.items():
            # Copy all non-semconv custom attributes (includes traceloop.* and request_* etc.)
            if not k.startswith("gen_ai."):
                try:
                    span.set_attribute(k, v)
                except Exception:  # pragma: no cover
                    pass
        # Apply semantic convention attrs
        self._apply_semconv_start(invocation, span)
        # Input capture
        if self._capture_content and invocation.input_messages:
            try:
                span.set_attribute(
                    "traceloop.entity.input",
                    json.dumps([asdict(m) for m in invocation.input_messages]),
                )
            except Exception:  # pragma: no cover
                pass

    def finish(self, invocation: LLMInvocation) -> None:  # noqa: D401
        span = getattr(invocation, "traceloop_span", None)
        cm = getattr(invocation, "traceloop_cm", None)
        if span is None:
            return
        # Output capture
        if self._capture_content and invocation.output_messages:
            try:
                span.set_attribute(
                    "traceloop.entity.output",
                    json.dumps(
                        [asdict(m) for m in invocation.output_messages]
                    ),
                )
            except Exception:  # pragma: no cover
                pass
        # Apply finish-time semconv attributes + response id
        self._apply_semconv_finish(invocation, span)
        if cm and hasattr(cm, "__exit__"):
            try:  # pragma: no cover
                cm.__exit__(None, None, None)
            except Exception:
                pass
        span.end()

    def error(self, error: Error, invocation: LLMInvocation) -> None:  # noqa: D401
        span = getattr(invocation, "traceloop_span", None)
        cm = getattr(invocation, "traceloop_cm", None)
        if span is None:
            return
        try:
            span.set_status(Status(StatusCode.ERROR, error.message))
        except Exception:  # pragma: no cover
            pass
        # On error still apply finishing semconv attributes if any set
        self._apply_semconv_finish(invocation, span)
        if cm and hasattr(cm, "__exit__"):
            try:  # pragma: no cover
                cm.__exit__(None, None, None)
            except Exception:
                pass
        span.end()
