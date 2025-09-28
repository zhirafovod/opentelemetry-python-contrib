# Span emitter (moved from generators/span_emitter.py)
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.trace.status import Status, StatusCode

from ..attributes import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
)
from ..types import EmbeddingInvocation, Error, LLMInvocation, ToolCall


class SpanEmitter:
    """Span-focused emitter supporting optional content capture.

    Original implementation migrated from generators/span_emitter.py. Additional telemetry
    (metrics, content events) are handled by separate emitters composed via CompositeGenerator.
    """

    role = "span"
    name = "semconv_span"

    def __init__(
        self, tracer: Optional[Tracer] = None, capture_content: bool = False
    ):
        self._tracer: Tracer = tracer or trace.get_tracer(__name__)
        self._capture_content = capture_content

    def set_capture_content(
        self, value: bool
    ):  # pragma: no cover - trivial mutator
        self._capture_content = value

    def handles(self, obj: object) -> bool:
        return True

    # ---- helpers ---------------------------------------------------------
    def _serialize_messages(self, messages):
        try:
            return json.dumps([asdict(m) for m in messages])
        except Exception:  # pragma: no cover
            return None

    def _apply_start_attrs(
        self, invocation: LLMInvocation | EmbeddingInvocation
    ):
        span = getattr(invocation, "span", None)
        if span is None:
            return
        if isinstance(invocation, ToolCall):
            op_value = "tool_call"
        elif isinstance(invocation, EmbeddingInvocation):
            enum_val = getattr(
                GenAI.GenAiOperationNameValues, "EMBEDDING", None
            )
            op_value = enum_val.value if enum_val else "embedding"
        else:
            op_value = GenAI.GenAiOperationNameValues.CHAT.value
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, op_value)
        model_name = (
            invocation.name
            if isinstance(invocation, ToolCall)
            else invocation.request_model
        )
        span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, model_name)
        provider = getattr(invocation, "provider", None)
        if provider:
            span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
        # framework (named field)
        if isinstance(invocation, LLMInvocation) and invocation.framework:
            span.set_attribute("gen_ai.framework", invocation.framework)
        # function definitions (semantic conv derived from structured list)
        if (
            isinstance(invocation, LLMInvocation)
            and invocation.request_functions
        ):
            for idx, fn in enumerate(invocation.request_functions):
                name = fn.get("name")
                if name:
                    span.set_attribute(
                        f"gen_ai.request.function.{idx}.name", name
                    )
                desc = fn.get("description")
                if desc:
                    span.set_attribute(
                        f"gen_ai.request.function.{idx}.description", desc
                    )
                params = fn.get("parameters")
                if params is not None:
                    span.set_attribute(
                        f"gen_ai.request.function.{idx}.parameters",
                        str(params),
                    )
        # Backward compatibility: copy non-semconv, non-traceloop attributes present at start
        if isinstance(invocation, LLMInvocation):
            for k, v in invocation.attributes.items():
                if k.startswith("gen_ai.") or k.startswith("traceloop."):
                    continue
                try:
                    span.set_attribute(k, v)
                except Exception:  # pragma: no cover
                    pass

    def _apply_finish_attrs(
        self, invocation: LLMInvocation | EmbeddingInvocation
    ):
        span = getattr(invocation, "span", None)
        if span is None:
            return
        # Backfill input messages if capture was enabled late (e.g., refresh after span start)
        if (
            self._capture_content
            and isinstance(invocation, LLMInvocation)
            and GEN_AI_INPUT_MESSAGES not in span.attributes  # type: ignore[attr-defined]
            and invocation.input_messages
        ):
            serialized_in = self._serialize_messages(invocation.input_messages)
            if serialized_in is not None:
                span.set_attribute(GEN_AI_INPUT_MESSAGES, serialized_in)
        # Finish-time semconv attributes (response + usage tokens)
        if isinstance(invocation, LLMInvocation):
            if invocation.response_model_name:
                span.set_attribute(
                    GenAI.GEN_AI_RESPONSE_MODEL, invocation.response_model_name
                )
            if invocation.response_id:
                span.set_attribute(
                    GenAI.GEN_AI_RESPONSE_ID, invocation.response_id
                )
            if invocation.input_tokens is not None:
                span.set_attribute(
                    GenAI.GEN_AI_USAGE_INPUT_TOKENS, invocation.input_tokens
                )
            if invocation.output_tokens is not None:
                span.set_attribute(
                    GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, invocation.output_tokens
                )
            # Re-apply function definitions if added late (idempotent)
            if invocation.request_functions:
                for idx, fn in enumerate(invocation.request_functions):
                    name = fn.get("name")
                    if name:
                        span.set_attribute(
                            f"gen_ai.request.function.{idx}.name", name
                        )
                    desc = fn.get("description")
                    if desc:
                        span.set_attribute(
                            f"gen_ai.request.function.{idx}.description", desc
                        )
                    params = fn.get("parameters")
                    if params is not None:
                        span.set_attribute(
                            f"gen_ai.request.function.{idx}.parameters",
                            str(params),
                        )
            # Copy (or update) custom non-semconv, non-traceloop attributes added during invocation
            for k, v in invocation.attributes.items():
                if k.startswith("gen_ai.") or k.startswith("traceloop."):
                    continue
                try:
                    span.set_attribute(k, v)
                except Exception:  # pragma: no cover
                    pass
        if (
            self._capture_content
            and isinstance(invocation, LLMInvocation)
            and invocation.output_messages
        ):
            serialized = self._serialize_messages(invocation.output_messages)
            if serialized is not None:
                span.set_attribute(GEN_AI_OUTPUT_MESSAGES, serialized)

    # ---- lifecycle -------------------------------------------------------
    def start(self, invocation: LLMInvocation | EmbeddingInvocation) -> None:  # type: ignore[override]
        if isinstance(invocation, ToolCall):
            span_name = f"tool {invocation.name}"
        elif isinstance(invocation, EmbeddingInvocation):
            span_name = f"embedding {invocation.request_model}"
        else:
            span_name = f"chat {invocation.request_model}"
        cm = self._tracer.start_as_current_span(
            span_name, kind=SpanKind.CLIENT, end_on_exit=False
        )
        span = cm.__enter__()
        invocation.span = span  # type: ignore[assignment]
        invocation.context_token = cm  # type: ignore[assignment]
        self._apply_start_attrs(invocation)

    def finish(self, invocation: LLMInvocation | EmbeddingInvocation) -> None:  # type: ignore[override]
        span = getattr(invocation, "span", None)
        if span is None:
            return
        self._apply_finish_attrs(invocation)
        token = getattr(invocation, "context_token", None)
        if token is not None and hasattr(token, "__exit__"):
            try:  # pragma: no cover
                token.__exit__(None, None, None)  # type: ignore[misc]
            except Exception:  # pragma: no cover
                pass
        span.end()

    def error(
        self, error: Error, invocation: LLMInvocation | EmbeddingInvocation
    ) -> None:  # type: ignore[override]
        span = getattr(invocation, "span", None)
        if span is None:
            return
        span.set_status(Status(StatusCode.ERROR, error.message))
        if span.is_recording():
            span.set_attribute(
                ErrorAttributes.ERROR_TYPE, error.type.__qualname__
            )
        self._apply_finish_attrs(invocation)
        token = getattr(invocation, "context_token", None)
        if token is not None and hasattr(token, "__exit__"):
            try:  # pragma: no cover
                token.__exit__(None, None, None)  # type: ignore[misc]
            except Exception:  # pragma: no cover
                pass
        span.end()
