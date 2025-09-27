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
"""Lightweight span-only telemetry emitter for GenAI invocations.

This implementation consolidates span lifecycle & attribute logic into a single class.
"""

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
from opentelemetry.trace import (
    SpanKind,
    Tracer,
)
from opentelemetry.trace.status import Status, StatusCode

from ..attributes import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
)
from ..types import EmbeddingInvocation, Error, LLMInvocation, ToolCall


class SpanGenerator:  # primary span emitter (backward compatible name)
    """Span-focused emitter supporting optional content capture.

    Replaces previous inheritance chain (BaseTelemetryGenerator → BaseSpanGenerator → SpanGenerator)
    with a single small implementation. Additional telemetry types (metrics, content events)
    are handled by separate emitters composed via CompositeGenerator.
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

    def handles(self, obj: any) -> bool:
        """Return True to allow span emitter to handle any invocation type."""
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
        # Determine operation name based on object type
        if isinstance(invocation, ToolCall):
            op_value = "tool_call"
        elif isinstance(invocation, EmbeddingInvocation):
            # Prefer semconv enum if present; otherwise fall back to string literal
            enum_val = getattr(
                GenAI.GenAiOperationNameValues, "EMBEDDING", None
            )
            op_value = enum_val.value if enum_val else "embedding"
        else:
            op_value = GenAI.GenAiOperationNameValues.CHAT.value
        span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, op_value)
        # Model name: for ToolCall, use invocation.name as request model
        model_name = (
            invocation.name
            if isinstance(invocation, ToolCall)
            else invocation.request_model
        )
        span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, model_name)
        provider = getattr(invocation, "provider", None)
        if provider:
            span.set_attribute(GEN_AI_PROVIDER_NAME, provider)
        for k, v in getattr(invocation, "attributes", {}).items():
            span.set_attribute(k, v)
        # Only LLMInvocation has input_messages
        if (
            self._capture_content
            and isinstance(invocation, LLMInvocation)
            and invocation.input_messages
        ):
            serialized = self._serialize_messages(invocation.input_messages)
            if serialized is not None:
                span.set_attribute(GEN_AI_INPUT_MESSAGES, serialized)

    def _apply_finish_attrs(
        self, invocation: LLMInvocation | EmbeddingInvocation
    ):
        span = getattr(invocation, "span", None)
        if span is None:
            return
        for k, v in getattr(invocation, "attributes", {}).items():
            span.set_attribute(k, v)
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
        # Determine span name by type
        if isinstance(invocation, ToolCall):
            span_name = f"tool {invocation.name}"
        elif isinstance(invocation, EmbeddingInvocation):
            span_name = f"embedding {invocation.request_model}"
        else:
            span_name = f"chat {invocation.request_model}"
        # Start span as current to automatically inherit parent
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


# Backward compatibility alias expected by imports and tests
class SpanEmitter(SpanGenerator):  # forward-looking alias class
    pass
