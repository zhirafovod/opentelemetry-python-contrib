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
"""Lightweight span-only telemetry generator for GenAI invocations.

Responsibilities:
- Create a CLIENT span named "chat {request_model}".
- Apply core GenAI semantic convention attributes.
- Optionally capture input / output messages (when capture_content enabled).
- Finalize the span (success or error) with appropriate attributes.

This class intentionally does NOT record metrics or emit log events.
For richer telemetry use SpanMetricGenerator or SpanMetricEventGenerator.
"""

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
from opentelemetry.trace import SpanKind, Tracer, use_span
from opentelemetry.trace.status import Status, StatusCode

from ..types import Error, LLMInvocation
from .base_generator import BaseTelemetryGenerator


class SpanGenerator(BaseTelemetryGenerator):
    def __init__(
        self, tracer: Optional[Tracer] = None, capture_content: bool = False
    ):
        self._tracer: Tracer = tracer or trace.get_tracer(__name__)
        self._capture_content = capture_content

    # --- internal helpers -------------------------------------------------
    def _serialize_messages(
        self, messages
    ):  # list[InputMessage|OutputMessage]
        try:
            return json.dumps([asdict(m) for m in messages])
        except (
            Exception
        ):  # defensive; don't break span on serialization issues
            return None

    def _set_common_start_attrs(self, span, invocation: LLMInvocation):
        span.set_attribute(
            GenAI.GEN_AI_OPERATION_NAME,
            GenAI.GenAiOperationNameValues.CHAT.value,
        )
        span.set_attribute(
            GenAI.GEN_AI_REQUEST_MODEL, invocation.request_model
        )
        if invocation.provider:
            span.set_attribute("gen_ai.provider.name", invocation.provider)
        for k, v in invocation.attributes.items():
            span.set_attribute(k, v)
        if self._capture_content and invocation.input_messages:
            serialized = self._serialize_messages(invocation.input_messages)
            if serialized is not None:
                span.set_attribute("gen_ai.input.messages", serialized)

    def _set_finish_attrs(self, span, invocation: LLMInvocation):
        # Include any new attributes
        for k, v in invocation.attributes.items():
            span.set_attribute(k, v)
        if self._capture_content and invocation.output_messages:
            serialized = self._serialize_messages(invocation.output_messages)
            if serialized is not None:
                span.set_attribute("gen_ai.output.messages", serialized)

    # --- public API -------------------------------------------------------
    def start(self, invocation: LLMInvocation) -> None:  # type: ignore[override]
        span_name = f"chat {invocation.request_model}"
        span = self._tracer.start_span(name=span_name, kind=SpanKind.CLIENT)
        invocation.span = span
        # Keep span active for potential child spans until finish()/error()
        cm = use_span(span, end_on_exit=False)
        cm.__enter__()
        invocation.context_token = cm  # store context manager for later exit
        self._set_common_start_attrs(span, invocation)

    def finish(self, invocation: LLMInvocation) -> None:  # type: ignore[override]
        span = invocation.span
        if span is None:
            return
        self._set_finish_attrs(span, invocation)
        # Exit context & end span
        if invocation.context_token is not None:
            try:
                invocation.context_token.__exit__(None, None, None)
            except Exception:
                pass
        span.end()

    def error(self, error: Error, invocation: LLMInvocation) -> None:  # type: ignore[override]
        span = invocation.span
        if span is None:
            return
        span.set_status(Status(StatusCode.ERROR, error.message))
        if span.is_recording():
            span.set_attribute(
                ErrorAttributes.ERROR_TYPE, error.type.__qualname__
            )
        self._set_finish_attrs(span, invocation)
        if invocation.context_token is not None:
            try:
                invocation.context_token.__exit__(None, None, None)
            except Exception:
                pass
        span.end()
