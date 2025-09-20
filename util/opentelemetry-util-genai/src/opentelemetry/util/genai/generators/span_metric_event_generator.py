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

import json
from dataclasses import asdict
from typing import Dict, Optional
from uuid import UUID

from opentelemetry import trace
from opentelemetry._logs import Logger
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import (
    Span,
    SpanKind,
    Tracer,
    set_span_in_context,
    use_span,
)
from opentelemetry.trace.status import Status, StatusCode

from ..instruments import Instruments
from ..types import Error, LLMInvocation
from .base_generator import BaseTelemetryGenerator
from .utils import (
    _SpanState,
)


class SpanMetricEventGenerator(BaseTelemetryGenerator):
    """
    Generates spans, metrics, and events for a full telemetry picture.
    """

    def __init__(
        self,
        logger: Optional[Logger] = None,
        tracer: Optional[Tracer] = None,
        meter: Optional[Meter] = None,
        capture_content: bool = False,
    ):
        self._tracer: Tracer = tracer or trace.get_tracer(__name__)
        _meter: Meter = meter or get_meter(__name__)
        instruments = Instruments(_meter)
        self._duration_histogram: Histogram = (
            instruments.operation_duration_histogram
        )
        self._token_histogram: Histogram = instruments.token_usage_histogram
        self._logger: Optional[Logger] = logger
        self._capture_content: bool = capture_content
        # Internal map retained for possible future extensions; not used for parent/child logic now.
        self.spans: Dict[UUID, _SpanState] = {}

    def _start_span(
        self,
        name: str,
        kind: SpanKind,
        parent_run_id: Optional[UUID] = None,
    ) -> Span:
        if parent_run_id is not None and parent_run_id in self.spans:
            parent_span = self.spans[parent_run_id].span
            ctx = set_span_in_context(parent_span)
            span = self._tracer.start_span(name=name, kind=kind, context=ctx)
        else:
            span = self._tracer.start_span(name=name, kind=kind)
        return span

    def _end_span(self, run_id: UUID):
        state = self.spans[run_id]
        for child_id in state.children:
            child_state = self.spans.get(child_id)
            if child_state:
                child_state.span.end()
        state.span.end()

    def start(self, invocation: LLMInvocation):
        # Create span name pattern expected by tests: "chat {request_model}".
        span_name = f"chat {invocation.request_model}"
        span = self._tracer.start_span(name=span_name, kind=SpanKind.CLIENT)
        # Attach context so subsequently started spans (child invocations) become children automatically.
        cm = use_span(span, end_on_exit=False)
        cm.__enter__()
        # Reuse context_token field to hold context manager for later __exit__ call.
        invocation.context_token = cm
        invocation.span = span

        # Set initial semantic attributes.
        span.set_attribute(
            GenAI.GEN_AI_OPERATION_NAME,
            GenAI.GenAiOperationNameValues.CHAT.value,
        )
        span.set_attribute(
            GenAI.GEN_AI_REQUEST_MODEL, invocation.request_model
        )
        if invocation.provider:
            span.set_attribute("gen_ai.provider.name", invocation.provider)
        # Copy custom attributes present at start time.
        for k, v in invocation.attributes.items():
            span.set_attribute(k, v)

        # Capture input messages if enabled.
        if self._capture_content:
            try:
                input_messages_json = json.dumps(
                    [asdict(m) for m in invocation.input_messages]
                )
                span.set_attribute(
                    "gen_ai.input.messages", input_messages_json
                )
            except (
                Exception
            ):  # defensive; do not fail span creation on serialization issues
                pass

    def finish(self, invocation: LLMInvocation):
        span = invocation.span
        if span is None:
            # Fallback: create span if start was not called.
            span = self._tracer.start_span(
                name=f"chat {invocation.request_model}", kind=SpanKind.CLIENT
            )
            invocation.span = span

        # Update / set any new attributes added between start and finish.
        for k, v in invocation.attributes.items():
            span.set_attribute(k, v)

        # Capture output messages if enabled.
        if self._capture_content:
            try:
                output_messages_json = json.dumps(
                    [asdict(m) for m in invocation.output_messages]
                )
                span.set_attribute(
                    "gen_ai.output.messages", output_messages_json
                )
            except Exception:
                pass

        # Exit context manager first (if stored) so subsequent spans are not parented unexpectedly.
        if invocation.context_token is not None:
            try:
                invocation.context_token.__exit__(None, None, None)
            except Exception:
                pass
        # End span lifecycle.
        span.end()

    def error(self, error: Error, invocation: LLMInvocation):
        # Ensure span exists
        span = invocation.span
        if span is None:
            span = self._tracer.start_span(
                name=f"chat {invocation.request_model}", kind=SpanKind.CLIENT
            )
            invocation.span = span
        span.set_status(Status(StatusCode.ERROR, error.message))
        if span.is_recording():
            span.set_attribute(
                ErrorAttributes.ERROR_TYPE, error.type.__qualname__
            )
        # Copy latest attributes
        for k, v in invocation.attributes.items():
            span.set_attribute(k, v)
        if invocation.context_token is not None:
            try:
                invocation.context_token.__exit__(None, None, None)
            except Exception:
                pass
        span.end()
        # Record duration metric if end_time provided
        if invocation.end_time is not None:
            elapsed: float = invocation.end_time - invocation.start_time
            self._duration_histogram.record(elapsed, attributes={})
