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

from typing import Dict, Optional
from uuid import UUID

from opentelemetry import trace
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
    _collect_finish_reasons,
    _get_metric_attributes,
    _maybe_set_input_messages,
    _record_duration,
    _record_token_metrics,
    _set_chat_generation_attrs,
    _set_initial_span_attributes,
    _set_response_and_usage_attributes,
    _SpanState,
)


class SpanMetricGenerator(BaseTelemetryGenerator):
    """
    Generates only spans and metrics (no events).
    """

    def __init__(
        self,
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
        self._capture_content: bool = capture_content
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
        if (
            invocation.parent_run_id is not None
            and invocation.parent_run_id in self.spans
        ):
            self.spans[invocation.parent_run_id].children.append(
                invocation.run_id
            )

    def finish(self, invocation: LLMInvocation):
        system = invocation.attributes.get("system")
        span = self._start_span(
            name=f"{system}.chat",
            kind=SpanKind.CLIENT,
            parent_run_id=invocation.parent_run_id,
        )

        with use_span(span, end_on_exit=False) as span:
            request_model = invocation.attributes.get("request_model")
            span_state = _SpanState(
                span=span,
                context=trace.get_current(),
                request_model=request_model,
                system=system,
                start_time=invocation.start_time,
            )
            self.spans[invocation.run_id] = span_state

            framework = invocation.attributes.get("framework")
            _set_initial_span_attributes(
                span, request_model, system, framework
            )

            finish_reasons = _collect_finish_reasons(
                invocation.chat_generations
            )
            if finish_reasons:
                span.set_attribute(
                    GenAI.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons
                )

            response_model = invocation.attributes.get("response_model_name")
            response_id = invocation.attributes.get("response_id")
            prompt_tokens = invocation.attributes.get("input_tokens")
            completion_tokens = invocation.attributes.get("output_tokens")
            _set_response_and_usage_attributes(
                span,
                response_model,
                response_id,
                prompt_tokens,
                completion_tokens,
            )

            _maybe_set_input_messages(
                span, invocation.messages, self._capture_content
            )
            _set_chat_generation_attrs(span, invocation.chat_generations)

            metric_attributes = _get_metric_attributes(
                request_model,
                response_model,
                GenAI.GenAiOperationNameValues.CHAT.value,
                system,
                framework,
            )
            _record_token_metrics(
                self._token_histogram,
                prompt_tokens,
                completion_tokens,
                metric_attributes,
            )

            self._end_span(invocation.run_id)
            _record_duration(
                self._duration_histogram, invocation, metric_attributes
            )

    def error(self, error: Error, invocation: LLMInvocation):
        system = invocation.attributes.get("system")
        span = self._start_span(
            name=f"{system}.chat",
            kind=SpanKind.CLIENT,
            parent_run_id=invocation.parent_run_id,
        )

        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            request_model = invocation.attributes.get("request_model")
            span_state = _SpanState(
                span=span,
                context=trace.get_current(),
                request_model=request_model,
                system=system,
                start_time=invocation.start_time,
            )
            self.spans[invocation.run_id] = span_state

            span.set_status(Status(StatusCode.ERROR, error.message))
            if span.is_recording():
                span.set_attribute(
                    ErrorAttributes.ERROR_TYPE, error.type.__qualname__
                )

            self._end_span(invocation.run_id)

            response_model = invocation.attributes.get("response_model_name")
            framework = invocation.attributes.get("framework")

            metric_attributes = _get_metric_attributes(
                request_model,
                response_model,
                GenAI.GenAiOperationNameValues.CHAT.value,
                system,
                framework,
            )

            if invocation.end_time is not None:
                elapsed: float = invocation.end_time - invocation.start_time
                self._duration_histogram.record(
                    elapsed, attributes=metric_attributes
                )
