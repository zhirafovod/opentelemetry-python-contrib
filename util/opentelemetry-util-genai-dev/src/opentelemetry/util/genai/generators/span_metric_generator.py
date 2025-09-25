# Copyright The OpenTelemetry Authors
# (file rewritten during composite refactor Phase 1 to remove legacy duplicate logic)
from __future__ import annotations

from typing import Optional

from opentelemetry import trace
from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode

from ..instruments import Instruments
from ..types import Error, LLMInvocation
from .base_span_generator import BaseSpanGenerator
from .utils import (
    _collect_finish_reasons,
    _get_metric_attributes,
    _maybe_set_input_messages,
    _record_duration,
    _record_token_metrics,
    _set_chat_generation_attrs,
    _set_response_and_usage_attributes,
)


class SpanMetricGenerator(BaseSpanGenerator):
    """Spans + metrics (no events)."""

    def __init__(
        self,
        tracer: Optional[Tracer] = None,
        meter: Optional[Meter] = None,
        capture_content: bool = False,
    ):
        super().__init__(
            tracer=tracer or trace.get_tracer(__name__),
            capture_content=capture_content,
        )
        _meter: Meter = meter or get_meter(__name__)
        instruments = Instruments(_meter)
        self._duration_histogram: Histogram = (
            instruments.operation_duration_histogram
        )
        self._token_histogram: Histogram = instruments.token_usage_histogram

    def _on_before_end(
        self, invocation: LLMInvocation, error: Optional[Error]
    ):  # type: ignore[override]
        span = invocation.span
        if span is None:
            return
        messages = (
            getattr(invocation, "messages", None) or invocation.input_messages
        )
        chat_generations = (
            getattr(invocation, "chat_generations", None)
            or invocation.output_messages
        )
        try:
            setattr(invocation, "messages", messages)
            setattr(invocation, "chat_generations", chat_generations)
        except Exception:  # pragma: no cover
            pass
        if error is None:
            finish_reasons = _collect_finish_reasons(chat_generations)
            if finish_reasons:
                span.set_attribute(
                    GenAI.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons
                )
            _set_response_and_usage_attributes(
                span,
                invocation.response_model_name,
                invocation.response_id,
                invocation.input_tokens,
                invocation.output_tokens,
            )
            _maybe_set_input_messages(span, messages, self._capture_content)
            _set_chat_generation_attrs(span, chat_generations)
        else:
            span.set_attribute(
                ErrorAttributes.ERROR_TYPE, error.type.__qualname__
            )
        metric_attrs = _get_metric_attributes(
            invocation.request_model,
            invocation.response_model_name,
            GenAI.GenAiOperationNameValues.CHAT.value,
            invocation.provider,
            invocation.attributes.get("framework"),
        )
        if error is None:
            _record_token_metrics(
                self._token_histogram,
                invocation.input_tokens,
                invocation.output_tokens,
                metric_attrs,
            )
        _record_duration(self._duration_histogram, invocation, metric_attrs)

    def error(self, error: Error, invocation: LLMInvocation) -> None:  # type: ignore[override]
        span = invocation.span
        if span is None:
            self.start(invocation)
            span = invocation.span
        if span is None:
            return
        span.set_status(Status(StatusCode.ERROR, error.message))
        self._on_before_end(invocation, error)
        if invocation.context_token is not None:
            try:
                invocation.context_token.__exit__(None, None, None)
            except Exception:  # pragma: no cover
                pass
        span.end()
