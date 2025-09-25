from __future__ import annotations

from typing import Optional

from opentelemetry.metrics import Histogram, Meter, get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

from ..generators.utils import (
    _get_metric_attributes,
    _record_duration,
    _record_token_metrics,
)
from ..instruments import Instruments
from ..types import Error, LLMInvocation


class MetricsEmitter:
    """Emits GenAI metrics (duration + token usage).

    Role: metric
    Ordering: runs after span.start (no-op) and before span.finish (records metrics).
    """

    role = "metric"
    name = "semconv_metrics"

    def __init__(self, meter: Optional[Meter] = None):
        _meter: Meter = meter or get_meter(__name__)
        instruments = Instruments(_meter)
        self._duration_histogram: Histogram = (
            instruments.operation_duration_histogram
        )
        self._token_histogram: Histogram = instruments.token_usage_histogram

    # Lifecycle API --------------------------------------------------------
    def start(self, invocation: LLMInvocation) -> None:  # no-op
        return None

    def finish(self, invocation: LLMInvocation) -> None:
        metric_attrs = _get_metric_attributes(
            invocation.request_model,
            invocation.response_model_name,
            GenAI.GenAiOperationNameValues.CHAT.value,
            invocation.provider,
            invocation.attributes.get("framework"),
        )
        # Record tokens only on success (error indicated by missing end_time? span status? we assume caller sets error separately)
        _record_token_metrics(
            self._token_histogram,
            invocation.input_tokens,
            invocation.output_tokens,
            metric_attrs,
        )
        _record_duration(self._duration_histogram, invocation, metric_attrs)

    def error(self, error: Error, invocation: LLMInvocation) -> None:
        # On error record only duration (if any)
        metric_attrs = _get_metric_attributes(
            invocation.request_model,
            invocation.response_model_name,
            GenAI.GenAiOperationNameValues.CHAT.value,
            invocation.provider,
            invocation.attributes.get("framework"),
        )
        _record_duration(self._duration_histogram, invocation, metric_attrs)
