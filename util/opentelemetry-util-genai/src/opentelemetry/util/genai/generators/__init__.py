from .base_generator import BaseTelemetryGenerator
from .span_metric_event_generator import SpanMetricEventGenerator
from .span_metric_generator import SpanMetricGenerator

# Alias for backwards compatibility / public API expected by handler
SpanGenerator = SpanMetricEventGenerator

__all__ = [
    "BaseTelemetryGenerator",
    "SpanMetricEventGenerator",
    "SpanMetricGenerator",
    "SpanGenerator",
]
