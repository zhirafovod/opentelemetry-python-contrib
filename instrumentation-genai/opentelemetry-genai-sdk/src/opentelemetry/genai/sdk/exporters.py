from opentelemetry import trace, metrics
from opentelemetry.trace import Tracer
from opentelemetry.metrics import Meter
from .types import LLMInvocation


class BaseExporter:
    """
    Abstract base for exporters mapping GenAI types -> OpenTelemetry.
    """
    def export(self, invocation: LLMInvocation):
        raise NotImplementedError


class SpanMetricEventExporter(BaseExporter):
    """
    Emits spans, metrics and events for a full telemetry picture.
    """
    def __init__(self, tracer: Tracer = None, meter: Meter = None):
        self.tracer = tracer or trace.get_tracer(__name__)
        self.meter = meter or metrics.get_meter(__name__)

    def export(self, invocation: LLMInvocation):
        # Start span
        with self.tracer.start_as_current_span(
            "llm.invocation",
            attributes={
                "model.name": invocation.model_name,
                **invocation.attributes
            }
        ) as span:
            # record timing
            span.set_attribute("duration.ms", (invocation.end_time - invocation.start_time).total_seconds() * 1000)

            # record response as event
            span.add_event(
                "llm.response",
                {"text": invocation.response}
            )

        # record a metric
        histogram = self.meter.create_histogram(
            "llm.invocation.duration_ms"
        )
        histogram.record(
            (invocation.end_time - invocation.start_time).total_seconds() * 1000,
            attributes={"model.name": invocation.model_name}
        )


class SpanMetricExporter(BaseExporter):
    """
    Emits only spans and metrics (no events).
    """
    def __init__(self, tracer: Tracer = None, meter: Meter = None):
        self.tracer = tracer or trace.get_tracer(__name__)
        self.meter = meter or metrics.get_meter(__name__)

    def export(self, invocation: LLMInvocation):
        with self.tracer.start_as_current_span(
            "llm.invocation",
            attributes={"model.name": invocation.model_name}
        ) as span:
            span.set_attribute("duration.ms", (invocation.end_time - invocation.start_time).total_seconds() * 1000)

        counter = self.meter.create_counter(
            "llm.invocations.count"
        )
        counter.add(1, attributes={"model.name": invocation.model_name})
