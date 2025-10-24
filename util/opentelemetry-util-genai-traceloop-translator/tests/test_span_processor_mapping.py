import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from opentelemetry.util.genai.traceloop import enable_traceloop_translator


def test_traceloop_span_processor_maps_attributes():
    os.environ["OTEL_GENAI_CONTENT_CAPTURE"] = "1"  # enable content capture
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    
    # TraceloopSpanProcessor is automatically registered via _auto_enable()
    # which wraps trace.set_tracer_provider()

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("chat gpt-4") as span:
        span.set_attribute("traceloop.workflow.name", "flowA")
        span.set_attribute("traceloop.entity.name", "AgentA")
        span.set_attribute("traceloop.entity.path", "flowA/AgentA/step1")
        span.set_attribute("traceloop.entity.input", ["Hello"])  # will be normalized
        span.set_attribute("traceloop.span.kind", "workflow")

    spans = exporter.get_finished_spans()
    
    # With single processor instance, we should get exactly 2 spans:
    # 1. Original span with traceloop.* attributes
    # 2. Synthetic span with transformed gen_ai.* attributes
    assert len(spans) == 2, f"Expected 2 spans (original + synthetic), got {len(spans)}"
    
    # Verify we have the original span with traceloop.* attributes
    original_spans = [s for s in spans if s.attributes and "traceloop.workflow.name" in s.attributes]
    assert len(original_spans) == 1, "Should have exactly 1 original span with traceloop.* attributes"
    
    # Find the span with the transformed attributes (should be one of the synthetic spans)
    transformed_span = None
    for span in spans:
        if span.attributes and span.attributes.get("gen_ai.workflow.name") == "flowA":
            transformed_span = span
            break
    
    assert transformed_span is not None, "Should find a span with transformed gen_ai.* attributes"
    
    # Verify all mapped attributes are present
    assert transformed_span.attributes.get("gen_ai.workflow.name") == "flowA"
    assert transformed_span.attributes.get("gen_ai.agent.name") == "AgentA"
    assert transformed_span.attributes.get("gen_ai.workflow.path") == "flowA/AgentA/step1"
    assert transformed_span.attributes.get("gen_ai.span.kind") == "workflow"
    
    # Content should be mapped and normalized
    assert "gen_ai.input.messages" in transformed_span.attributes
    input_messages = transformed_span.attributes["gen_ai.input.messages"]
    assert "Hello" in input_messages  # Content should be preserved
    
    # Operation should be inferred
    assert transformed_span.attributes.get("gen_ai.operation.name") == "chat"
