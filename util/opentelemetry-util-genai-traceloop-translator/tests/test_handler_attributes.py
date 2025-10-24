"""Test that the telemetry handler correctly sets custom attributes."""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import LLMInvocation


def test_handler_sets_custom_attributes():
    """Test that custom gen_ai.* attributes can be set via the handler."""
    os.environ["OTEL_GENAI_CONTENT_CAPTURE"] = "1"
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    handler = get_telemetry_handler()
    invocation = LLMInvocation(
        request_model="gpt-4",
        attributes={
            "gen_ai.workflow.name": "test_workflow",
            "gen_ai.agent.name": "test_agent",
            "gen_ai.workflow.path": "test/path",
            "custom.attribute": "custom_value",
        },
    )

    handler.start_llm(invocation)
    handler.stop_llm(invocation)

    spans = exporter.get_finished_spans()
    print(f"\nTotal spans: {len(spans)}")
    for i, s in enumerate(spans):
        print(f"Span {i + 1}: {s.name}")
        print(f"  Attributes: {dict(s.attributes)}")

    # Find the span with our custom attributes
    target_span = None
    for span in spans:
        if (
            span.attributes
            and span.attributes.get("gen_ai.workflow.name") == "test_workflow"
        ):
            target_span = span
            break

    assert target_span is not None, (
        "Should find a span with gen_ai.workflow.name"
    )

    # Check if custom attributes are present
    assert (
        target_span.attributes.get("gen_ai.workflow.name") == "test_workflow"
    )
    assert target_span.attributes.get("gen_ai.agent.name") == "test_agent"
    assert target_span.attributes.get("gen_ai.workflow.path") == "test/path"
    assert target_span.attributes.get("custom.attribute") == "custom_value"
