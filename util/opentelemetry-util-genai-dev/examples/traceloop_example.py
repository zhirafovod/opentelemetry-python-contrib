#!/usr/bin/env python3
"""
Traceloop Span Transformation Examples
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.util.genai.processors import TraceloopSpanProcessor


def example_automatic_processing():
    """Example: Automatic span processing with TraceloopSpanProcessor."""

    # Set up tracer provider
    provider = TracerProvider()

    # Add TraceloopSpanProcessor - transforms ALL matching spans automatically
    traceloop_processor = TraceloopSpanProcessor(
        attribute_transformations={
            "remove": ["debug_info", "internal_id"],
            "rename": {
                "model_ver": "ai.model.version",
                "llm.provider": "ai.system.vendor",
            },
            "add": {"service.name": "my-llm-service"},
        }
    )
    provider.add_span_processor(traceloop_processor)

    # Add console exporter to see results
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    # Create spans - they get automatically transformed!
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("chat gpt-4") as span:
        span.set_attribute(
            "model_ver", "1.0"
        )  # Will be renamed to ai.model.version
        span.set_attribute(
            "llm.provider", "openai"
        )  # Will be renamed to ai.system.vendor
        span.set_attribute("debug_info", "remove_me")  # Will be removed
        print("Span automatically transformed when it ends!")

    print("Automatic processing complete\n")


def example_simple_setup():
    """Example: Minimal setup for common use case."""
    print("=== Simple Setup ===")

    # Minimal setup - just add the processor with basic rules
    provider = TracerProvider()

    processor = TraceloopSpanProcessor(
        attribute_transformations={"add": {"service.name": "my-ai-service"}},
        traceloop_attributes={"traceloop.entity.name": "AI-Service"},
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    print("TraceloopSpanProcessor added - all AI spans will be transformed!")
    print("Simple setup complete\n")


if __name__ == "__main__":
    print("Traceloop Span Transformation Examples\n")

    # Show automatic processing (recommended approach)
    example_automatic_processing()

    # Show minimal setup
    example_simple_setup()

    print("All examples complete!")
