#!/usr/bin/env python3

from __future__ import annotations

"""Example: Emitting Traceloop-compatible spans and translating legacy attributes.

This example shows how to enable the external Traceloop compatibility emitter
(`traceloop_compat`) alongside standard semantic convention spans. The legacy
TraceloopSpanProcessor & transformation rules have been removed.

Prerequisites:
    pip install opentelemetry-util-genai-emitters-traceloop

Environment (basic – compat only):
    export OTEL_INSTRUMENTATION_GENAI_EMITTERS=traceloop_compat

Environment (semantic + compat + translator promotion – simple flag):
        export OTEL_GENAI_ENABLE_TRACELOOP_TRANSLATOR=1
        export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span,traceloop_compat

Alternative (explicit token if registered via entry point):
        export OTEL_INSTRUMENTATION_GENAI_EMITTERS=span,traceloop_translator,traceloop_compat
    (If ordering needs enforcement you can use category override, e.g.
        export OTEL_INSTRUMENTATION_GENAI_EMITTERS_SPAN=prepend:TraceloopTranslator )

Optional: capture message content (both span + event):
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES=both

Run this example to see two spans per invocation: the semconv span and the
Traceloop-compatible span.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter,
)

from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    LLMInvocation,
    InputMessage,
    OutputMessage,
    Text,
)


def run_example():
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    # Build a telemetry handler (singleton) – emitters are chosen via env vars
    handler = get_telemetry_handler(tracer_provider=provider)

    # Include a few illustrative Traceloop-style attributes.
    # These will be mapped/prefixed automatically by the Traceloop compat emitter.
    invocation = LLMInvocation(
        request_model="gpt-4",
        input_messages=[InputMessage(role="user", parts=[Text("Hello")])],
        attributes={
            "custom.attribute": "value",  # arbitrary user attribute
            "traceloop.entity.name": "ChatLLM",
            "traceloop.workflow.name": "main_flow",
            "traceloop.entity.path": "root/branch/leaf",
            "traceloop.entity.input": "Hi"
        },
    )

    handler.start_llm(invocation)
    # Simulate model output
    invocation.output_messages = [
        OutputMessage(
            role="assistant", parts=[Text("Hi there!")], finish_reason="stop"
        )
    ]
    handler.stop_llm(invocation)

    print("\nInvocation complete. Check exporter output above for:"
        "\n  * SemanticConvention span containing promoted gen_ai.* keys"
        "\n  * Traceloop compat span (legacy format)"
        "\nIf translator emitter enabled, attributes like gen_ai.agent.name should be present.\n")


if __name__ == "__main__":
    run_example()
