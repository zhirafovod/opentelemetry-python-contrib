#!/usr/bin/env python3

from __future__ import annotations

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
            "hello.attribute": "value",
            "traceloop.entity.name": "ChatLLM",
            "traceloop.workflow.name": "main_flow",
            "traceloop.entity.path": "root/branch/leaf",
            "traceloop.entity.input": [
                {"role": "user", "content": "Hi"}
            ],
        },
    )



    handler.start_llm(invocation)
    # Simulate model output
    invocation.output_messages = [
        OutputMessage(
            role="assistant", parts=[Text("Hi there!")], finish_reason="stop"
        )
    ]
    print("Before stop:", invocation.attributes)
    handler.stop_llm(invocation)


if __name__ == "__main__":
    run_example()