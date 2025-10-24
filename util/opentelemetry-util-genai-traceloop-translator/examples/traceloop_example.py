#!/usr/bin/env python3

from __future__ import annotations

from opentelemetry import trace
import os
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
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env if available
# Traceloop imports for workflow annotation
try:
    from traceloop.sdk import Traceloop
    from traceloop.sdk.decorators import workflow

    # Initialize Traceloop (disable_batch so spans flush immediately for local demos)
    Traceloop.init(disable_batch=True, api_endpoint="http://localhost:4318")
except ImportError:
    print(
        "[traceloop] traceloop-sdk not installed. Run 'pip install traceloop-sdk' to enable workflow tracing."
    )


@workflow(name="llm_invocation_example")
def run_example(provider=None):
    llm = ChatOpenAI(
        model="gpt-4o",
        timeout=None,
        max_retries=2,
    )
    messages = [
        (
            "system",
            "You are a helpful assistant.",
        ),
        ("human", "what is the significance of 42?."),
    ]
    ai_msg = llm.invoke(messages)
    return ai_msg


if __name__ == "__main__":
    # Enable translator emitter + keep legacy keys for demonstration
    os.environ.setdefault(
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span,traceloop_translator"
    )
    os.environ.setdefault(
        "OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY", "false"
    )  # keep original traceloop.* to compare
    os.environ.setdefault(
        "OTEL_GENAI_CONTENT_CAPTURE", "1"
    )  # ensure input/output content mapping

    # Avoid overriding an existing SDK TracerProvider. Reuse if already configured.
    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        provider = existing
        # Attach a ConsoleSpanExporter for demo purposes (may duplicate if already added).
        try:
            provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )
        except Exception:
            pass
    else:
        # No SDK provider installed yet (likely the default stub); install one.
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        try:
            trace.set_tracer_provider(provider)
        except Exception:
            # If another component set a provider concurrently, fall back to that.
            provider = trace.get_tracer_provider()

    # Build a telemetry handler (singleton) – emitters are chosen via env vars
    handler = get_telemetry_handler(tracer_provider=provider)

    # Include a few illustrative Traceloop-style attributes.
    # These will be mapped/prefixed automatically by the Traceloop compat emitter.
    invocation = LLMInvocation(
        request_model="gpt-4",
        input_messages=[InputMessage(role="user", parts=[Text("Hello")])],
    )
    # Populate attributes after construction (avoids mismatch if constructor signature changes)
    invocation.attributes.update(
        {
            "traceloop.workflow.name": "demo_flow",  # workflow identifier
            "traceloop.entity.name": "ChatLLM",  # agent/entity name
            "traceloop.entity.path": "demo_flow/ChatLLM/step_1",  # hierarchical path
            "traceloop.entity.output": [  # raw input messages (will be normalized & mapped)
                {"role": "user", "content": "Hello"}
            ],
            "traceloop.span.kind": "workflow",  # helps infer gen_ai.operation.name
        }
    )

    print("Before start (raw attributes):", invocation.attributes)
    handler.start_llm(invocation)
    # Simulate model output
    invocation.output_messages = [
        OutputMessage(
            role="assistant", parts=[Text("Hi there!")], finish_reason="stop"
        )
    ]
    # # Add output content before stop for translator to map
    # invocation.attributes["traceloop.entity.output"] = [
    #     {"role": "assistant", "content": "Hi there!"}
    # ]
    handler.stop_llm(invocation)
    print("After stop (translated attributes):", invocation.attributes)

    run_example(provider=provider)
