#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from opentelemetry.util.genai.processors.traceloop_span_processor import TraceloopSpanProcessor

RULE_SPEC = {
    "rules": [
        {
            # NOTE: In Python dicts, duplicate keys are overwritten. The earlier
            # version used two separate "rename" entries so only the last one
            # survived. Combine them into a single mapping and optionally
            # remove noisy attributes.
            "attribute_transformations": {
                "rename": {
                    "traceloop.entity.input": "gen_ai.input.messages",
                    "traceloop.entity.output": "gen_ai.output.messages",
                },
                # Demonstrate removal (uncomment to test):
                # "remove": ["debug_info"],
            },
            "name_transformations": {"chat *": "genai.chat"},
        }
    ]
}
os.environ["OTEL_GENAI_SPAN_TRANSFORM_RULES"] = json.dumps(RULE_SPEC)

# Set up tracing
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
# Add the Traceloop processor
provider.add_span_processor(TraceloopSpanProcessor())
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

print("Creating spans ...\n")

with tracer.start_as_current_span("chat gpt-4") as span:
    span.set_attribute("traceloop.entity.input", "some data")
    span.set_attribute("debug_info", "remove me if rule had remove")

with tracer.start_as_current_span("vector encode") as span:
    span.set_attribute("custom.kind", "embedding")
