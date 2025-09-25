"""
Minimal example: Using opentelemetry-util-genai with a mock LLM
"""

import sys
import traceback

from dotenv import load_dotenv
from simple_llm import SimpleLLM

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util.genai.generators.semconv_generator import (
    SemConvGenerator,
)
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    Text,
)

# Load .env file before any OpenTelemetry imports
load_dotenv()
# Setup
llm = SimpleLLM()
handler = get_telemetry_handler()
generator = SemConvGenerator()

# Configure tracing
trace.set_tracer_provider(TracerProvider())
span_processor = BatchSpanProcessor(OTLPSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, world!"

try:
    # Start telemetry
    with handler.llm() as invocation:
        invocation.request_model = llm.name
        invocation.input_messages = [
            InputMessage(role="user", parts=[Text(content=prompt)])
        ]
        invocation.provider = "mock-provider"
        invocation.attributes = {"example": True}
        # Populate outputs and any additional attributes
        output = llm.generate(prompt)
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[Text(content=output)],
                finish_reason="stop",
            )
        ]
        print(f"LLM output: {output}")
except Exception as e:
    # Error handling and telemetry
    handler.fail_llm(invocation, e)
    print("LLM call failed:", e)
    traceback.print_exc()
