from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import LLMInvocation, InputMessage, Text


def test_basic_promotion(monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span,traceloop_translator")
    handler = get_telemetry_handler()
    inv = LLMInvocation(
        request_model="gpt-4",
        input_messages=[InputMessage(role="user", parts=[Text("Hi")])],
        attributes={
            "traceloop.workflow.name": "flowX",
            "traceloop.entity.name": "AgentX",
            "traceloop.callback.name": "root_cb",
        },
    )
    handler.start_llm(inv)
    handler.stop_llm(inv)
    assert inv.attributes.get("gen_ai.workflow.name") == "flowX"
    assert inv.attributes.get("gen_ai.agent.name") == "AgentX"
