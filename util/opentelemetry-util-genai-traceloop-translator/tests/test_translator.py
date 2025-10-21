import json
from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import LLMInvocation


def _handler(monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span,traceloop_translator")
    monkeypatch.setenv("OTEL_GENAI_CONTENT_CAPTURE", "1")  # enable content
    return get_telemetry_handler()


def test_normalizes_string_blob(monkeypatch):
    h = _handler(monkeypatch)
    blob = "{\"messages\":[{\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"content\":\"Hello\"}]}]}"
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.entity.input": blob})
    h.start_llm(inv)
    data = json.loads(inv.attributes.get("gen_ai.input.messages"))
    assert isinstance(data, list)
    assert data[0]["role"] == "user"
    assert data[0]["parts"][0]["type"] == "text"


def test_normalizes_inputs_dict(monkeypatch):
    h = _handler(monkeypatch)
    raw = json.dumps({"inputs": {"title": "Tragedy at sunset on the beach", "era": "Victorian England"}})
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.entity.input": raw})
    h.start_llm(inv)
    arr = json.loads(inv.attributes.get("gen_ai.input.messages"))
    assert arr[0]["role"] == "user"
    # content should include title key
    assert "Tragedy at sunset" in arr[0]["parts"][0]["content"]


def test_normalizes_list_of_strings(monkeypatch):
    h = _handler(monkeypatch)
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.entity.input": ["Hello", "World"]})
    h.start_llm(inv)
    arr = json.loads(inv.attributes.get("gen_ai.input.messages"))
    assert len(arr) == 2
    assert arr[0]["parts"][0]["content"] == "Hello"


def test_normalizes_dict_messages(monkeypatch):
    h = _handler(monkeypatch)
    raw = {"messages": [{"role": "user", "content": "Ping"}, {"role": "assistant", "content": "Pong"}]}
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.entity.input": raw})
    h.start_llm(inv)
    arr = json.loads(inv.attributes.get("gen_ai.input.messages"))
    assert arr[1]["role"] == "assistant"
    assert arr[1]["parts"][0]["content"] == "Pong"


def test_output_normalization(monkeypatch):
    h = _handler(monkeypatch)
    out_raw = [{"role": "assistant", "parts": ["Answer"], "finish_reason": "stop"}]
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.entity.output": out_raw})
    h.start_llm(inv)
    arr = json.loads(inv.attributes.get("gen_ai.output.messages"))
    assert arr[0]["finish_reason"] == "stop"


def test_output_openai_choices(monkeypatch):
    h = _handler(monkeypatch)
    raw = {"choices": [
        {"message": {"role": "assistant", "content": "Hello there"}, "finish_reason": "stop"},
        {"message": {"role": "assistant", "content": "Hi again"}, "finish_reason": "length"},
    ]}
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.entity.output": raw})
    h.start_llm(inv)
    arr = json.loads(inv.attributes.get("gen_ai.output.messages"))
    assert arr[1]["parts"][0]["content"] == "Hi again"
    assert arr[1]["finish_reason"] == "length"


def test_output_candidates(monkeypatch):
    h = _handler(monkeypatch)
    raw = {"candidates": [
        {"role": "assistant", "content": [{"text": "Choice A"}]},
        {"role": "assistant", "content": [{"text": "Choice B"}], "finish_reason": "stop"},
    ]}
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.entity.output": raw})
    h.start_llm(inv)
    arr = json.loads(inv.attributes.get("gen_ai.output.messages"))
    assert arr[0]["parts"][0]["content"].startswith("Choice A")


def test_output_tool_calls(monkeypatch):
    h = _handler(monkeypatch)
    raw = {"tool_calls": [
        {"name": "get_weather", "arguments": {"city": "Paris"}, "id": "call1"},
        {"name": "lookup_user", "arguments": {"id": 42}, "id": "call2", "finish_reason": "tool_call"},
    ]}
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.entity.output": raw})
    h.start_llm(inv)
    arr = json.loads(inv.attributes.get("gen_ai.output.messages"))
    assert arr[0]["parts"][0]["type"] == "tool_call"
    assert arr[1]["finish_reason"] == "tool_call"


def test_no_content_capture(monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span,traceloop_translator")
    monkeypatch.setenv("OTEL_GENAI_CONTENT_CAPTURE", "0")  # disable
    h = get_telemetry_handler()
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.entity.input": "Hello"})
    h.start_llm(inv)
    assert "gen_ai.input.messages" not in inv.attributes



def test_workflow_name_mapping(monkeypatch):
    h = _handler(monkeypatch)
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.workflow.name": "FlowA"})
    h.start_llm(inv)
    assert inv.attributes.get("gen_ai.workflow.name") == "FlowA"


def test_strip_legacy(monkeypatch):
    h = _handler(monkeypatch)
    monkeypatch.setenv("OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY", "1")
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.entity.path": "x/y/z"})
    h.start_llm(inv)
    assert inv.attributes.get("gen_ai.workflow.path") == "x/y/z"
    assert "traceloop.entity.path" not in inv.attributes


def test_conversation_id_mapping(monkeypatch):
    h = _handler(monkeypatch)
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.correlation.id": "conv_123"})
    h.start_llm(inv)
    assert inv.attributes.get("gen_ai.conversation.id") == "conv_123"


def test_conversation_id_invalid(monkeypatch):
    h = _handler(monkeypatch)
    bad = "this id has spaces"  # fails regex
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.correlation.id": bad})
    h.start_llm(inv)
    assert inv.attributes.get("gen_ai.conversation.id") is None


def test_operation_inference(monkeypatch):
    h = _handler(monkeypatch)
    inv = LLMInvocation(request_model="gpt-4", input_messages=[], attributes={"traceloop.span.kind": "tool"})
    h.start_llm(inv)
    assert inv.attributes.get("gen_ai.operation.name") == "execute_tool"