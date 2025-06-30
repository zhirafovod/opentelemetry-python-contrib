import pytest
from datetime import datetime, timedelta
from opentelemetry.genai.sdk.types import LLMInvocation, ToolInvocation
from opentelemetry.genai.sdk.api import llm_start, llm_stop, tool_start, tool_stop
from opentelemetry.genai.sdk.evals import get_evaluator, EvaluationResult
from opentelemetry.genai.sdk.exporters import SpanMetricEventExporter, SpanMetricExporter

class DummyLLM(LLMInvocation):
    pass

@pytest.fixture
def sample_llm_invocation():
    inv = llm_start("test-model", "hello world", custom_attr="value")
    # simulate processing time
    inv.start_time = datetime.utcnow() - timedelta(seconds=1)
    inv = llm_stop(inv, response="hello back", extra="info")
    return inv

@pytest.fixture
def sample_tool_invocation():
    inv = tool_start("test-tool", {"input": 123}, flag=True)
    inv.start_time = datetime.utcnow() - timedelta(milliseconds=500)
    inv = tool_stop(inv, {"output": "ok"}, status="done")
    return inv

def test_llm_start_and_stop(sample_llm_invocation):
    inv = sample_llm_invocation
    assert inv.model_name == "test-model"
    assert inv.prompt == "hello world"
    assert inv.response == "hello back"
    assert inv.attributes.get("custom_attr") == "value"
    assert inv.attributes.get("extra") == "info"
    assert inv.end_time >= inv.start_time

def test_tool_start_and_stop(sample_tool_invocation):
    inv = sample_tool_invocation
    assert inv.tool_name == "test-tool"
    assert inv.input == {"input": 123}
    assert inv.output == {"output": "ok"}
    assert inv.attributes.get("flag") is True
    assert inv.attributes.get("status") == "done"
    assert inv.end_time >= inv.start_time

@pytest.mark.parametrize("name,expected_method", [
    ("deepevals", "deepevals"),
    ("openlit", "openlit"),
])
def test_evaluator_factory_and_evaluate(name, expected_method, sample_llm_invocation):
    evaluator = get_evaluator(name)
    result = evaluator.evaluate(sample_llm_invocation)
    assert isinstance(result, EvaluationResult)
    assert result.details.get("method") == expected_method

def test_exporters_do_not_raise(sample_llm_invocation):
    full_exporter = SpanMetricEventExporter()
    span_exporter = SpanMetricExporter()
    # Should not raise
    full_exporter.export(sample_llm_invocation)
    span_exporter.export(sample_llm_invocation)