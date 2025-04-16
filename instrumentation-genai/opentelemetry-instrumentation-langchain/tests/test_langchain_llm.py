import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import pytest
from typing import Optional

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    event_attributes as EventAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics


# span_exporter, log_exporter, openai_client, instrument_no_content are coming from
# fixtures deinfed in conftest.py
@pytest.mark.vcr()
def test_langchain_call(
    span_exporter, log_exporter, openai_client, instrument_no_content
):
    llm_model_value = "gpt-3.5-turbo"
    llm = ChatOpenAI(model=llm_model_value)

    messages = [
        SystemMessage(content="You are a helpful assistant!"),
        HumanMessage(content="What is the capital of France?"),
    ]

    response = llm.invoke(messages)
    assert response.content == "The capital of France is Paris."

    spans = span_exporter.get_finished_spans()
    print(f"spans: {spans}")
    for span in spans:
        print(f"span: {span}")
        print(f"span attributes: {span.attributes}")
    # TODO: fix the code and ensure the assertions are correct
    assert_completion_attributes(spans[0], llm_model_value, response)

    logs = log_exporter.get_finished_logs()
    print(f"logs: {logs}")
    # TODO: fix the cod to ensure we emit 2 correct logs
    # assert len(logs) == 2

    # TODO: metrics - we should get 2 metrics


### Utils ###
# TDDO: modify to do the correct assertion. This is a copy paste from the openai
# We should get the same telemetry from the openai and langchain, with the only difference that
# gen_ai.system is either 'openai' or 'langchain'
def assert_completion_attributes(
    span: ReadableSpan,
    request_model: str,
    response: Optional,
    operation_name: str = "chat",
):
    return assert_all_attributes(
        span,
        request_model,
        response.response_metadata.get("model_name"),
        response.response_metadata.get("token_usage").get("prompt_tokens"),
        response.response_metadata.get("token_usage").get("completion_tokens"),
        operation_name,
    )

# this is a sample assertion copied from openai
def assert_all_attributes(
    span: ReadableSpan,
    request_model: str,
    response_model: str = "gpt-3.5-turbo-0125",
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    operation_name: str = "chat",
    span_name: str = "ChatOpenAI.chat",
    system: str = "langchain",
):
    assert span.name == span_name

    assert operation_name == span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]

    assert system == span.attributes[GenAIAttributes.GEN_AI_SYSTEM]

    assert request_model == "gpt-3.5-turbo"

    assert response_model == "gpt-3.5-turbo-0125"

    assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes

    if input_tokens:
        assert (
            input_tokens
            == span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        )
    else:
        assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in span.attributes

    if output_tokens:
        assert (
            output_tokens
            == span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        )
    else:
        assert (
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in span.attributes
        )
