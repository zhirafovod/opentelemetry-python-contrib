from datetime import datetime
from .types import LLMInvocation, ToolInvocation


def llm_start(model_name: str, prompt: str, **attrs) -> LLMInvocation:
    """
    Begin tracking an LLM invocation.
    """
    invocation = LLMInvocation(
        model_name=model_name,
        prompt=prompt,
        attributes=attrs
    )
    return invocation


def llm_stop(invocation: LLMInvocation, response: str, **attrs) -> LLMInvocation:
    """
    Complete tracking of an LLM invocation, recording the response.
    """
    invocation.end_time = datetime.utcnow()
    invocation.response = response
    invocation.attributes.update(attrs)
    return invocation


def tool_start(tool_name: str, input: dict, **attrs) -> ToolInvocation:
    """
    Begin tracking a tool invocation.
    """
    invocation = ToolInvocation(
        tool_name=tool_name,
        input=input,
        attributes=attrs
    )
    return invocation


def tool_stop(invocation: ToolInvocation, output: dict, **attrs) -> ToolInvocation:
    """
    Complete tracking of a tool invocation, recording the output.
    """
    invocation.end_time = datetime.utcnow()
    invocation.output = output
    invocation.attributes.update(attrs)
    return invocation