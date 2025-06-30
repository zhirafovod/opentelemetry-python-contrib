from datetime import datetime
from threading import Lock
from .types import LLMInvocation, ToolInvocation

# Internal registries
_LLM_REGISTRY: dict[str, LLMInvocation] = {}
_TOOL_REGISTRY: dict[str, ToolInvocation] = {}
_registry_lock = Lock()


def llm_start(model_name: str, prompt: str, **attrs) -> str:
    """
    Begin tracking an LLM invocation. Returns a run_id.
    """
    invocation = LLMInvocation(
        model_name=model_name,
        prompt=prompt,
        attributes=attrs
    )
    with _registry_lock:
        _LLM_REGISTRY[invocation.invocation_id] = invocation
    return invocation.invocation_id


def llm_stop(run_id: str, response: str, **attrs) -> LLMInvocation:
    """
    Complete tracking of an LLM invocation, recording the response.
    """
    with _registry_lock:
        invocation = _LLM_REGISTRY.pop(run_id)
    invocation.end_time = datetime.utcnow()
    invocation.response = response
    invocation.attributes.update(attrs)
    return invocation


def llm_fail(run_id: str, error: str, **attrs) -> LLMInvocation:
    """
    Mark an LLM invocation as failed, recording an error.
    """
    with _registry_lock:
        invocation = _LLM_REGISTRY.pop(run_id)
    invocation.end_time = datetime.utcnow()
    invocation.attributes.update({"error": error, **attrs})
    return invocation


def tool_start(tool_name: str, input: dict, **attrs) -> str:
    """
    Begin tracking a tool invocation. Returns a run_id.
    """
    invocation = ToolInvocation(
        tool_name=tool_name,
        input=input,
        attributes=attrs
    )
    with _registry_lock:
        _TOOL_REGISTRY[invocation.invocation_id] = invocation
    return invocation.invocation_id


def tool_stop(run_id: str, output: dict, **attrs) -> ToolInvocation:
    """
    Complete tracking of a tool invocation, recording the output.
    """
    with _registry_lock:
        invocation = _TOOL_REGISTRY.pop(run_id)
    invocation.end_time = datetime.utcnow()
    invocation.output = output
    invocation.attributes.update(attrs)
    return invocation


def tool_fail(run_id: str, error: str, **attrs) -> ToolInvocation:
    """
    Mark a tool invocation as failed, recording an error.
    """
    with _registry_lock:
        invocation = _TOOL_REGISTRY.pop(run_id)
    invocation.end_time = datetime.utcnow()
    invocation.attributes.update({"error": error, **attrs})
    return invocation
