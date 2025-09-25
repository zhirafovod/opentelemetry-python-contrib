"""
Generic base dataclass for GenAI telemetry invocations.

This type is intended for use with telemetry generators and handlers, and can be
subclassed or extended for specific GenAI data types (e.g., LLMInvocation).
"""

from contextvars import Token
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from opentelemetry.context import Context
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue


@dataclass
class GenAI:
    """
    Generic base class for GenAI invocation data used in telemetry generation.
    Includes common fields required for telemetry: model, span, timing, attributes, and context.
    """

    request_model: str
    span: Optional[Span] = None
    start_time: float = field(default=0.0)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    context_token: Optional[Token[Context]] = None


# Move LLMInvocation and related message types here from __init__.py


class ContentCapturingMode(Enum):
    NO_CONTENT = 0
    SPAN_ONLY = 1
    EVENT_ONLY = 2
    SPAN_AND_EVENT = 3


@dataclass()
class ToolCall:
    arguments: Any
    name: str
    id: Optional[str]
    type: Literal["tool_call"] = "tool_call"


@dataclass()
class ToolCallResponse:
    response: Any
    id: Optional[str]
    type: Literal["tool_call_response"] = "tool_call_response"


FinishReason = Literal[
    "content_filter", "error", "length", "stop", "tool_calls"
]


@dataclass()
class Text:
    content: str
    type: Literal["text"] = "text"


MessagePart = Union[Text, ToolCall, ToolCallResponse, Any]


@dataclass()
class InputMessage:
    role: str
    parts: list[MessagePart]


@dataclass()
class OutputMessage:
    role: str
    parts: list[MessagePart]
    finish_reason: Union[str, FinishReason]


def _new_input_messages() -> List[InputMessage]:
    return []


def _new_output_messages() -> List[OutputMessage]:
    return []


def _new_str_any_dict() -> Dict[str, Any]:
    return {}


@dataclass
class LLMInvocation(GenAI):
    """
    Represents a single LLM call invocation. When creating an LLMInvocation object,
    only update the data attributes. The span and context_token attributes are
    set by the TelemetryHandler.
    """

    input_messages: List[InputMessage] = field(
        default_factory=_new_input_messages
    )
    output_messages: List[OutputMessage] = field(
        default_factory=_new_output_messages
    )
    provider: Optional[str] = None
    response_model_name: Optional[str] = None
    response_id: Optional[str] = None
    input_tokens: Optional[AttributeValue] = None
    output_tokens: Optional[AttributeValue] = None
