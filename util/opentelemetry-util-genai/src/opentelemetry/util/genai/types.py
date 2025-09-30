# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
from contextvars import Token
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, Union

from typing_extensions import TypeAlias

from opentelemetry.context import Context
from opentelemetry.trace import Span

ContextToken: TypeAlias = Token[Context]


class ContentCapturingMode(Enum):
    # Do not capture content (default).
    NO_CONTENT = 0
    # Only capture content in spans.
    SPAN_ONLY = 1
    # Only capture content in events.
    EVENT_ONLY = 2
    # Capture content in both spans and events.
    SPAN_AND_EVENT = 3


@dataclass(frozen=True)
class ToolCall:
    arguments: Any
    name: str
    id: Optional[str]
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True)
class ToolCallResponse:
    response: Any
    id: Optional[str]
    type: Literal["tool_call_response"] = "tool_call_response"


FinishReason = Literal[
    "content_filter", "error", "length", "stop", "tool_calls"
]


@dataclass(frozen=True)
class Text:
    content: str
    type: Literal["text"] = "text"


MessagePart = Union[Text, ToolCall, ToolCallResponse, Any]


@dataclass(frozen=True)
class InputMessage:
    role: str
    parts: list[MessagePart]


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class LLMRequest:
    """
    Immutable request data for an LLM invocation.
    Contains all the input parameters and configuration for the LLM call.
    """
    request_model: str
    input_messages: List[InputMessage] = field(default_factory=_new_input_messages)
    provider: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=_new_str_any_dict)


@dataclass(frozen=True)
class LLMResponse:
    """
    Immutable response data from an LLM invocation.
    Contains the outputs and metadata from the LLM call.
    """
    output_messages: List[OutputMessage] = field(default_factory=_new_output_messages)
    response_model_name: Optional[str] = None
    response_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


@dataclass(frozen=True)
class LLMInvocation:
    """
    Immutable representation of a complete LLM invocation lifecycle.
    Combines request, response, and telemetry metadata.
    """
    request: LLMRequest
    response: Optional[LLMResponse] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    # Internal telemetry fields - managed by TelemetryHandler
    span: Optional[Span] = field(default=None, repr=False)
    context_token: Optional[ContextToken] = field(default=None, repr=False)


@dataclass(frozen=True)
class Error:
    message: str
    type: Type[BaseException]
