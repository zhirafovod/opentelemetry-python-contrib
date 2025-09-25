"""
Types for LLM and Tool invocations for GenAI telemetry.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from opentelemetry.util.genai.types.generic import (
    GenAI,
    InputMessage,
    OutputMessage,
)
from opentelemetry.util.types import AttributeValue


@dataclass
class LLMInvocation(GenAI):
    """
    Represents a single LLM call invocation. When creating an LLMInvocation object,
    only update the data attributes. The span and context_token attributes are
    set by the TelemetryHandler.
    """

    input_messages: List[InputMessage] = field(default_factory=list)
    output_messages: List[OutputMessage] = field(default_factory=list)
    provider: Optional[str] = None
    response_model_name: Optional[str] = None
    response_id: Optional[str] = None
    input_tokens: Optional[AttributeValue] = None
    output_tokens: Optional[AttributeValue] = None


@dataclass
class ToolInvocation(GenAI):
    """
    Represents a single Tool call invocation for GenAI telemetry.
    """

    tool_name: str = ""
    arguments: Optional[dict] = None
    result: Optional[dict] = None
    provider: Optional[str] = None
    input_tokens: Optional[AttributeValue] = None
    output_tokens: Optional[AttributeValue] = None
