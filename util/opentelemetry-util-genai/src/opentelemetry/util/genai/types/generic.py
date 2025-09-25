"""
Generic base dataclass for GenAI telemetry invocations.

This type is intended for use with telemetry generators and handlers, and can be
subclassed or extended for specific GenAI data types (e.g., LLMInvocation).
"""

from contextvars import Token
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from opentelemetry.context import Context
from opentelemetry.trace import Span


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
