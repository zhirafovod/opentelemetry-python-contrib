from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class LLMInvocation:
    """
    Represents a single LLM call invocation.
    """
    invocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = None
    prompt: str = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = None
    response: str = None
    attributes: dict = field(default_factory=dict)

@dataclass
class ToolInvocation:
    """
    Represents a single tool call invocation within a GenAI workflow.
    """
    invocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = None
    input: dict = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime = None
    output: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)