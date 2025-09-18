from enum import Enum


class ObserveSpanKindValues(Enum):
    WORKFLOW = "workflow"
    TASK = "task"
    AGENT = "agent"
    TOOL = "tool"
    LLM = "llm"
    UNKNOWN = "unknown"
