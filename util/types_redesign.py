# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Modern, composable architecture for OpenTelemetry GenAI types.

Design Principles:
1. Composition over inheritance
2. Immutable core types with builders
3. Separation of concerns (telemetry, business data, semantic conventions)
4. Type safety and validation
5. Self-documenting code
"""

import time
from abc import ABC, abstractmethod
from contextvars import Token
from dataclasses import dataclass, field, fields as dataclass_fields
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Type, Union
from uuid import UUID, uuid4

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAIAttributes
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

# Type aliases for clarity
ContextToken = Token
GenAIOperationType = Literal["chat", "completion", "embedding", "agent", "workflow", "task", "tool_call"]
FinishReason = Literal["content_filter", "error", "length", "stop", "tool_calls"]

# ============================================================================
# CORE ARCHITECTURE: Composition-based design
# ============================================================================

@dataclass(frozen=True)
class TelemetryContext:
    """Immutable telemetry context - separates concerns from business data."""
    
    context_token: Optional[ContextToken] = None
    span: Optional[Span] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    run_id: UUID = field(default_factory=uuid4)
    parent_run_id: Optional[UUID] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Calculate duration if both start and end times are available."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def with_end_time(self, end_time: Optional[float] = None) -> "TelemetryContext":
        """Create new context with end time (immutable update)."""
        return TelemetryContext(
            context_token=self.context_token,
            span=self.span,
            start_time=self.start_time,
            end_time=end_time or time.time(),
            run_id=self.run_id,
            parent_run_id=self.parent_run_id,
            attributes=self.attributes.copy()
        )


@dataclass(frozen=True)
class ProviderInfo:
    """Provider and system information - separate concern."""
    
    provider: Optional[str] = None
    framework: Optional[str] = None
    system: Optional[str] = None
    model: Optional[str] = None


@dataclass(frozen=True)
class AgentInfo:
    """Agent-specific information - separate concern."""
    
    agent_name: Optional[str] = None
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    data_source_id: Optional[str] = None


class SemanticConventionProvider(Protocol):
    """Protocol for types that can provide semantic convention attributes."""
    
    def semantic_convention_attributes(self) -> Dict[str, Any]:
        """Return semantic convention attributes for this type."""
        ...


# ============================================================================
# BASE TYPES: Clean, focused responsibilities
# ============================================================================

@dataclass(frozen=True)
class GenAIBase(SemanticConventionProvider):
    """
    Base type for all GenAI operations using composition.
    
    Uses composition instead of inheritance to avoid complex inheritance chains.
    Immutable by default with builder methods for modifications.
    """
    
    operation_type: GenAIOperationType
    telemetry: TelemetryContext = field(default_factory=TelemetryContext)
    provider: ProviderInfo = field(default_factory=ProviderInfo)
    agent: AgentInfo = field(default_factory=AgentInfo)

    def semantic_convention_attributes(self) -> Dict[str, Any]:
        """Extract semantic convention attributes from composed data."""
        result = {}
        
        # Provider attributes
        if self.provider.provider:
            result[GenAIAttributes.GEN_AI_PROVIDER_NAME] = self.provider.provider
        if self.provider.system:
            result[GenAIAttributes.GEN_AI_SYSTEM] = self.provider.system
        if self.provider.model:
            result[GenAIAttributes.GEN_AI_REQUEST_MODEL] = self.provider.model
            
        # Agent attributes
        if self.agent.agent_name:
            result[GenAIAttributes.GEN_AI_AGENT_NAME] = self.agent.agent_name
        if self.agent.agent_id:
            result[GenAIAttributes.GEN_AI_AGENT_ID] = self.agent.agent_id
        if self.agent.conversation_id:
            result[GenAIAttributes.GEN_AI_CONVERSATION_ID] = self.agent.conversation_id
        if self.agent.data_source_id:
            result[GenAIAttributes.GEN_AI_DATA_SOURCE_ID] = self.agent.data_source_id
            
        return result

    def with_telemetry(self, **updates) -> "GenAIBase":
        """Create new instance with updated telemetry context."""
        new_telemetry = TelemetryContext(
            context_token=updates.get('context_token', self.telemetry.context_token),
            span=updates.get('span', self.telemetry.span),
            start_time=updates.get('start_time', self.telemetry.start_time),
            end_time=updates.get('end_time', self.telemetry.end_time),
            run_id=updates.get('run_id', self.telemetry.run_id),
            parent_run_id=updates.get('parent_run_id', self.telemetry.parent_run_id),
            attributes=updates.get('attributes', self.telemetry.attributes)
        )
        return self.__class__(
            operation_type=self.operation_type,
            telemetry=new_telemetry,
            provider=self.provider,
            agent=self.agent
        )


# ============================================================================
# MESSAGE TYPES: Clean, focused data structures
# ============================================================================

@dataclass(frozen=True)
class TextContent:
    """Text content with explicit type."""
    content: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class ToolCallContent:
    """Tool call content with validation."""
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None
    type: Literal["tool_call"] = "tool_call"
    
    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Tool call name cannot be empty")


@dataclass(frozen=True)
class ToolCallResponse:
    """Tool call response with clear structure."""
    response: Any
    id: Optional[str] = None
    type: Literal["tool_call_response"] = "tool_call_response"


# Union type for message parts
MessagePart = Union[TextContent, ToolCallContent, ToolCallResponse]


@dataclass(frozen=True)
class Message:
    """Generic message structure - immutable and validating."""
    role: str
    parts: List[MessagePart]
    
    def __post_init__(self):
        if not self.role.strip():
            raise ValueError("Message role cannot be empty")
        if not self.parts:
            raise ValueError("Message must have at least one part")

    @classmethod
    def from_text(cls, role: str, content: str) -> "Message":
        """Factory method for simple text messages."""
        return cls(role=role, parts=[TextContent(content=content)])

    @classmethod
    def from_tool_call(cls, role: str, name: str, arguments: Dict[str, Any], id: Optional[str] = None) -> "Message":
        """Factory method for tool call messages."""
        return cls(role=role, parts=[ToolCallContent(name=name, arguments=arguments, id=id)])


@dataclass(frozen=True)
class OutputMessage(Message):
    """Output message with finish reason."""
    finish_reason: FinishReason = "stop"


# ============================================================================
# BUSINESS DOMAIN TYPES: Clean, specific responsibilities
# ============================================================================

@dataclass(frozen=True)
class LLMInvocation(GenAIBase):
    """
    Large Language Model invocation with clean separation of concerns.
    
    No inheritance issues, clear validation, immutable by default.
    """
    
    # Core LLM data
    input_messages: List[Message] = field(default_factory=list)
    output_messages: List[OutputMessage] = field(default_factory=list)
    
    # Model parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    
    # Usage statistics
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    
    # Response metadata
    response_id: Optional[str] = None
    finish_reasons: List[FinishReason] = field(default_factory=list)
    
    def __post_init__(self):
        # Validation
        if self.operation_type not in ["chat", "completion"]:
            raise ValueError(f"Invalid operation type for LLM: {self.operation_type}")

    @classmethod
    def create_chat(
        cls,
        model: str,
        messages: Optional[List[Message]] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> "LLMInvocation":
        """Factory method for chat completions."""
        return cls(
            operation_type="chat",
            input_messages=messages or [],
            provider=ProviderInfo(provider=provider, model=model),
            **kwargs
        )

    def semantic_convention_attributes(self) -> Dict[str, Any]:
        """Extend base attributes with LLM-specific ones."""
        result = super().semantic_convention_attributes()
        
        # Add LLM-specific attributes
        result[GenAIAttributes.GEN_AI_OPERATION_NAME] = self.operation_type
        
        if self.temperature is not None:
            result[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] = self.temperature
        if self.top_p is not None:
            result[GenAIAttributes.GEN_AI_REQUEST_TOP_P] = self.top_p
        if self.top_k is not None:
            result[GenAIAttributes.GEN_AI_REQUEST_TOP_K] = self.top_k
        if self.max_tokens is not None:
            result[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] = self.max_tokens
        if self.stop_sequences:
            result[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] = self.stop_sequences
        if self.input_tokens is not None:
            result[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] = self.input_tokens
        if self.output_tokens is not None:
            result[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] = self.output_tokens
        if self.response_id:
            result[GenAIAttributes.GEN_AI_RESPONSE_ID] = self.response_id
        if self.finish_reasons:
            result[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] = self.finish_reasons
            
        return result


@dataclass(frozen=True)
class EmbeddingInvocation(GenAIBase):
    """Embedding model invocation with clear structure."""
    
    input_texts: List[str] = field(default_factory=list)
    dimension_count: Optional[int] = None
    encoding_formats: List[str] = field(default_factory=list)
    input_tokens: Optional[int] = None
    
    def __post_init__(self):
        if self.operation_type != "embedding":
            raise ValueError(f"Invalid operation type for embedding: {self.operation_type}")
        if not self.input_texts:
            raise ValueError("Embedding invocation must have input texts")

    @classmethod
    def create(
        cls,
        model: str,
        texts: List[str],
        provider: Optional[str] = None,
        **kwargs
    ) -> "EmbeddingInvocation":
        """Factory method for embeddings."""
        return cls(
            operation_type="embedding",
            input_texts=texts,
            provider=ProviderInfo(provider=provider, model=model),
            **kwargs
        )


@dataclass(frozen=True)
class ToolCall(GenAIBase):
    """Tool call invocation with validation."""
    
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    tool_id: Optional[str] = None
    
    def __post_init__(self):
        if self.operation_type != "tool_call":
            raise ValueError(f"Invalid operation type for tool call: {self.operation_type}")
        if not self.name.strip():
            raise ValueError("Tool call name cannot be empty")

    @classmethod
    def create(
        cls,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "ToolCall":
        """Factory method for tool calls."""
        return cls(
            operation_type="tool_call",
            name=name,
            arguments=arguments or {},
            **kwargs
        )


@dataclass(frozen=True)
class AgentInvocation(GenAIBase):
    """Agent invocation with clear semantics."""
    
    name: str
    operation: Literal["create_agent", "invoke_agent"] = "invoke_agent"
    agent_type: Optional[str] = None
    description: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    system_instructions: Optional[str] = None
    input_context: Optional[str] = None
    output_result: Optional[str] = None
    
    def __post_init__(self):
        if self.operation_type != "agent":
            raise ValueError(f"Invalid operation type for agent: {self.operation_type}")
        if not self.name.strip():
            raise ValueError("Agent name cannot be empty")

    @classmethod
    def create(
        cls,
        name: str,
        operation: Literal["create_agent", "invoke_agent"] = "invoke_agent",
        **kwargs
    ) -> "AgentInvocation":
        """Factory method for agent invocations."""
        return cls(
            operation_type="agent",
            name=name,
            operation=operation,
            **kwargs
        )


@dataclass(frozen=True)
class Workflow(GenAIBase):
    """Workflow orchestration with clear structure."""
    
    name: str
    workflow_type: Optional[str] = None  # sequential, parallel, graph, dynamic
    description: Optional[str] = None
    initial_input: Optional[str] = None
    final_output: Optional[str] = None
    
    def __post_init__(self):
        if self.operation_type != "workflow":
            raise ValueError(f"Invalid operation type for workflow: {self.operation_type}")
        if not self.name.strip():
            raise ValueError("Workflow name cannot be empty")

    @classmethod
    def create(
        cls,
        name: str,
        workflow_type: Optional[str] = None,
        **kwargs
    ) -> "Workflow":
        """Factory method for workflows."""
        return cls(
            operation_type="workflow",
            name=name,
            workflow_type=workflow_type,
            **kwargs
        )


@dataclass(frozen=True)
class Task(GenAIBase):
    """Task execution with clear semantics."""
    
    name: str
    objective: Optional[str] = None
    task_type: Optional[str] = None
    source: Optional[Literal["workflow", "agent"]] = None
    assigned_agent: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    
    def __post_init__(self):
        if self.operation_type != "task":
            raise ValueError(f"Invalid operation type for task: {self.operation_type}")
        if not self.name.strip():
            raise ValueError("Task name cannot be empty")

    @classmethod
    def create(
        cls,
        name: str,
        objective: Optional[str] = None,
        **kwargs
    ) -> "Task":
        """Factory method for tasks."""
        return cls(
            operation_type="task",
            name=name,
            objective=objective,
            **kwargs
        )


# ============================================================================
# EVALUATION TYPES: Clean, focused evaluation data
# ============================================================================

@dataclass(frozen=True)
class EvaluationError:
    """Evaluation error with clear structure."""
    message: str
    error_type: Type[BaseException] = Exception
    
    def __post_init__(self):
        if not self.message.strip():
            raise ValueError("Error message cannot be empty")


@dataclass(frozen=True)
class EvaluationResult:
    """
    Evaluation result with validation and clear semantics.
    
    Immutable and self-validating.
    """
    metric_name: str
    score: Optional[float] = None
    label: Optional[str] = None
    explanation: Optional[str] = None
    error: Optional[EvaluationError] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.metric_name.strip():
            raise ValueError("Metric name cannot be empty")
        if self.score is not None and not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")

    @property
    def is_successful(self) -> bool:
        """Check if evaluation was successful."""
        return self.error is None

    @classmethod
    def success(
        cls,
        metric_name: str,
        score: float,
        label: Optional[str] = None,
        explanation: Optional[str] = None,
        **kwargs
    ) -> "EvaluationResult":
        """Factory method for successful evaluations."""
        return cls(
            metric_name=metric_name,
            score=score,
            label=label,
            explanation=explanation,
            **kwargs
        )

    @classmethod
    def failure(
        cls,
        metric_name: str,
        error_message: str,
        error_type: Type[BaseException] = Exception,
        **kwargs
    ) -> "EvaluationResult":
        """Factory method for failed evaluations."""
        return cls(
            metric_name=metric_name,
            error=EvaluationError(message=error_message, error_type=error_type),
            **kwargs
        )


# ============================================================================
# BUILDER PATTERN: For complex object construction
# ============================================================================

class LLMInvocationBuilder:
    """Builder for complex LLM invocations."""
    
    def __init__(self, model: str, operation_type: GenAIOperationType = "chat"):
        self._model = model
        self._operation_type = operation_type
        self._messages: List[Message] = []
        self._provider: Optional[str] = None
        self._temperature: Optional[float] = None
        self._max_tokens: Optional[int] = None
        self._kwargs: Dict[str, Any] = {}

    def provider(self, provider: str) -> "LLMInvocationBuilder":
        self._provider = provider
        return self

    def message(self, role: str, content: str) -> "LLMInvocationBuilder":
        self._messages.append(Message.from_text(role, content))
        return self

    def messages(self, messages: List[Message]) -> "LLMInvocationBuilder":
        self._messages.extend(messages)
        return self

    def temperature(self, temperature: float) -> "LLMInvocationBuilder":
        self._temperature = temperature
        return self

    def max_tokens(self, max_tokens: int) -> "LLMInvocationBuilder":
        self._max_tokens = max_tokens
        return self

    def build(self) -> LLMInvocation:
        """Build the final LLMInvocation."""
        return LLMInvocation(
            operation_type=self._operation_type,
            input_messages=self._messages,
            provider=ProviderInfo(provider=self._provider, model=self._model),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **self._kwargs
        )


# ============================================================================
# FACTORY FUNCTIONS: Convenient creation patterns
# ============================================================================

def create_chat_completion(
    model: str,
    messages: List[Message],
    provider: Optional[str] = None,
    **kwargs
) -> LLMInvocation:
    """Factory function for chat completions."""
    return LLMInvocation.create_chat(
        model=model,
        messages=messages,
        provider=provider,
        **kwargs
    )


def create_embedding(
    model: str,
    texts: List[str],
    provider: Optional[str] = None,
    **kwargs
) -> EmbeddingInvocation:
    """Factory function for embeddings."""
    return EmbeddingInvocation.create(
        model=model,
        texts=texts,
        provider=provider,
        **kwargs
    )


# Export all public types
__all__ = [
    # Core types
    "TelemetryContext",
    "ProviderInfo", 
    "AgentInfo",
    "GenAIBase",
    
    # Message types
    "TextContent",
    "ToolCallContent", 
    "ToolCallResponse",
    "MessagePart",
    "Message",
    "OutputMessage",
    
    # Business domain types
    "LLMInvocation",
    "EmbeddingInvocation",
    "ToolCall",
    "AgentInvocation", 
    "Workflow",
    "Task",
    
    # Evaluation types
    "EvaluationError",
    "EvaluationResult",
    
    # Builders and factories
    "LLMInvocationBuilder",
    "create_chat_completion",
    "create_embedding",
    
    # Type aliases and enums
    "GenAIOperationType",
    "FinishReason",
    "ContextToken",
]
