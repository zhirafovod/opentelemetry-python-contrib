# Shared utility functions for GenAI emitters (migrated from generators/utils.py)
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from opentelemetry import trace
from opentelemetry._logs import Logger
from opentelemetry.metrics import Histogram
from opentelemetry.sdk._logs._internal import LogRecord as SDKLogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.types import AttributeValue

from ..attributes import (
    GEN_AI_COMPLETION_PREFIX,
    GEN_AI_FRAMEWORK,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
)
from ..types import InputMessage, LLMInvocation, OutputMessage, Text


@dataclass
class _SpanState:
    span: trace.Span
    context: trace.Context
    start_time: float
    request_model: Optional[str] = None
    system: Optional[str] = None
    children: List[UUID] = field(default_factory=list)


def _message_to_log_record(
    message: InputMessage,
    provider_name: Optional[str],
    framework: Optional[str],
    capture_content: bool,
) -> Optional[SDKLogRecord]:
    body = asdict(message)
    if not capture_content and body and body.get("parts"):
        for part in body.get("parts", []):
            if part.get("content"):
                part["content"] = ""

    attributes: Dict[str, Any] = {
        GEN_AI_FRAMEWORK: framework,
        GEN_AI_PROVIDER_NAME: provider_name,
        "event.name": "gen_ai.client.inference.operation.details",
    }

    if capture_content:
        attributes[GEN_AI_INPUT_MESSAGES] = body

    return SDKLogRecord(
        body=body or None,
        attributes=attributes,
        event_name="gen_ai.client.inference.operation.details",
    )


def _chat_generation_to_log_record(
    chat_generation: OutputMessage,
    index: int,
    provider_name: Optional[str],
    framework: Optional[str],
    capture_content: bool,
) -> Optional[SDKLogRecord]:
    if not chat_generation:
        return None
    attributes = {
        GEN_AI_FRAMEWORK: framework,
        GEN_AI_PROVIDER_NAME: provider_name,
        "event.name": "gen_ai.choice",
    }
    content: Optional[str] = None
    for part in chat_generation.parts:
        if isinstance(part, Text):
            content = part.content
            break
    message = {"type": chat_generation.role}
    if capture_content and content is not None:
        message["content"] = content

    body = {
        "index": index,
        "finish_reason": chat_generation.finish_reason or "error",
        "message": message,
    }
    return SDKLogRecord(
        body=body or None,
        attributes=attributes,
        event_name="gen_ai.choice",
    )


def _get_metric_attributes(
    request_model: Optional[str],
    response_model: Optional[str],
    operation_name: Optional[str],
    system: Optional[str],
    framework: Optional[str],
) -> Dict[str, AttributeValue]:
    attributes: Dict[str, AttributeValue] = {}
    if framework is not None:
        attributes[GEN_AI_FRAMEWORK] = framework
    if system:
        attributes[GEN_AI_PROVIDER_NAME] = system
    if operation_name:
        attributes[GenAI.GEN_AI_OPERATION_NAME] = operation_name
    if request_model:
        attributes[GenAI.GEN_AI_REQUEST_MODEL] = request_model
    if response_model:
        attributes[GenAI.GEN_AI_RESPONSE_MODEL] = response_model
    return attributes


def _record_token_metrics(
    token_histogram: Histogram,
    prompt_tokens: Optional[AttributeValue],
    completion_tokens: Optional[AttributeValue],
    metric_attributes: Dict[str, AttributeValue],
) -> None:
    prompt_attrs: Dict[str, AttributeValue] = {
        GenAI.GEN_AI_TOKEN_TYPE: GenAI.GenAiTokenTypeValues.INPUT.value
    }
    prompt_attrs.update(metric_attributes)
    if isinstance(prompt_tokens, (int, float)):
        token_histogram.record(prompt_tokens, attributes=prompt_attrs)

    completion_attrs: Dict[str, AttributeValue] = {
        GenAI.GEN_AI_TOKEN_TYPE: GenAI.GenAiTokenTypeValues.COMPLETION.value
    }
    completion_attrs.update(metric_attributes)
    if isinstance(completion_tokens, (int, float)):
        token_histogram.record(completion_tokens, attributes=completion_attrs)


def _record_duration(
    duration_histogram: Histogram,
    invocation: LLMInvocation,
    metric_attributes: Dict[str, AttributeValue],
) -> None:
    if invocation.end_time is not None:
        elapsed: float = invocation.end_time - invocation.start_time
        duration_histogram.record(elapsed, attributes=metric_attributes)

