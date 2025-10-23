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

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import (
    Span,
)
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
    get_content_capturing_mode,
    is_experimental_mode,
)


def _collect_invocation_attributes(
    invocation: LLMInvocation,
) -> Dict[str, Any]:
    """Build baseline GenAI semantic convention attributes for the invocation."""
    attributes: Dict[str, Any] = {
        GenAI.GEN_AI_OPERATION_NAME: GenAI.GenAiOperationNameValues.CHAT.value,
    }
    if invocation.request_model:
        attributes[GenAI.GEN_AI_REQUEST_MODEL] = invocation.request_model
    if invocation.provider is not None:
        attributes[GenAI.GEN_AI_PROVIDER_NAME] = invocation.provider
    if invocation.output_messages:
        attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] = [
            gen.finish_reason for gen in invocation.output_messages
        ]
    if invocation.response_model_name is not None:
        attributes[GenAI.GEN_AI_RESPONSE_MODEL] = (
            invocation.response_model_name
        )
    if invocation.response_id is not None:
        attributes[GenAI.GEN_AI_RESPONSE_ID] = invocation.response_id
    if invocation.input_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] = invocation.input_tokens
    if invocation.output_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] = invocation.output_tokens
    return attributes


def _messages_to_json(messages: List[Any]) -> str:
    """Serialize message dataclasses to JSON using the GenAI utility."""
    return gen_ai_json_dumps([asdict(message) for message in messages])


def _apply_common_span_attributes(
    span: Span, invocation: LLMInvocation
) -> None:
    """Apply attributes shared by finish() and error() and compute metrics."""
    span.update_name(
        f"{GenAI.GenAiOperationNameValues.CHAT.value} {invocation.request_model}".strip()
    )
    span.set_attributes(_collect_invocation_attributes(invocation))


def _maybe_set_span_messages(
    span: Span,
    input_messages: List[InputMessage],
    output_messages: List[OutputMessage],
) -> None:
    if not is_experimental_mode() or get_content_capturing_mode() not in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    ):
        return
    if input_messages:
        span.set_attribute(
            GenAI.GEN_AI_INPUT_MESSAGES,
            _messages_to_json(input_messages),
        )
    if output_messages:
        span.set_attribute(
            GenAI.GEN_AI_OUTPUT_MESSAGES,
            _messages_to_json(output_messages),
        )


def _apply_finish_attributes(span: Span, invocation: LLMInvocation) -> None:
    """Apply attributes/messages common to finish() paths."""
    _apply_common_span_attributes(span, invocation)
    _maybe_set_span_messages(
        span, invocation.input_messages, invocation.output_messages
    )
    span.set_attributes(invocation.attributes)


def _apply_error_attributes(span: Span, error: Error) -> None:
    """Apply status and error attributes common to error() paths."""
    span.set_status(Status(StatusCode.ERROR, error.message))
    if span.is_recording():
        span.set_attribute(ErrorAttributes.ERROR_TYPE, error.type.__qualname__)


def _build_event_attributes(invocation: LLMInvocation) -> Dict[str, Any]:
    attributes = _collect_invocation_attributes(invocation)
    if invocation.attributes:
        attributes.update(invocation.attributes)
    return attributes


def _build_event_body(invocation: LLMInvocation) -> Optional[Dict[str, str]]:
    body: Dict[str, str] = {}
    if invocation.input_messages:
        body["input_messages"] = _messages_to_json(invocation.input_messages)
    if invocation.output_messages:
        body["output_messages"] = _messages_to_json(invocation.output_messages)
    return body or None


def _emit_content_event(logger: Logger, invocation: LLMInvocation) -> None:
    if not is_experimental_mode():
        return
    try:
        content_mode = get_content_capturing_mode()
    except ValueError:
        return
    if content_mode not in (
        ContentCapturingMode.EVENT_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    ):
        return

    event_body = _build_event_body(invocation)
    event_attributes = _build_event_attributes(invocation)
    log_record = LogRecord(
        event_name="gen_ai.client.inference.operation.details",
        attributes=event_attributes,
        body=event_body,
    )
    logger.emit(log_record)


__all__ = [
    "_apply_finish_attributes",
    "_apply_error_attributes",
    "_emit_content_event",
]
