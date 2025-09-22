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

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

from opentelemetry import trace
from opentelemetry._events import Event
from opentelemetry._logs import LogRecord
from opentelemetry.context import Context, get_current
from opentelemetry.metrics import Meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import (
    Span,
    SpanKind,
    Tracer,
    set_span_in_context,
    use_span,
)
from opentelemetry.trace.status import Status, StatusCode

from .data import Error
from .instruments import Instruments
from .types import LLMInvocation, ToolInvocation


@dataclass
class _SpanState:
    span: Span
    context: Context
    start_time: float
    children: List[UUID] = field(default_factory=list)


def _message_to_event(
    message, tool_functions, provider_name, framework
) -> Optional[Event]:
    content = getattr(message, "content", None)
    mtype = getattr(message, "type", None)
    body = {}
    if mtype == "tool":
        name = getattr(message, "name", None)
        tool_call_id = getattr(message, "tool_call_id", None)
        body.update(
            {
                "content": content,
                "name": name,
                "tool_call_id": tool_call_id,
            }
        )
    elif mtype == "ai":
        tool_function_calls = [
            {
                "id": tfc.id,
                "name": tfc.name,
                "arguments": tfc.arguments,
                "type": getattr(tfc, "type", None),
            }
            for tfc in (message.tool_function_calls or [])
        ]
        body.update(
            {
                "content": content or "",
                "tool_calls": str(tool_function_calls)
                if tool_function_calls
                else "",
            }
        )
    elif mtype in ("human", "system"):
        normalized_type = "user" if mtype == "human" else "system"
        mtype = normalized_type
        body.update({"content": content})

    attributes = {
        "gen_ai.framework": framework,
        "gen_ai.provider.name": provider_name,
    }

    if tool_functions is not None:
        for index, tool_function in enumerate(tool_functions):
            attributes.update(
                {
                    f"gen_ai.request.function.{index}.name": tool_function.name,
                    f"gen_ai.request.function.{index}.description": tool_function.description,
                    f"gen_ai.request.function.{index}.parameters": tool_function.parameters,
                }
            )

    return Event(
        name=f"gen_ai.{mtype}.message",
        attributes=attributes,
        body=body or None,
    )


def _create_inference_operation_event(
    invocation: LLMInvocation,
    provider_name: Optional[str],
    framework: Optional[str],
) -> Optional[Event]:
    attributes: Dict[str, str] = {}
    if framework:
        attributes["gen_ai.framework"] = framework
    if provider_name:
        attributes["gen_ai.provider.name"] = provider_name
    inv_attrs_for_op: Dict[str, Any] = (
        dict(invocation.attributes) if invocation.attributes else {}
    )
    op_name_attr = (
        str(inv_attrs_for_op.get("operation_name")).lower()
        if inv_attrs_for_op.get("operation_name") is not None
        else None
    )
    op_name_map = {
        "chat": GenAI.GenAiOperationNameValues.CHAT.value,
        "messages": GenAI.GenAiOperationNameValues.CHAT.value,
        "completion": GenAI.GenAiOperationNameValues.CHAT.value,
        "completions": GenAI.GenAiOperationNameValues.CHAT.value,
        "execute_tool": GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
    }
    derived_operation_name = (
        op_name_map.get(op_name_attr) if op_name_attr else None
    )
    attributes[GenAI.GEN_AI_OPERATION_NAME] = (
        derived_operation_name or GenAI.GenAiOperationNameValues.CHAT.value
    )
    inv_attrs: Dict[str, Any] = (
        dict(invocation.attributes) if invocation.attributes else {}
    )
    request_model = inv_attrs.get("request_model")
    if request_model:
        attributes[GenAI.GEN_AI_REQUEST_MODEL] = request_model
    temperature = inv_attrs.get("request_temperature")
    if temperature is not None:
        attributes[GenAI.GEN_AI_REQUEST_TEMPERATURE] = temperature
    top_p = inv_attrs.get("request_top_p")
    if top_p is not None:
        attributes[GenAI.GEN_AI_REQUEST_TOP_P] = top_p
    freq_penalty = inv_attrs.get("request_frequency_penalty")
    if freq_penalty is not None:
        attributes[GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY] = freq_penalty
    pres_penalty = inv_attrs.get("request_presence_penalty")
    if pres_penalty is not None:
        attributes[GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY] = pres_penalty
    stop_sequences = inv_attrs.get("request_stop_sequences")
    if stop_sequences:
        attributes[GenAI.GEN_AI_REQUEST_STOP_SEQUENCES] = stop_sequences
    seed = inv_attrs.get("request_seed")
    if seed is not None:
        attributes[GenAI.GEN_AI_REQUEST_SEED] = seed
    max_tokens = inv_attrs.get("request_max_tokens")
    if max_tokens is not None:
        attributes[GenAI.GEN_AI_REQUEST_MAX_TOKENS] = max_tokens

    response_model = inv_attrs.get("response_model_name")
    if response_model:
        attributes[GenAI.GEN_AI_RESPONSE_MODEL] = response_model
    response_id = inv_attrs.get("response_id")
    if response_id:
        attributes[GenAI.GEN_AI_RESPONSE_ID] = response_id

    input_tokens = inv_attrs.get("input_tokens")
    if input_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
    output_tokens = inv_attrs.get("output_tokens")
    if output_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens

    # TODO add missing attributes https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-events.md#event-eventgen_aiclientinferenceoperationdetails

    attributes["event.name"] = (
        "event.gen_ai.client.inference.operation.details"
    )

    tool_definitions: List[Dict[str, Any]] = []
    if invocation.tool_functions:
        for index, tool_function in enumerate(invocation.tool_functions):
            tool_definitions.append(
                {
                    "name": tool_function.name,
                    "description": tool_function.description,
                    "parameters": tool_function.parameters,
                }
            )

    body: Dict[str, Any] = {}

    # Process input messages
    input_messages: List[Dict[str, Any]] = []
    system_instructions: List[Dict[str, Any]] = []

    for message in invocation.messages:
        content = message.content
        message_type = message.type

        if message_type == "system":
            if content:
                system_instructions.append(
                    {"type": "text", "content": content}
                )
        elif message_type in ["user", "human", "ai", "assistant", "tool"]:
            role = (
                "user"
                if message_type in ["user", "human"]
                else (
                    "assistant"
                    if message_type in ["ai", "assistant"]
                    else "tool"
                )
            )
            parts: List[Dict[str, Any]] = []
            # text part if content present
            if content:
                parts.append({"type": "text", "content": content})
            # tool call parts for assistant messages
            if role == "assistant" and getattr(
                message, "tool_function_calls", None
            ):
                for tfc in message.tool_function_calls:
                    args_obj = tfc.arguments
                    if isinstance(args_obj, str):
                        try:
                            args_obj = json.loads(args_obj)
                        except Exception:
                            pass
                    parts.append(
                        {
                            "type": "tool_call",
                            "id": getattr(tfc, "id", None),
                            "name": getattr(tfc, "name", None),
                            "arguments": args_obj,
                        }
                    )
            # tool call response parts for tool role
            if role == "tool":
                parts = [
                    {
                        "type": "tool_call_response",
                        "id": getattr(message, "tool_call_id", None),
                        "result": content,
                    }
                ]
            input_messages.append({"role": role, "parts": parts})

    if input_messages:
        body["gen_ai.input.messages"] = input_messages
    if system_instructions:
        body["gen_ai.system_instructions"] = system_instructions

    output_messages: List[Dict[str, Any]] = []
    finish_reasons: List[str] = []
    for chat_generation in invocation.chat_generations:
        if chat_generation:
            parts: List[Dict[str, Any]] = []
            if chat_generation.content:
                parts.append(
                    {"type": "text", "content": chat_generation.content}
                )
            output_message_data: Dict[str, Any] = {
                "role": "assistant",
                "parts": parts,
                "finish_reason": chat_generation.finish_reason or "error",
            }
            if chat_generation.finish_reason:
                finish_reasons.append(chat_generation.finish_reason)
            output_messages.append(output_message_data)

    if output_messages:
        body["gen_ai.output.messages"] = output_messages
    if tool_definitions:
        # Ensure each definition has type:function
        for td in tool_definitions:
            td["type"] = "function"
        body["gen_ai.tool.definitions"] = tool_definitions

    return Event(
        name="gen_ai.client.inference.operation.details",
        attributes=attributes,
        body=body if body else None,
    )


def _create_inference_operation_log_record(
    invocation: LLMInvocation,
    provider_name: Optional[str],
    framework: Optional[str],
) -> Optional[LogRecord]:
    attributes: Dict[str, Any] = {}
    if framework:
        attributes["gen_ai.framework"] = framework
    if provider_name:
        attributes["gen_ai.provider.name"] = provider_name
    attributes[GenAI.GEN_AI_OPERATION_NAME] = (
        GenAI.GenAiOperationNameValues.CHAT.value
    )

    inv_attrs: Dict[str, Any] = (
        dict(invocation.attributes) if invocation.attributes else {}
    )
    request_model = inv_attrs.get("request_model")
    if request_model:
        attributes[GenAI.GEN_AI_REQUEST_MODEL] = request_model
    temperature = inv_attrs.get("request_temperature")
    if temperature is not None:
        attributes[GenAI.GEN_AI_REQUEST_TEMPERATURE] = temperature
    top_p = inv_attrs.get("request_top_p")
    if top_p is not None:
        attributes[GenAI.GEN_AI_REQUEST_TOP_P] = top_p
    freq_penalty = inv_attrs.get("request_frequency_penalty")
    if freq_penalty is not None:
        attributes[GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY] = freq_penalty
    pres_penalty = inv_attrs.get("request_presence_penalty")
    if pres_penalty is not None:
        attributes[GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY] = pres_penalty
    stop_sequences = inv_attrs.get("request_stop_sequences")
    if stop_sequences:
        attributes[GenAI.GEN_AI_REQUEST_STOP_SEQUENCES] = stop_sequences
    seed = inv_attrs.get("request_seed")
    if seed is not None:
        attributes[GenAI.GEN_AI_REQUEST_SEED] = seed
    max_tokens = inv_attrs.get("request_max_tokens")
    if max_tokens is not None:
        attributes[GenAI.GEN_AI_REQUEST_MAX_TOKENS] = max_tokens

    response_model = inv_attrs.get("response_model_name")
    if response_model:
        attributes[GenAI.GEN_AI_RESPONSE_MODEL] = response_model
    response_id = inv_attrs.get("response_id")
    if response_id:
        attributes[GenAI.GEN_AI_RESPONSE_ID] = response_id

    input_tokens = inv_attrs.get("input_tokens")
    if input_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
    output_tokens = inv_attrs.get("output_tokens")
    if output_tokens is not None:
        attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens

    tool_definitions: List[Dict[str, Any]] = []
    if invocation.tool_functions:
        for tool_function in invocation.tool_functions:
            tool_definitions.append(
                {
                    "name": tool_function.name,
                    "description": tool_function.description,
                    "parameters": tool_function.parameters,
                }
            )

    body: Dict[str, Any] = {}

    input_messages: List[Dict[str, Any]] = []
    system_instructions: List[Dict[str, Any]] = []
    for message in invocation.messages:
        content = message.content
        message_type = message.type
        if message_type == "system":
            if content:
                system_instructions.append(
                    {"type": "text", "content": content}
                )
        elif message_type in ["user", "human", "ai", "assistant", "tool"]:
            role = (
                "user"
                if message_type in ["user", "human"]
                else (
                    "assistant"
                    if message_type in ["ai", "assistant"]
                    else "tool"
                )
            )
            parts: List[Dict[str, Any]] = []
            if content:
                parts.append({"type": "text", "content": content})
            if role == "assistant" and getattr(
                message, "tool_function_calls", None
            ):
                for tfc in message.tool_function_calls:
                    args_obj = tfc.arguments
                    if isinstance(args_obj, str):
                        try:
                            args_obj = json.loads(args_obj)
                        except Exception:
                            pass
                    parts.append(
                        {
                            "type": "tool_call",
                            "id": getattr(tfc, "id", None),
                            "name": getattr(tfc, "name", None),
                            "arguments": args_obj,
                        }
                    )
            if role == "tool":
                parts = [
                    {
                        "type": "tool_call_response",
                        "id": getattr(message, "tool_call_id", None),
                        "result": content,
                    }
                ]
            input_messages.append({"role": role, "parts": parts})

    if input_messages:
        body["gen_ai.input.messages"] = input_messages
    if system_instructions:
        body["gen_ai.system_instructions"] = system_instructions

    output_messages: List[Dict[str, Any]] = []
    finish_reasons: List[str] = []
    for chat_generation in invocation.chat_generations:
        parts: List[Dict[str, Any]] = []
        if chat_generation.content:
            parts.append({"type": "text", "content": chat_generation.content})
        output_message_data: Dict[str, Any] = {
            "role": "assistant",
            "parts": parts,
            "finish_reason": chat_generation.finish_reason or "error",
        }
        if chat_generation.finish_reason:
            finish_reasons.append(chat_generation.finish_reason)
        output_messages.append(output_message_data)
    if output_messages:
        body["gen_ai.output.messages"] = output_messages
    if tool_definitions:
        # Ensure each definition has type:function
        for td in tool_definitions:
            td["type"] = "function"
        body["gen_ai.tool.definitions"] = tool_definitions

    return LogRecord(
        event_name="event.gen_ai.client.inference.operation.details",
        attributes=attributes,
        body=body if body else None,
    )


def _create_tool_operation_event(
    invocation: ToolInvocation, framework: Optional[str]
) -> Optional[Event]:
    attributes: Dict[str, Any] = {}
    if framework:
        attributes["gen_ai.framework"] = framework
    tool_name = (
        invocation.attributes.get("tool_name")
        if invocation.attributes
        else None
    )
    description = (
        invocation.attributes.get("description")
        if invocation.attributes
        else None
    )
    if tool_name:
        attributes[GenAI.GEN_AI_TOOL_NAME] = tool_name
    if description:
        attributes["gen_ai.tool.description"] = description
    # Provide event name as attribute for log backends that don't propagate event_name
    attributes["event.name"] = "event.gen_ai.tool.operation.details"
    # operation name per semconv
    attributes[GenAI.GEN_AI_OPERATION_NAME] = (
        GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value
    )

    body: Dict[str, Any] = {}
    input_messages: List[Dict[str, Any]] = []
    if invocation.input_str is not None:
        input_messages.append(
            {"role": "tool", "content": invocation.input_str}
        )
    if input_messages:
        body["gen_ai.input.messages"] = input_messages

    output_messages: List[Dict[str, Any]] = []
    if invocation.output is not None:
        output_messages.append(
            {
                "role": "tool",
                "content": invocation.output.content,
                "id": invocation.output.tool_call_id,
            }
        )
    if output_messages:
        body["gen_ai.output.messages"] = output_messages

    return Event(
        name="event.gen_ai.tool.operation.details",
        attributes=attributes,
        body=body if body else None,
    )


def _create_tool_operation_log_record(
    invocation: ToolInvocation, framework: Optional[str]
) -> Optional[LogRecord]:
    attributes: Dict[str, Any] = {}
    if framework:
        attributes["gen_ai.framework"] = framework
    tool_name = (
        invocation.attributes.get("tool_name")
        if invocation.attributes
        else None
    )
    description = (
        invocation.attributes.get("description")
        if invocation.attributes
        else None
    )
    if tool_name:
        attributes[GenAI.GEN_AI_TOOL_NAME] = tool_name
    if description:
        attributes["gen_ai.tool.description"] = description

    body: Dict[str, Any] = {}
    input_messages: List[Dict[str, Any]] = []
    if invocation.input_str is not None:
        input_messages.append(
            {"role": "tool", "content": invocation.input_str}
        )
    if input_messages:
        body["gen_ai.input.messages"] = input_messages

    output_messages: List[Dict[str, Any]] = []
    if invocation.output is not None:
        output_messages.append(
            {
                "role": "tool",
                "content": invocation.output.content,
                "id": invocation.output.tool_call_id,
            }
        )
    if output_messages:
        body["gen_ai.output.messages"] = output_messages

    return LogRecord(
        event_name="event.gen_ai.tool.operation.details",
        attributes=attributes,
        body=body if body else None,
    )


def _chat_generation_to_event(
    chat_generation, index, prefix, provider_name, framework
) -> Optional[Event]:
    if chat_generation:
        attributes = {
            "gen_ai.framework": framework,
            "gen_ai.provider.name": provider_name,
        }

        message = {
            "content": chat_generation.content,
            "type": chat_generation.type,
        }
        body = {
            "index": index,
            "finish_reason": chat_generation.finish_reason or "error",
            "message": message,
        }

        tool_function_calls = getattr(
            chat_generation, "tool_function_calls", None
        )
        if tool_function_calls is not None:
            attributes.update(
                chat_generation_tool_function_calls_attributes(
                    tool_function_calls, prefix
                )
            )

        return Event(
            name="gen_ai.choice",
            attributes=attributes,
            body=body or None,
        )


def _input_to_event(input):
    # TODO: add check should_collect_content()
    if input is not None:
        body = {
            "content": input,
            "role": "tool",
        }
        attributes = {
            "gen_ai.framework": "langchain",
        }

        return Event(
            name="gen_ai.tool.message",
            attributes=attributes,
            body=body if body else None,
        )


def _input_to_log_record(input):
    # TODO: add check should_collect_content()
    if input is not None:
        body = {
            "content": input,
            "role": "tool",
        }
        attributes = {
            "gen_ai.framework": "langchain",
        }

        return LogRecord(
            event_name="gen_ai.tool.message",
            attributes=attributes,
            body=body if body else None,
        )


def _output_to_event(output):
    if output is not None:
        body = {
            "content": output.content,
            "id": output.tool_call_id,
            "role": "tool",
        }
        attributes = {
            "gen_ai.framework": "langchain",
        }

        return Event(
            name="gen_ai.tool.message",
            attributes=attributes,
            body=body if body else None,
        )


def _output_to_log_record(output):
    if output is not None:
        body = {
            "content": output.content,
            "id": output.tool_call_id,
            "role": "tool",
        }
        attributes = {
            "gen_ai.framework": "langchain",
        }

        return LogRecord(
            event_name="gen_ai.tool.message",
            attributes=attributes,
            body=body if body else None,
        )


def _get_metric_attributes_llm(
    request_model: Optional[str],
    response_model: Optional[str],
    operation_name: Optional[str],
    provider_name: Optional[str],
    framework: Optional[str],
) -> Dict:
    attributes = {
        # TODO: add below to opentelemetry.semconv._incubating.attributes.gen_ai_attributes
        "gen_ai.framework": framework,
    }
    if provider_name:
        attributes["gen_ai.provider.name"] = provider_name
    if operation_name:
        attributes[GenAI.GEN_AI_OPERATION_NAME] = operation_name
    if request_model:
        attributes[GenAI.GEN_AI_REQUEST_MODEL] = request_model
    if response_model:
        attributes[GenAI.GEN_AI_RESPONSE_MODEL] = response_model

    return attributes


def chat_generation_tool_function_calls_attributes(
    tool_function_calls, prefix
):
    attributes = {}
    for idx, tool_function_call in enumerate(tool_function_calls):
        tool_call_prefix = f"{prefix}.tool_calls.{idx}"
        attributes[f"{tool_call_prefix}.id"] = tool_function_call.id
        attributes[f"{tool_call_prefix}.name"] = tool_function_call.name
        attributes[f"{tool_call_prefix}.arguments"] = (
            tool_function_call.arguments
        )
    return attributes


class BaseExporter:
    """
    Abstract base for exporters mapping GenAI types -> OpenTelemetry.
    """

    def init_llm(self, invocation: LLMInvocation):
        raise NotImplementedError

    def init_tool(self, invocation: ToolInvocation):
        raise NotImplementedError

    def export_llm(self, invocation: LLMInvocation):
        raise NotImplementedError

    def export_tool(self, invocation: ToolInvocation):
        raise NotImplementedError

    def error_llm(self, error: Error, invocation: LLMInvocation):
        raise NotImplementedError

    def error_tool(self, error: Error, invocation: ToolInvocation):
        raise NotImplementedError


class SpanMetricEventExporter(BaseExporter):
    """
    Emits spans, metrics and events for a full telemetry picture.
    """

    def __init__(
        self, event_logger, logger, tracer: Tracer = None, meter: Meter = None
    ):
        self._tracer = tracer or trace.get_tracer(__name__)
        instruments = Instruments(meter)
        self._duration_histogram = instruments.operation_duration_histogram
        self._token_histogram = instruments.token_usage_histogram
        self._event_logger = event_logger
        self._logger = logger

        # Map from run_id -> _SpanState, to keep track of spans and parent/child relationships
        self.spans: Dict[UUID, _SpanState] = {}

    def _start_span(
        self,
        name: str,
        kind: SpanKind,
        parent_run_id: Optional[UUID] = None,
    ) -> Span:
        if parent_run_id is not None and parent_run_id in self.spans:
            parent_span = self.spans[parent_run_id].span
            ctx = set_span_in_context(parent_span)
            span = self._tracer.start_span(name=name, kind=kind, context=ctx)
        else:
            # top-level or missing parent
            span = self._tracer.start_span(name=name, kind=kind)

        return span

    def _end_span(self, run_id: UUID):
        state = self.spans[run_id]
        for child_id in state.children:
            child_state = self.spans.get(child_id)
            if child_state and child_state.span._end_time is None:
                child_state.span.end()
        if state.span._end_time is None:
            state.span.end()

    def init_llm(self, invocation: LLMInvocation):
        if (
            invocation.parent_run_id is not None
            and invocation.parent_run_id in self.spans
        ):
            self.spans[invocation.parent_run_id].children.append(
                invocation.run_id
            )

    def export_llm(self, invocation: LLMInvocation):
        request_model = invocation.attributes.get("request_model")
        span = self._start_span(
            name=f"{GenAI.GenAiOperationNameValues.CHAT.value} {request_model}",
            kind=SpanKind.CLIENT,
            parent_run_id=invocation.parent_run_id,
        )
        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            # single event
            provider_name = (
                str(invocation.attributes.get("provider_name"))
                if invocation.attributes
                and invocation.attributes.get("provider_name")
                else None
            )
            framework = (
                str(invocation.attributes.get("framework"))
                if invocation.attributes
                and invocation.attributes.get("framework")
                else None
            )
            consolidated_event = _create_inference_operation_event(
                invocation, provider_name, framework
            )
            if consolidated_event:
                self._event_logger.emit(consolidated_event)
            # consolidated_log = _create_inference_operation_log_record(invocation, provider_name, framework)
            # if consolidated_log:
            #     self._logger.emit(consolidated_log)
            # multiple message event
            if invocation.messages:
                for message in invocation.messages:
                    legacy_event = _message_to_event(
                        message=message,
                        tool_functions=invocation.tool_functions,
                        provider_name=provider_name,
                        framework=framework,
                    )
                    if legacy_event:
                        self._event_logger.emit(legacy_event)

            span_state = _SpanState(
                span=span,
                context=get_current(),
                start_time=invocation.start_time,
            )
            self.spans[invocation.run_id] = span_state

            provider_name = ""
            attributes: Dict[str, Any] = (
                dict(invocation.attributes) if invocation.attributes else {}
            )
            if attributes:
                top_p = attributes.get("request_top_p")
                if top_p:
                    span.set_attribute(GenAI.GEN_AI_REQUEST_TOP_P, top_p)
                frequency_penalty = attributes.get("request_frequency_penalty")
                if frequency_penalty:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY,
                        frequency_penalty,
                    )
                presence_penalty = attributes.get("request_presence_penalty")
                if presence_penalty:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY, presence_penalty
                    )
                stop_sequences = attributes.get("request_stop_sequences")
                if stop_sequences:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_STOP_SEQUENCES, stop_sequences
                    )
                seed = attributes.get("request_seed")
                if seed:
                    span.set_attribute(GenAI.GEN_AI_REQUEST_SEED, seed)
                max_tokens = attributes.get("request_max_tokens")
                if max_tokens:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_MAX_TOKENS, max_tokens
                    )
                provider_name = attributes.get("provider_name")
                if provider_name:
                    # TODO: add to semantic conventions
                    span.set_attribute("gen_ai.provider.name", provider_name)
                temperature = attributes.get("request_temperature")
                if temperature:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_TEMPERATURE, temperature
                    )
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.CHAT.value,
            )
            if request_model:
                span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, request_model)

            # TODO: add below to opentelemetry.semconv._incubating.attributes.gen_ai_attributes
            framework = invocation.attributes.get("framework")
            if framework:
                span.set_attribute("gen_ai.framework", framework)

            # tools function during 1st and 2nd llm invocation request attributes start --
            if invocation.tool_functions is not None:
                for index, tool_function in enumerate(
                    invocation.tool_functions
                ):
                    span.set_attribute(
                        f"gen_ai.request.function.{index}.name",
                        tool_function.name,
                    )
                    span.set_attribute(
                        f"gen_ai.request.function.{index}.description",
                        tool_function.description,
                    )
                    span.set_attribute(
                        f"gen_ai.request.function.{index}.parameters",
                        tool_function.parameters,
                    )
            # tools request attributes end --

            # span.set_attribute(GenAI.GEN_AI_SYSTEM, system)

            # Add response details as span attributes
            tool_calls_attributes = {}
            for index, chat_generation in enumerate(
                invocation.chat_generations
            ):
                # tools generation during first invocation of llm start --
                prefix = f"{GenAI.GEN_AI_COMPLETION}.{index}"
                tool_function_calls = chat_generation.tool_function_calls
                if tool_function_calls is not None:
                    tool_calls_attributes.update(
                        chat_generation_tool_function_calls_attributes(
                            tool_function_calls, prefix
                        )
                    )
                # tools attributes end --

                # TODO: remove deprecated event logging and its initialization and use below logger instead
                self._event_logger.emit(
                    _chat_generation_to_event(
                        chat_generation,
                        index,
                        prefix,
                        provider_name,
                        framework,
                    )
                )
                # TODO: logger is not emitting event name, fix it
                # self._logger.emit(_chat_generation_to_log_record(chat_generation, index, prefix, provider_name, framework))
                # span.set_attribute(f"{GenAI.GEN_AI_RESPONSE_FINISH_REASONS}.{index}", chat_generation.finish_reason)

            # TODO: decide if we want to show this as span attributes
            # span.set_attributes(tool_calls_attributes)

            response_model = attributes.get("response_model_name")
            if response_model:
                span.set_attribute(GenAI.GEN_AI_RESPONSE_MODEL, response_model)

            response_id = attributes.get("response_id")
            if response_id:
                span.set_attribute(GenAI.GEN_AI_RESPONSE_ID, response_id)

            # usage
            prompt_tokens = attributes.get("input_tokens")
            if prompt_tokens:
                span.set_attribute(
                    GenAI.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )

            completion_tokens = attributes.get("output_tokens")
            if completion_tokens:
                span.set_attribute(
                    GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
                )

            metric_attributes = _get_metric_attributes_llm(
                request_model,
                response_model,
                GenAI.GenAiOperationNameValues.CHAT.value,
                provider_name,
                framework,
            )

            # Record token usage metrics
            prompt_tokens_attributes = {
                GenAI.GEN_AI_TOKEN_TYPE: GenAI.GenAiTokenTypeValues.INPUT.value
            }
            prompt_tokens_attributes.update(metric_attributes)
            self._token_histogram.record(
                prompt_tokens, attributes=prompt_tokens_attributes
            )

            completion_tokens_attributes = {
                GenAI.GEN_AI_TOKEN_TYPE: GenAI.GenAiTokenTypeValues.COMPLETION.value
            }
            completion_tokens_attributes.update(metric_attributes)
            self._token_histogram.record(
                completion_tokens, attributes=completion_tokens_attributes
            )

            # End the LLM span
            self._end_span(invocation.run_id)
            invocation.span_id = span_state.span.get_span_context().span_id
            invocation.trace_id = span_state.span.get_span_context().trace_id

            # Record overall duration metric
            elapsed = invocation.end_time - invocation.start_time
            self._duration_histogram.record(
                elapsed, attributes=metric_attributes
            )

    def error_llm(self, error: Error, invocation: LLMInvocation):
        request_model = invocation.attributes.get("request_model")
        span = self._start_span(
            name=f"{GenAI.GenAiOperationNameValues.CHAT.value} {request_model}",
            kind=SpanKind.CLIENT,
            parent_run_id=invocation.parent_run_id,
        )

        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            span_state = _SpanState(
                span=span,
                context=get_current(),
                start_time=invocation.start_time,
            )
            self.spans[invocation.run_id] = span_state

            provider_name = ""
            attributes = invocation.attributes
            if attributes:
                top_p = attributes.get("request_top_p")
                if top_p:
                    span.set_attribute(GenAI.GEN_AI_REQUEST_TOP_P, top_p)
                frequency_penalty = attributes.get("request_frequency_penalty")
                if frequency_penalty:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY,
                        frequency_penalty,
                    )
                presence_penalty = attributes.get("request_presence_penalty")
                if presence_penalty:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY, presence_penalty
                    )
                stop_sequences = attributes.get("request_stop_sequences")
                if stop_sequences:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_STOP_SEQUENCES, stop_sequences
                    )
                seed = attributes.get("request_seed")
                if seed:
                    span.set_attribute(GenAI.GEN_AI_REQUEST_SEED, seed)
                max_tokens = attributes.get("request_max_tokens")
                if max_tokens:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_MAX_TOKENS, max_tokens
                    )
                provider_name = attributes.get("provider_name")
                if provider_name:
                    # TODO: add to semantic conventions
                    span.set_attribute("gen_ai.provider.name", provider_name)
                temperature = attributes.get("request_temperature")
                if temperature:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_TEMPERATURE, temperature
                    )
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.CHAT.value,
            )
            if request_model:
                span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, request_model)

            # TODO: add below to opentelemetry.semconv._incubating.attributes.gen_ai_attributes
            framework = attributes.get("framework")
            if framework:
                span.set_attribute("gen_ai.framework", framework)

            span.set_status(Status(StatusCode.ERROR, error.message))
            if span.is_recording():
                span.set_attribute(
                    ErrorAttributes.ERROR_TYPE, error.type.__qualname__
                )

            self._end_span(invocation.run_id)

            framework = attributes.get("framework")

            metric_attributes = _get_metric_attributes_llm(
                request_model,
                "",
                GenAI.GenAiOperationNameValues.CHAT.value,
                provider_name,
                framework,
            )

            # Record overall duration metric
            elapsed = invocation.end_time - invocation.start_time
            self._duration_histogram.record(
                elapsed, attributes=metric_attributes
            )

    def init_tool(self, invocation: ToolInvocation):
        if (
            invocation.parent_run_id is not None
            and invocation.parent_run_id in self.spans
        ):
            self.spans[invocation.parent_run_id].children.append(
                invocation.run_id
            )

    def export_tool(self, invocation: ToolInvocation):
        attributes = invocation.attributes
        tool_name = attributes.get("tool_name")
        span = self._start_span(
            name=f"{GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value} {tool_name}",
            kind=SpanKind.INTERNAL,
            parent_run_id=invocation.parent_run_id,
        )
        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            self._event_logger.emit(_input_to_event(invocation.input_str))
            # TODO: logger is not emitting event name, fix it
            self._logger.emit(_input_to_log_record(invocation.input_str))

            span_state = _SpanState(
                span=span,
                context=get_current(),
                start_time=invocation.start_time,
            )
            self.spans[invocation.run_id] = span_state

            description = attributes.get("description")
            span.set_attribute("gen_ai.tool.description", description)
            span.set_attribute(GenAI.GEN_AI_TOOL_NAME, tool_name)
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
            )

            # TODO: if should_collect_content():
            span.set_attribute(
                GenAI.GEN_AI_TOOL_CALL_ID, invocation.output.tool_call_id
            )
            # TODO: remove deprecated event logging and its initialization and use below logger instead
            self._event_logger.emit(_output_to_event(invocation.output))
            # TODO: logger is not emitting event name, fix it
            self._logger.emit(_output_to_log_record(invocation.output))

            self._end_span(invocation.run_id)

            # Record overall duration metric
            elapsed = invocation.end_time - invocation.start_time
            metric_attributes = {
                GenAI.GEN_AI_OPERATION_NAME: GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value
            }
            self._duration_histogram.record(
                elapsed, attributes=metric_attributes
            )

    def error_tool(self, error: Error, invocation: ToolInvocation):
        tool_name = invocation.attributes.get("tool_name")
        span = self._start_span(
            name=f"{GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value} {tool_name}",
            kind=SpanKind.INTERNAL,
            parent_run_id=invocation.parent_run_id,
        )
        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            description = invocation.attributes.get("description")
            span.set_attribute("gen_ai.tool.description", description)
            span.set_attribute(GenAI.GEN_AI_TOOL_NAME, tool_name)
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
            )

            span_state = _SpanState(
                span=span,
                span_context=get_current(),
                start_time=invocation.start_time,
                system=tool_name,
            )
            self.spans[invocation.run_id] = span_state

            span.set_status(Status(StatusCode.ERROR, error.message))
            if span.is_recording():
                span.set_attribute(
                    ErrorAttributes.ERROR_TYPE, error.type.__qualname__
                )

            self._end_span(invocation.run_id)

            # Record overall duration metric
            elapsed = invocation.end_time - invocation.start_time
            metric_attributes = {
                GenAI.GEN_AI_SYSTEM: tool_name,
                GenAI.GEN_AI_OPERATION_NAME: GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
            }
            self._duration_histogram.record(
                elapsed, attributes=metric_attributes
            )


class SpanMetricExporter(BaseExporter):
    """
    Emits only spans and metrics (no events).
    """

    def __init__(self, tracer: Tracer = None, meter: Meter = None):
        self._tracer = tracer or trace.get_tracer(__name__)
        instruments = Instruments(meter)
        self._duration_histogram = instruments.operation_duration_histogram
        self._token_histogram = instruments.token_usage_histogram

        # Map from run_id -> _SpanState, to keep track of spans and parent/child relationships
        self.spans: Dict[UUID, _SpanState] = {}

    def _start_span(
        self,
        name: str,
        kind: SpanKind,
        parent_run_id: Optional[UUID] = None,
    ) -> Span:
        if parent_run_id is not None and parent_run_id in self.spans:
            parent_span = self.spans[parent_run_id].span
            ctx = set_span_in_context(parent_span)
            span = self._tracer.start_span(name=name, kind=kind, context=ctx)
        else:
            # top-level or missing parent
            span = self._tracer.start_span(name=name, kind=kind)

        return span

    def _end_span(self, run_id: UUID):
        state = self.spans[run_id]
        for child_id in state.children:
            child_state = self.spans.get(child_id)
            if child_state and child_state.span._end_time is None:
                child_state.span.end()
        if state.span._end_time is None:
            state.span.end()

    def init_llm(self, invocation: LLMInvocation):
        if (
            invocation.parent_run_id is not None
            and invocation.parent_run_id in self.spans
        ):
            self.spans[invocation.parent_run_id].children.append(
                invocation.run_id
            )

    def export_llm(self, invocation: LLMInvocation):
        request_model = invocation.attributes.get("request_model")
        span = self._start_span(
            name=f"{GenAI.GenAiOperationNameValues.CHAT.value} {request_model}",
            kind=SpanKind.CLIENT,
            parent_run_id=invocation.parent_run_id,
        )

        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            span_state = _SpanState(
                span=span,
                context=get_current(),
                start_time=invocation.start_time,
            )
            self.spans[invocation.run_id] = span_state

            provider_name = ""
            attributes = invocation.attributes
            if attributes:
                top_p = attributes.get("request_top_p")
                if top_p:
                    span.set_attribute(GenAI.GEN_AI_REQUEST_TOP_P, top_p)
                frequency_penalty = attributes.get("request_frequency_penalty")
                if frequency_penalty:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY,
                        frequency_penalty,
                    )
                presence_penalty = attributes.get("request_presence_penalty")
                if presence_penalty:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY, presence_penalty
                    )
                stop_sequences = attributes.get("request_stop_sequences")
                if stop_sequences:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_STOP_SEQUENCES, stop_sequences
                    )
                seed = attributes.get("request_seed")
                if seed:
                    span.set_attribute(GenAI.GEN_AI_REQUEST_SEED, seed)
                max_tokens = attributes.get("request_max_tokens")
                if max_tokens:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_MAX_TOKENS, max_tokens
                    )
                provider_name = attributes.get("provider_name")
                if provider_name:
                    # TODO: add to semantic conventions
                    span.set_attribute("gen_ai.provider.name", provider_name)
                temperature = attributes.get("request_temperature")
                if temperature:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_TEMPERATURE, temperature
                    )
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.CHAT.value,
            )
            if request_model:
                span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, request_model)

            # TODO: add below to opentelemetry.semconv._incubating.attributes.gen_ai_attributes
            framework = invocation.attributes.get("framework")
            if framework:
                span.set_attribute("gen_ai.framework", framework)
            # span.set_attribute(GenAI.GEN_AI_SYSTEM, system)

            # tools function during 1st and 2nd llm invocation request attributes start --
            if invocation.tool_functions is not None:
                for index, tool_function in enumerate(
                    invocation.tool_functions
                ):
                    span.set_attribute(
                        f"gen_ai.request.function.{index}.name",
                        tool_function.name,
                    )
                    span.set_attribute(
                        f"gen_ai.request.function.{index}.description",
                        tool_function.description,
                    )
                    span.set_attribute(
                        f"gen_ai.request.function.{index}.parameters",
                        tool_function.parameters,
                    )
            # tools request attributes end --

            # tools support for 2nd llm invocation request attributes start --
            messages = invocation.messages if invocation.messages else None
            for index, message in enumerate(messages):
                content = message.content
                type = message.type
                tool_call_id = message.tool_call_id
                # TODO: if should_collect_content():
                if type == "human" or type == "system":
                    span.set_attribute(
                        f"gen_ai.prompt.{index}.content", content
                    )
                    span.set_attribute(f"gen_ai.prompt.{index}.role", "human")
                elif type == "tool":
                    span.set_attribute(
                        f"gen_ai.prompt.{index}.content", content
                    )
                    span.set_attribute(f"gen_ai.prompt.{index}.role", "tool")
                    span.set_attribute(
                        f"gen_ai.prompt.{index}.tool_call_id", tool_call_id
                    )
                elif type == "ai":
                    tool_function_calls = message.tool_function_calls
                    if tool_function_calls is not None:
                        for index3, tool_function_call in enumerate(
                            tool_function_calls
                        ):
                            span.set_attribute(
                                f"gen_ai.prompt.{index}.tool_calls.{index3}.id",
                                tool_function_call.id,
                            )
                            span.set_attribute(
                                f"gen_ai.prompt.{index}.tool_calls.{index3}.arguments",
                                tool_function_call.arguments,
                            )
                            span.set_attribute(
                                f"gen_ai.prompt.{index}.tool_calls.{index3}.name",
                                tool_function_call.name,
                            )

            # tools request attributes end --

            # Add response details as span attributes
            tool_calls_attributes = {}
            for index, chat_generation in enumerate(
                invocation.chat_generations
            ):
                # tools attributes start --
                prefix = f"{GenAI.GEN_AI_COMPLETION}.{index}"
                tool_function_calls = chat_generation.tool_function_calls
                if tool_function_calls is not None:
                    tool_calls_attributes.update(
                        chat_generation_tool_function_calls_attributes(
                            tool_function_calls, prefix
                        )
                    )
                # tools attributes end --
                span.set_attribute(
                    f"{GenAI.GEN_AI_RESPONSE_FINISH_REASONS} {index}",
                    chat_generation.finish_reason,
                )

            span.set_attributes(tool_calls_attributes)

            response_model = attributes.get("response_model_name")
            if response_model:
                span.set_attribute(GenAI.GEN_AI_RESPONSE_MODEL, response_model)

            response_id = attributes.get("response_id")
            if response_id:
                span.set_attribute(GenAI.GEN_AI_RESPONSE_ID, response_id)

            # usage
            prompt_tokens = attributes.get("input_tokens")
            if prompt_tokens:
                span.set_attribute(
                    GenAI.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )

            completion_tokens = attributes.get("output_tokens")
            if completion_tokens:
                span.set_attribute(
                    GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
                )

            # Add output content as span
            for index, chat_generation in enumerate(
                invocation.chat_generations
            ):
                span.set_attribute(
                    f"gen_ai.completion.{index}.content",
                    chat_generation.content,
                )
                span.set_attribute(
                    f"gen_ai.completion.{index}.role", chat_generation.type
                )

            metric_attributes = _get_metric_attributes_llm(
                request_model,
                response_model,
                GenAI.GenAiOperationNameValues.CHAT.value,
                provider_name,
                framework,
            )

            # Record token usage metrics
            prompt_tokens_attributes = {
                GenAI.GEN_AI_TOKEN_TYPE: GenAI.GenAiTokenTypeValues.INPUT.value
            }
            prompt_tokens_attributes.update(metric_attributes)
            self._token_histogram.record(
                prompt_tokens, attributes=prompt_tokens_attributes
            )

            completion_tokens_attributes = {
                GenAI.GEN_AI_TOKEN_TYPE: GenAI.GenAiTokenTypeValues.COMPLETION.value
            }
            completion_tokens_attributes.update(metric_attributes)
            self._token_histogram.record(
                completion_tokens, attributes=completion_tokens_attributes
            )

            # End the LLM span
            self._end_span(invocation.run_id)
            invocation.span_id = span_state.span.get_span_context().span_id
            invocation.trace_id = span_state.span.get_span_context().trace_id

            # Record overall duration metric
            elapsed = invocation.end_time - invocation.start_time
            self._duration_histogram.record(
                elapsed, attributes=metric_attributes
            )

    def error_llm(self, error: Error, invocation: LLMInvocation):
        request_model = invocation.attributes.get("request_model")
        span = self._start_span(
            name=f"{GenAI.GenAiOperationNameValues.CHAT.value} {request_model}",
            kind=SpanKind.CLIENT,
            parent_run_id=invocation.parent_run_id,
        )

        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            span_state = _SpanState(
                span=span,
                context=get_current(),
                start_time=invocation.start_time,
            )
            self.spans[invocation.run_id] = span_state

            provider_name = ""
            attributes = invocation.attributes
            if attributes:
                top_p = attributes.get("request_top_p")
                if top_p:
                    span.set_attribute(GenAI.GEN_AI_REQUEST_TOP_P, top_p)
                frequency_penalty = attributes.get("request_frequency_penalty")
                if frequency_penalty:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY,
                        frequency_penalty,
                    )
                presence_penalty = attributes.get("request_presence_penalty")
                if presence_penalty:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY, presence_penalty
                    )
                stop_sequences = attributes.get("request_stop_sequences")
                if stop_sequences:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_STOP_SEQUENCES, stop_sequences
                    )
                seed = attributes.get("request_seed")
                if seed:
                    span.set_attribute(GenAI.GEN_AI_REQUEST_SEED, seed)
                max_tokens = attributes.get("request_max_tokens")
                if max_tokens:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_MAX_TOKENS, max_tokens
                    )
                provider_name = attributes.get("provider_name")
                if provider_name:
                    # TODO: add to semantic conventions
                    span.set_attribute("gen_ai.provider.name", provider_name)
                temperature = attributes.get("request_temperature")
                if temperature:
                    span.set_attribute(
                        GenAI.GEN_AI_REQUEST_TEMPERATURE, temperature
                    )
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.CHAT.value,
            )
            if request_model:
                span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, request_model)

            # TODO: add below to opentelemetry.semconv._incubating.attributes.gen_ai_attributes
            framework = attributes.get("framework")
            if framework:
                span.set_attribute("gen_ai.framework", framework)

            # tools support for 2nd llm invocation request attributes start --
            messages = invocation.messages if invocation.messages else None
            for index, message in enumerate(messages):
                content = message.content
                type = message.type
                tool_call_id = message.tool_call_id
                # TODO: if should_collect_content():
                if type == "human" or type == "system":
                    span.set_attribute(
                        f"gen_ai.prompt.{index}.content", content
                    )
                    span.set_attribute(f"gen_ai.prompt.{index}.role", "human")
                elif type == "tool":
                    span.set_attribute(
                        f"gen_ai.prompt.{index}.content", content
                    )
                    span.set_attribute(f"gen_ai.prompt.{index}.role", "tool")
                    span.set_attribute(
                        f"gen_ai.prompt.{index}.tool_call_id", tool_call_id
                    )
                elif type == "ai":
                    tool_function_calls = message.tool_function_calls
                    if tool_function_calls is not None:
                        for index3, tool_function_call in enumerate(
                            tool_function_calls
                        ):
                            span.set_attribute(
                                f"gen_ai.prompt.{index}.tool_calls.{index3}.id",
                                tool_function_call.id,
                            )
                            span.set_attribute(
                                f"gen_ai.prompt.{index}.tool_calls.{index3}.arguments",
                                tool_function_call.arguments,
                            )
                            span.set_attribute(
                                f"gen_ai.prompt.{index}.tool_calls.{index3}.name",
                                tool_function_call.name,
                            )

            span.set_status(Status(StatusCode.ERROR, error.message))
            if span.is_recording():
                span.set_attribute(
                    ErrorAttributes.ERROR_TYPE, error.type.__qualname__
                )

            self._end_span(invocation.run_id)

            framework = attributes.get("framework")

            metric_attributes = _get_metric_attributes_llm(
                request_model,
                "",
                GenAI.GenAiOperationNameValues.CHAT.value,
                provider_name,
                framework,
            )

            # Record overall duration metric
            elapsed = invocation.end_time - invocation.start_time
            self._duration_histogram.record(
                elapsed, attributes=metric_attributes
            )

    def init_tool(self, invocation: ToolInvocation):
        if (
            invocation.parent_run_id is not None
            and invocation.parent_run_id in self.spans
        ):
            self.spans[invocation.parent_run_id].children.append(
                invocation.run_id
            )

    def export_tool(self, invocation: ToolInvocation):
        attributes = invocation.attributes
        tool_name = attributes.get("tool_name")
        span = self._start_span(
            name=f"{GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value} {tool_name}",
            kind=SpanKind.INTERNAL,
            parent_run_id=invocation.parent_run_id,
        )
        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            span_state = _SpanState(
                span=span,
                context=get_current(),
                start_time=invocation.start_time,
            )
            self.spans[invocation.run_id] = span_state

            description = attributes.get("description")
            span.set_attribute("gen_ai.tool.description", description)
            span.set_attribute(GenAI.GEN_AI_TOOL_NAME, tool_name)
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
            )

            # TODO: if should_collect_content():
            span.set_attribute(
                GenAI.GEN_AI_TOOL_CALL_ID, invocation.output.tool_call_id
            )
            # Output content available in consolidated event/log emitted in SpanMetricEventExporter

            self._end_span(invocation.run_id)

            # Record overall duration metric
            elapsed = invocation.end_time - invocation.start_time
            metric_attributes = {
                GenAI.GEN_AI_OPERATION_NAME: GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value
            }
            self._duration_histogram.record(
                elapsed, attributes=metric_attributes
            )

    def error_tool(self, error: Error, invocation: ToolInvocation):
        attributes = invocation.attributes
        tool_name = attributes.get("tool_name")
        span = self._start_span(
            name=f"{GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value} {tool_name}",
            kind=SpanKind.INTERNAL,
            parent_run_id=invocation.parent_run_id,
        )
        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            span_state = _SpanState(
                span=span,
                context=get_current(),
                start_time=invocation.start_time,
            )
            self.spans[invocation.run_id] = span_state

            description = attributes.get("description")
            span.set_attribute("gen_ai.tool.description", description)
            span.set_attribute(GenAI.GEN_AI_TOOL_NAME, tool_name)
            span.set_attribute(
                GenAI.GEN_AI_OPERATION_NAME,
                GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value,
            )

            span.set_status(Status(StatusCode.ERROR, error.message))
            if span.is_recording():
                span.set_attribute(
                    ErrorAttributes.ERROR_TYPE, error.type.__qualname__
                )

            self._end_span(invocation.run_id)

            # Record overall duration metric
            elapsed = invocation.end_time - invocation.start_time
            metric_attributes = {
                GenAI.GEN_AI_OPERATION_NAME: GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value
            }
            self._duration_histogram.record(
                elapsed, attributes=metric_attributes
            )
