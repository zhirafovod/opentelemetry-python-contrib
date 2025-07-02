import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry._events import EventLogger
from opentelemetry.context import get_current, Context
from opentelemetry.metrics import Histogram
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from opentelemetry.trace import Span, SpanKind, set_span_in_context, use_span
from opentelemetry.trace.status import Status, StatusCode

from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.instrumentation.langchain.utils import (
    dont_throw,
    should_collect_content,
    CallbackFilteredJSONEncoder,
)
from .utils import (
    chat_generation_to_event,
    message_to_event,
    query_to_event,
    document_to_event,
    input_to_event,
    output_to_event,
    chat_generation_tool_calls_attributes,
    get_property_value,
)

logger = logging.getLogger(__name__)


@dataclass
class _SpanState:
    span: Span
    span_context: Context
    start_time: float = field(default_factory=time.time)
    request_model: Optional[str] = None
    db_system: Optional[str] = None
    children: List[UUID] = field(default_factory=list)

# TODO:
#  1. for POC leave just LLM Invocation support ONLY (on_llm_start, on_llm_end)
#  2. move all of the telemetry creation to opentelemetry-genai-sdk.exporter implementations
#  3. replace opentelemetry.sdk in the code here with opentelemetry-genai-sdk.api.* calls
#
class OpenTelemetryLangChainCallbackHandler(BaseCallbackHandler):
    """
    A callback handler for LangChain that uses OpenTelemetry to create spans
    for chains, LLM calls, and tools.
    """

    # TODO: pass telemetryClient from the LangChainInstrumentor class
    def __init__(
        self,
        tracer,
        duration_histogram: Histogram,
        token_histogram: Histogram,
        event_logger: EventLogger,
    ) -> None:
        super().__init__()
        self._tracer = tracer
        self._duration_histogram = duration_histogram
        self._token_histogram = token_histogram
        self._event_logger = event_logger

        # Map from run_id -> _SpanState, to keep track of spans and parent/child relationships
        self.spans: Dict[UUID, _SpanState] = {}
        self.run_inline = True  # for synchronous usage

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
            if child_state and child_state.span.end_time is None:
                child_state.span.end()
        if state.span.end_time is None:
            state.span.end()

    def _record_duration_metric(self, run_id: UUID, request_model: Optional[str], response_model: Optional[str], operation_name: str):
        """
        Records a histogram measurement for how long the operation took.
        """
        if run_id not in self.spans:
            return

        elapsed = time.time() - self.spans[run_id].start_time
        attributes = {
            GenAI.GEN_AI_SYSTEM: "langchain",
            GenAI.GEN_AI_OPERATION_NAME: operation_name
        }
        if request_model:
            attributes[GenAI.GEN_AI_REQUEST_MODEL] = request_model
        if response_model:
             attributes[GenAI.GEN_AI_RESPONSE_MODEL] = response_model

        self._duration_histogram.record(elapsed, attributes=attributes)

    def _record_duration_metric_db(self, run_id: UUID, request_model: Optional[str], response_model: Optional[str], operation_name: str, db_system: Optional[str]):
        """
        Records a histogram measurement for how long the operation took.
        """
        if run_id not in self.spans:
            return

        elapsed = time.time() - self.spans[run_id].start_time
        attributes = {
            GenAI.GEN_AI_SYSTEM: "langchain",
            GenAI.GEN_AI_OPERATION_NAME: operation_name
        }
        if request_model:
            attributes[GenAI.GEN_AI_REQUEST_MODEL] = request_model
        if response_model:
             attributes[GenAI.GEN_AI_RESPONSE_MODEL] = response_model
        if db_system:
            attributes["db.system"] = db_system

        self._duration_histogram.record(elapsed, attributes=attributes)

    def _record_duration_metric_tool(self, run_id: UUID, operation_name: str):
        """
        Records a histogram measurement for how long the operation took.
        """
        if run_id not in self.spans:
            return

        elapsed = time.time() - self.spans[run_id].start_time
        attributes = {
            GenAI.GEN_AI_SYSTEM: "langchain",
            GenAI.GEN_AI_OPERATION_NAME: operation_name
        }

        self._duration_histogram.record(elapsed, attributes=attributes)

    def _record_token_usage(self, token_count: int, request_model: Optional[str], response_model: Optional[str], token_type: str, operation_name: str):
        """
        Record usage of input or output tokens to a histogram.
        """
        if token_count <= 0:
            return
        attributes = {
            GenAI.GEN_AI_SYSTEM: "langchain",
            GenAI.GEN_AI_TOKEN_TYPE: token_type,
            GenAI.GEN_AI_OPERATION_NAME: operation_name
        }
        if request_model:
            attributes[GenAI.GEN_AI_REQUEST_MODEL] = request_model
        if response_model:
            attributes[GenAI.GEN_AI_RESPONSE_MODEL] = response_model

        self._token_histogram.record(token_count, attributes=attributes)

    @dont_throw
    def on_chain_start(
        self,
        serialized: dict,
        inputs: dict,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        if Config.is_instrumentation_suppressed():
            return

        chain_name = kwargs.get("name") or "Chain"
        span = self._start_span(
            name=f"{chain_name}",
            kind=SpanKind.INTERNAL,
            parent_run_id=parent_run_id,
        )
        with use_span(
                span,
                end_on_exit=False,
        ) as span:
            span.set_attribute("gen_ai.framework","langchain")

            if should_collect_content():
                span.set_attribute("langchain.entity_input", json.dumps(inputs, cls=CallbackFilteredJSONEncoder))

            span_state = _SpanState(span=span, span_context=get_current())
            self.spans[run_id] = span_state

            if parent_run_id is not None and parent_run_id in self.spans:
                self.spans[parent_run_id].children.append(run_id)

    @dont_throw
    def on_chain_end(
        self,
        outputs: dict,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        if Config.is_instrumentation_suppressed():
            return

        state = self.spans.get(run_id)
        if not state:
            return

        with use_span(
            state.span,
            end_on_exit=False,
        ) as span:
            if should_collect_content():
                span.set_attribute("langchain.entity_output", json.dumps(outputs, cls=CallbackFilteredJSONEncoder))

        self._end_span(run_id=run_id)

    # TODO: replace telemetry creation with opentelemetry.genai.sdk.api.on_llm_start
    @dont_throw
    def on_llm_start(
        self,
        serialized: dict,
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        if Config.is_instrumentation_suppressed():
            return

        name = serialized.get("name") or kwargs.get("name") or "LLM"
        # LLM calls typically are "CLIENT" kind
        span = self._start_span(
            name=f"{name}.completion",
            kind=SpanKind.SERVER,
            parent_run_id=parent_run_id,
        )
        span_state = _SpanState(span=span, span_context=get_current())
        self.spans[run_id] = span_state

        if parent_run_id is not None and parent_run_id in self.spans:
            self.spans[parent_run_id].children.append(run_id)

        # Try to record prompt content if environment says to do so:
        if should_collect_content():
            for i, p in enumerate(prompts):
                span.set_attribute(f"langchain.prompts.{i}", p)

    # TODO: replace telemetry creation with opentelemetry.genai.sdk.api.on_llm_stop
    @dont_throw
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs,
    ):
        if Config.is_instrumentation_suppressed():
            return

        state = self.spans.get(run_id)
        if not state:
            return

        with use_span(
            state.span,
            end_on_exit=False,
        ) as span:
            finish_reasons = []
            tool_calls_attributes = {}
            for generation in getattr(response, "generations", []):
                for index, chat_generation in enumerate(generation):
                    # changes for bedrock start
                    if chat_generation.message and chat_generation.message.usage_metadata:
                        input_tokens = chat_generation.message.usage_metadata.get("input_tokens", 0)
                        output_tokens = chat_generation.message.usage_metadata.get("output_tokens", 0)
                        span.set_attribute(GenAI.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
                        span.set_attribute(GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
                    # changes for bedrock end

                    prefix = f"{GenAI.GEN_AI_COMPLETION}.{index}"
                    tool_calls = chat_generation.message.additional_kwargs.get("tool_calls")
                    if tool_calls is not None:
                        tool_calls_attributes.update(
                            chat_generation_tool_calls_attributes(tool_calls, prefix)
                        )
                    generation_info = chat_generation.generation_info
                    if generation_info is not None:
                        finish_reason = generation_info.get("finish_reason")
                        if finish_reason is not None and span.is_recording():
                            finish_reasons.append(finish_reason or "error")
                    # changes for bedrock start
                    elif chat_generation.message and chat_generation.message.response_metadata:
                        finish_reason = chat_generation.message.response_metadata.get("stopReason")
                        if finish_reason is not None and span.is_recording():
                            finish_reasons.append(finish_reason or "error")
                    # changes for bedrock end

                    span.set_attribute(f"{GenAI.GEN_AI_RESPONSE_FINISH_REASONS}.{index}", finish_reasons)
                    # changes for bedrock start
                    self._event_logger.emit(chat_generation_to_event(chat_generation, index, prefix))

            if should_collect_content():
                span.set_attributes(tool_calls_attributes)

            # If the LLM result includes usage:
            # changes for bedrock start
            model_name = None
            if response.llm_output is not None:
                model_name = response.llm_output.get("model_name") or response.llm_output.get("model")
                if model_name and span.is_recording():
                    span.set_attribute(GenAI.GEN_AI_RESPONSE_MODEL, model_name)

                response_id = response.llm_output.get("id")
                if response_id and span.is_recording():
                    span.set_attribute(GenAI.GEN_AI_RESPONSE_ID, response_id)

                # usage
                # changes for bedrock start
            #
            if model_name is None:
                model_name = state.request_model
                span.set_attribute(GenAI.GEN_AI_RESPONSE_MODEL, model_name)

            # Record metrics
            #
            self._record_token_usage(input_tokens, state.request_model, model_name, GenAI.GenAiTokenTypeValues.INPUT.value, GenAI.GenAiOperationNameValues.CHAT.value)
            self._record_token_usage(output_tokens, state.request_model, model_name, GenAI.GenAiTokenTypeValues.COMPLETION.value, GenAI.GenAiOperationNameValues.CHAT.value)

            # End the LLM span
            self._end_span(run_id)

            # Record overall duration metric
            # changes for bedrock end
            self._record_duration_metric(run_id, state.request_model, model_name, GenAI.GenAiOperationNameValues.CHAT.value)

    @dont_throw
    def on_chat_model_start(
        self,
        serialized: dict,
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        if Config.is_instrumentation_suppressed():
            return

        name = serialized.get("name") or kwargs.get("name") or "ChatLLM"
        span = self._start_span(
            name=f"{name}.chat",
            kind=SpanKind.CLIENT,
            parent_run_id=parent_run_id,
        )

        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, GenAI.GenAiOperationNameValues.CHAT.value)

            # changes for bedrock start
            invocation_params = kwargs.get("invocation_params")
            if invocation_params:
                model_name = invocation_params.get("model_name")
                if model_name:
                    request_model = model_name
                    span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, request_model)
                else:
                    model_id = invocation_params.get("model_id")
                    if model_id:
                        request_model = model_id
                        span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, request_model)
            # changes for bedrock end

            tools = kwargs.get("invocation_params").get("tools") if kwargs.get("invocation_params") else None
            if tools is not None:
                for index, tool in enumerate(tools):
                    function = tool.get("function")
                    if function is not None:
                        span.set_attribute(f"gen_ai.request.function.{index}.name",function.get("name"))
                        span.set_attribute(f"gen_ai.request.function.{index}.description", function.get("description"))
                        span.set_attribute(f"gen_ai.request.function.{index}.parameters", str(function.get("parameters")))
            # TODO: add below to opentelemetry.semconv._incubating.attributes.gen_ai_attributes
            span.set_attribute(GenAI.GEN_AI_SYSTEM, f"LangChain:{name}")


            span_state = _SpanState(span=span, span_context=get_current(), request_model=request_model)
            self.spans[run_id] = span_state

            for sub_messages in messages:
                for index, message in enumerate(sub_messages):
                    content = get_property_value(message, "content")
                    type = get_property_value(message, "type")
                    if tools and should_collect_content():
                        if type == "human" and len(sub_messages) > 1:
                            span.set_attribute(f"gen_ai.prompt.{index}.content", content)
                            span.set_attribute(f"gen_ai.prompt.{index}.role", "human")
                        elif type == "tool":
                            span.set_attribute(f"gen_ai.prompt.{index}.content", content)
                            span.set_attribute(f"gen_ai.prompt.{index}.role", "tool")
                            span.set_attribute(f"gen_ai.prompt.{index}.tool_call_id", get_property_value(message, "tool_call_id"))
                        elif type == "ai":
                            span.set_attribute(f"gen_ai.prompt.{index}.role", "ai")
                            additional_kwargs = get_property_value(message, "additional_kwargs")
                            tool_calls = get_property_value(additional_kwargs, "tool_calls")
                            if tool_calls is not None:
                                for index2, tool_call in enumerate(tool_calls):
                                    span.set_attribute(f"gen_ai.prompt.{index}.tool_calls.{index2}.id",tool_call.get("id"))
                                    function = tool_call.get("function")
                                    span.set_attribute(f"gen_ai.prompt.{index}.tool_calls.{index2}.arguments",function.get("arguments"))
                                    span.set_attribute(f"gen_ai.prompt.{index}.tool_calls.{index2}.name",function.get("name"))

                    self._event_logger.emit(message_to_event(message, tools, content, type))

            if parent_run_id is not None and parent_run_id in self.spans:
                self.spans[parent_run_id].children.append(run_id)


    @dont_throw
    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        if Config.is_instrumentation_suppressed():
            return

        tool_name = serialized.get("name") or kwargs.get("name") or "execute_tool"
        span = self._start_span(
            name=f"{tool_name}",
            kind=SpanKind.INTERNAL,
            parent_run_id=parent_run_id,
        )
        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            description = serialized.get("description")
            span.set_attribute("gen_ai.framework","langchain")
            span.set_attribute(GenAI.GEN_AI_SYSTEM, tool_name)

            span.set_attribute("gen_ai.tool.description", description)
            span.set_attribute(GenAI.GEN_AI_TOOL_NAME, tool_name)
            span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value)

            span_state = _SpanState(span=span, span_context=get_current())
            self.spans[run_id] = span_state
            if parent_run_id is not None and parent_run_id in self.spans:
                self.spans[parent_run_id].children.append(run_id)

            if should_collect_content():
                self._event_logger.emit(input_to_event(input_str))

    @dont_throw
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        if Config.is_instrumentation_suppressed():
            return
        state = self.spans.get(run_id)
        if not state:
            return

        with use_span(
            state.span,
            end_on_exit=False,
        ) as span:
            if should_collect_content():
                span.set_attribute(GenAI.GEN_AI_TOOL_CALL_ID, output.tool_call_id)
                self._event_logger.emit(output_to_event(output))

            self._end_span(run_id)

            self._record_duration_metric_tool(run_id, GenAI.GenAiOperationNameValues.EXECUTE_TOOL.value)


    @dont_throw
    def on_retriever_start(
        self,
        serialized: dict,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        if Config.is_instrumentation_suppressed():
            return

        provider = metadata.get("ls_vector_store_provider") if metadata else "unknown"
        if provider == "WeaviateVectorStore":
            db_system = "weaviate"
        elif provider == "Chroma":
            db_system = "chroma"
        elif provider == "Milvus":
            db_system = "milvus"
        else:
            db_system = provider

        langchain_name = kwargs.get("name") # VectorStoreRetrival
        operation_name = "retrieval"
        span_name = f"LangChain:{langchain_name}({db_system})" # i.e. LangChain:VectorStoreRetrival(milvus)

        span = self._start_span(
            name=f"{span_name}",
            kind=SpanKind.CLIENT,
            parent_run_id=parent_run_id,
        )
        with use_span(
            span,
            end_on_exit=False,
        ) as span:
            span.set_attribute(GenAI.GEN_AI_OPERATION_NAME, operation_name)

            request_model = metadata.get("ls_embedding_provider")
            span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, request_model)
            span.set_attribute("gen_ai.framework", "langchain")

            if db_system:
                span.set_attribute("gen_ai.langchain.db.system", db_system)

            span.set_attribute(GenAI.GEN_AI_SYSTEM, f"{span_name}")

            if should_collect_content():
                span.set_attribute("db.query.text", query)
                self._event_logger.emit(query_to_event(query))

            span_state = _SpanState(span=span, span_context=get_current(), request_model=request_model, db_system=db_system)
            self.spans[run_id] = span_state

            if parent_run_id is not None and parent_run_id in self.spans:
                self.spans[parent_run_id].children.append(run_id)

    @dont_throw
    def on_retriever_end(
            self,
            documents: Sequence[Document],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs,
    ):
        if Config.is_instrumentation_suppressed():
            return

        state = self.spans.get(run_id)
        if not state:
            return

        with use_span(
            state.span,
            end_on_exit=False,
        ) as span:
            for index, document in enumerate(documents):
                self._event_logger.emit(document_to_event(document, index))

            # record metrics
            self._record_duration_metric_db(run_id, state.request_model, state.request_model, "retrieval",
                                            state.db_system)

        # End the LLM span
        self._end_span(run_id)



    @dont_throw
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        self._handle_error(error, run_id)

    @dont_throw
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        self._handle_error(error, run_id)

    @dont_throw
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        self._handle_error(error, run_id)

    @dont_throw
    def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None, # todo: fixme
        **kwargs,
    ):
        self._handle_error(error, run_id)

    @dont_throw
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        self._handle_error(error, run_id)

    def _handle_error(self, error: BaseException, run_id: UUID):
        if Config.is_instrumentation_suppressed():
            return
        state = self.spans.get(run_id)
        if not state:
            return
        span = state.span
        if span.is_recording():
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
        self._end_span(run_id)