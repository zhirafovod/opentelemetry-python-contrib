import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from collections.abc import Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.documents import Document

from opentelemetry._events import EventLogger
from opentelemetry.trace import Span, SpanKind, set_span_in_context, use_span
from opentelemetry.metrics import Histogram
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.context import get_current, Context, set_value
from opentelemetry.instrumentation.langchain.utils import (
    dont_throw,
    should_collect_content,
    CallbackFilteredJSONEncoder,
)
from opentelemetry.instrumentation.langchain.config import Config

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from .utils import (
    chat_generation_to_event,
    message_to_event,
    query_to_event,
    document_to_event,
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


class OpenTelemetryLangChainCallbackHandler(BaseCallbackHandler):
    """
    A callback handler for LangChain that uses OpenTelemetry to create spans
    for chains, LLM calls, and tools.
    """

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
            for generation in getattr(response, "generations", []):
                for index, chat_generation in enumerate(generation):
                    self._event_logger.emit(chat_generation_to_event(chat_generation, index))
                    generation_info = chat_generation.generation_info
                    if generation_info is not None:
                        finish_reason = generation_info.get("finish_reason")
                        if finish_reason is not None and span.is_recording():
                            finish_reasons.append(finish_reason or "error")
            span.set_attribute(GenAI.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)

            # If the LLM result includes usage:
            if response.llm_output is not None:
                model_name = response.llm_output.get("model_name") or response.llm_output.get("model")
                if model_name and span.is_recording():
                    span.set_attribute(GenAI.GEN_AI_RESPONSE_MODEL, model_name)

                response_id = response.llm_output.get("id")
                if response_id and span.is_recording():
                    span.set_attribute(GenAI.GEN_AI_RESPONSE_ID, response_id)

                # usage
                usage = response.llm_output.get("usage") or response.llm_output.get("token_usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

                    if span.is_recording():
                        span.set_attribute(GenAI.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
                        span.set_attribute(GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)
                        # TODO: fix GEN_AI_USAGE_COMPLETION_TOKENS to semantic convention
                        # state.span.set_attribute(GenAI.GEN_AI_USAGE_TOTAL_TOKENS, total_tokens)

                    # Record metrics
                    self._record_token_usage(prompt_tokens, state.request_model, model_name, GenAI.GenAiTokenTypeValues.INPUT.value, GenAI.GenAiOperationNameValues.CHAT.value)
                    self._record_token_usage(completion_tokens, state.request_model, model_name, GenAI.GenAiTokenTypeValues.COMPLETION.value, GenAI.GenAiOperationNameValues.CHAT.value)

            # End the LLM span
            self._end_span(run_id)

            # Record overall duration metric
            model_for_metric = (
                response.llm_output.get("model_name")
                if response.llm_output
                else None
            )
            self._record_duration_metric(run_id, state.request_model, model_for_metric, GenAI.GenAiOperationNameValues.CHAT.value)

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
            request_model = kwargs.get("invocation_params").get("model_name") if kwargs.get("invocation_params") and kwargs.get("invocation_params").get("model_name") else None
            span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, request_model)
            # TODO: add below to opentelemetry.semconv._incubating.attributes.gen_ai_attributes
            span.set_attribute(GenAI.GEN_AI_SYSTEM, "langchain")

            span_state = _SpanState(span=span, span_context=get_current(), request_model=request_model)
            self.spans[run_id] = span_state

            for sub_messages in messages:
                for message in sub_messages:
                    self._event_logger.emit(message_to_event(message))

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

        tool_name = serialized.get("name") or kwargs.get("name") or "Tool"
        span = self._start_span(
            name=f"{tool_name}",
            kind=SpanKind.INTERNAL,
            parent_run_id=parent_run_id,
        )
        span_state = _SpanState(span=span, span_context=get_current())
        self.spans[run_id] = span_state
        if parent_run_id is not None and parent_run_id in self.spans:
            self.spans[parent_run_id].children.append(run_id)

        if should_collect_content():
            span.set_attribute("langchain.tool_input", input_str)

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

        self._end_span(run_id)

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

        name = kwargs.get("name")
        operation_name = "retrieval"
        span_name = f"{operation_name} {name}"
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
            # TODO: add below to opentelemetry.semconv._incubating.attributes.gen_ai_attributes
            span.set_attribute(GenAI.GEN_AI_SYSTEM, "langchain")

            request_model = metadata.get("ls_embedding_provider")
            span.set_attribute(GenAI.GEN_AI_REQUEST_MODEL, request_model)
            vector_store_provider =  metadata.get("ls_vector_store_provider")
            db_system="weaviate" if vector_store_provider == "WeaviateVectorStore" else "chroma"
            span.set_attribute("db.system",db_system)
            span.set_attribute("langchain.retriever", f"{vector_store_provider} {request_model}")
            if should_collect_content():
                span.set_attribute("db.query", query)
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

            # End the LLM span
            self._end_span(run_id)

            # record metrics
            self._record_duration_metric_db(run_id, state.request_model, state.request_model,"retrieval", state.db_system)


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