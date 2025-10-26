"""Simplified LangChain callback handler (Phase 1).

Only maps callbacks to GenAI util types and delegates lifecycle to TelemetryHandler.
Complex logic removed (agent heuristics, child counting, prompt capture, events).
"""

from __future__ import annotations

import json
from typing import Any, Optional, List
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.trace import Tracer

from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    Workflow,
    Task,
    AgentInvocation,
    LLMInvocation,
    InputMessage,
    OutputMessage,
    Text,
    Error as GenAIError,
)


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except (TypeError, ValueError):
        return "<unrepr>"


def _serialize(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            return str(obj)
        except (TypeError, ValueError):
            return None


def _is_agent_root(tags: Optional[list[str]], metadata: Optional[dict[str, Any]]) -> bool:
    if tags:
        for tag in tags:
            try:
                if "agent" in str(tag).lower():
                    return True
            except (TypeError, ValueError):
                continue
    if metadata and metadata.get("agent_name"):
        return True
    return False


class LangchainCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        tracer: Tracer,
        duration_histogram: Any,
        token_histogram: Any,
        *,
        telemetry_handler: Optional[Any] = None,
        telemetry_handler_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.tracer = tracer
        self._handler = telemetry_handler or get_telemetry_handler(
            **(telemetry_handler_kwargs or {})
        )

    def on_chain_start(
        self,
    serialized: Optional[dict[str, Any]],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        payload = serialized or {}
        name_source = payload.get("name") or payload.get("id") or extra.get("name")
        name = _safe_str(name_source or "chain")
        attrs: dict[str, Any] = {}
        if metadata:
            attrs.update(metadata)
        if tags:
            attrs["tags"] = [str(t) for t in tags]
        if parent_run_id is None:
            if _is_agent_root(tags, metadata):
                agent = AgentInvocation(
                    name=name,
                    run_id=run_id,
                    attributes=attrs,
                )
                agent.input_context = _serialize(inputs)
                agent.agent_name = _safe_str(metadata.get("agent_name")) if metadata and metadata.get("agent_name") else name
                agent.parent_run_id = None
                agent.framework = "langchain"
                if metadata:
                    if metadata.get("agent_type"):
                        agent.agent_type = _safe_str(metadata.get("agent_type"))
                    if metadata.get("model_name"):
                        agent.model = _safe_str(metadata.get("model_name"))
                    if metadata.get("system"):
                        agent.system = _safe_str(metadata.get("system"))
                self._handler.start_agent(agent)
            else:
                wf = Workflow(name=name, run_id=run_id, attributes=attrs)
                wf.initial_input = _serialize(inputs)
                self._handler.start_workflow(wf)
        else:
            task = Task(
                name=name,
                run_id=run_id,
                parent_run_id=parent_run_id,
                task_type="chain",
                attributes=attrs,
            )
            parent_entity = self._handler.get_entity(parent_run_id)
            if isinstance(parent_entity, AgentInvocation):
                task.agent_name = parent_entity.name
                task.agent_id = str(parent_entity.run_id)
            task.input_data = _serialize(inputs)
            self._handler.start_task(task)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_kwargs: Any,
    ) -> None:
        entity = self._handler.get_entity(run_id)
        if entity is None:
            return
        if isinstance(entity, Workflow):
            entity.final_output = _serialize(outputs)
            self._handler.stop_workflow(entity)
        elif isinstance(entity, AgentInvocation):
            entity.output_result = _serialize(outputs)
            self._handler.stop_agent(entity)
        elif isinstance(entity, Task):
            entity.output_data = _serialize(outputs)
            self._handler.stop_task(entity)

    def on_chat_model_start(
        self,
        serialized: Optional[dict[str, Any]],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        payload = serialized or {}
        model_source = (
            payload.get("name")
            or payload.get("id")
            or (metadata.get("model_name") if metadata else None)
            or extra.get("name")
        )
        request_model = _safe_str(model_source or "model")
        input_messages: list[InputMessage] = []
        for batch in messages:
            for m in batch:
                content = getattr(m, "content", "")
                input_messages.append(InputMessage(role="user", parts=[Text(content=_safe_str(content))]))
        attrs: dict[str, Any] = {}
        if metadata:
            attrs.update(metadata)
        if tags:
            attrs["tags"] = [str(t) for t in tags]
        inv = LLMInvocation(
            request_model=request_model,
            input_messages=input_messages,
            attributes=attrs,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )
        if parent_run_id is not None:
            parent_entity = self._handler.get_entity(parent_run_id)
            if isinstance(parent_entity, AgentInvocation):
                inv.agent_name = parent_entity.name
                inv.agent_id = str(parent_entity.run_id)
        self._handler.start_llm(inv)

    def on_llm_start(
        self,
        serialized: Optional[dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        message_batches = [[HumanMessage(content=p)] for p in prompts]
        self.on_chat_model_start(
            serialized=serialized,
            messages=message_batches,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **extra,
        )
        inv = self._handler.get_entity(run_id)
        if isinstance(inv, LLMInvocation):
            inv.operation = "generate_text"

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_kwargs: Any,
    ) -> None:
        inv = self._handler.get_entity(run_id)
        if not isinstance(inv, LLMInvocation):
            return
        generations = getattr(response, "generations", [])
        content = None
        if generations and generations[0] and generations[0][0].message:
            content = getattr(generations[0][0].message, "content", None)
        if content is not None:
            inv.output_messages = [
                OutputMessage(role="assistant", parts=[Text(content=_safe_str(content))], finish_reason="stop")
            ]
        llm_output = getattr(response, "llm_output", {}) or {}
        usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
        inv.input_tokens = usage.get("prompt_tokens")
        inv.output_tokens = usage.get("completion_tokens")
        self._handler.stop_llm(inv)

    def on_tool_start(
        self,
        serialized: Optional[dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        payload = serialized or {}
        name_source = payload.get("name") or payload.get("id") or extra.get("name")
        name = _safe_str(name_source or "tool")
        task = Task(
            name=name,
            run_id=run_id,
            parent_run_id=parent_run_id,
            task_type="tool_use",
            attributes=metadata or {},
        )
        task.input_data = _serialize(inputs) or _safe_str(input_str)
        self._handler.start_task(task)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_kwargs: Any,
    ) -> None:
        task = self._handler.get_entity(run_id)
        if not isinstance(task, Task):
            return
        task.output_data = _serialize(output)
        self._handler.stop_task(task)

    def _fail(self, run_id: UUID, error: BaseException) -> None:
        self._handler.fail_by_run_id(run_id, GenAIError(message=str(error), type=type(error)))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **_: Any) -> None:
        self._fail(run_id, error)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **_: Any) -> None:
        self._fail(run_id, error)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **_: Any) -> None:
        self._fail(run_id, error)

    def on_agent_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **_: Any) -> None:
        self._fail(run_id, error)

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **_: Any) -> None:
        self._fail(run_id, error)
