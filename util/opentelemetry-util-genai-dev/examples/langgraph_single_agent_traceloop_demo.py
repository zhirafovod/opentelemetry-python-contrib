#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from typing import Any, Dict

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from opentelemetry.util.genai.handler import get_telemetry_handler
from opentelemetry.util.genai.types import (
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if present


def _setup_tracing() -> None:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


def _run_langgraph_chat(user_prompt: str) -> Dict[str, Any]:
    """Execute a *single node* LangGraph chat if dependencies are present.

    Returns a dict with keys: input_messages (list[dict]), output_messages
    (list[dict]), model_name (str), response_text (str)
    """
    try:
        from typing import TypedDict
        from langchain_openai import ChatOpenAI  # type: ignore
        from langgraph.graph import StateGraph, END  # type: ignore

        class ChatState(TypedDict):
            messages: list[dict]

        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.0)  # noqa: E501

        def agent(state: ChatState) -> ChatState:
            # The LangChain OpenAI wrapper returns an LC Message object; convert to dict
            resp = llm.invoke(state["messages"])
            # Append assistant message
            return {"messages": state["messages"] + [resp]}

        graph = StateGraph(ChatState)
        graph.add_node("agent", agent)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        executable = graph.compile()

        # LangChain expects list of dicts with role/content
        initial = {"messages": [{"role": "user", "content": user_prompt}]}
        final_state = executable.invoke(initial)
        msgs = final_state["messages"]
        # Last message assumed assistant
        assistant_msg = msgs[-1]
        resp_text = assistant_msg.content if hasattr(assistant_msg, "content") else assistant_msg.get("content")  # type: ignore[attr-defined]
        return {
            "input_messages": initial["messages"],
            "output_messages": [
                {"role": "assistant", "content": resp_text},
            ],
            "model_name": getattr(llm, "model_name", "unknown-model"),
            "response_text": resp_text,
        }
    except Exception as exc:  
        print(f"LangGraph execution failed: {exc}")
        print(f"Exception type: {type(exc)}")
        return {
            "input_messages": [{"role": "user", "content": user_prompt}],
            "output_messages": [
                {
                    "role": "assistant",
                    "content": "(stubbed response – install langgraph/langchain for real call)",  # noqa: E501
                }
            ],
            "model_name": "stub-model",
            "response_text": f"(stub) Echoing: {user_prompt[:50]}",
            "error": str(exc) if os.getenv("DEBUG_LG_DEMO") else None,
        }

# ---------------------------------------------------------------------------
def emit_llm_invocation(result: Dict[str, Any]) -> None:
    handler = get_telemetry_handler()  # Will auto-load translator if env flag set

    # Prepare serialized messages like traceloop.entity.input / output would carry
    serialized_input = json.dumps(result["input_messages"])  # already simple dicts
    serialized_output = json.dumps(result["output_messages"])

    inv = LLMInvocation(
        request_model=result["model_name"],
        input_messages=[
            InputMessage(role=m["role"], parts=[Text(m["content"])])
            for m in result["input_messages"]
        ],
        attributes={
            # Simulated legacy traceloop / openllmetry style attributes
            "traceloop.entity.name": "support_agent",  # -> gen_ai.agent.name
            "traceloop.workflow.name": "single_agent_workflow",  # -> gen_ai.workflow.name
            "traceloop.callback.name": "agent_step",  # -> gen_ai.callback.name & gen_ai.operation.source
            "traceloop.entity.input": serialized_input,  # -> gen_ai.input.messages
            # We'll set entity.output only after we have the model output
        },
    )

    handler.start_llm(inv)

    # Populate output messages and related legacy attribute then stop span
    inv.output_messages = [
        OutputMessage(role=m["role"], parts=[Text(m["content"])], finish_reason="stop")
        for m in result["output_messages"]
    ]
    inv.attributes["traceloop.entity.output"] = serialized_output  # -> gen_ai.output.messages
    handler.stop_llm(inv)


def main() -> None:  # pragma: no cover - example script
    _setup_tracing()

    user_prompt = "Hello, can you summarize OpenTelemetry GenAI?"
    print("Running single-agent LangGraph (or stub) flow…")
    result = _run_langgraph_chat(user_prompt)
    emit_llm_invocation(result)
    print("\nDone. Inspect span output above for gen_ai.* attributes.")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
