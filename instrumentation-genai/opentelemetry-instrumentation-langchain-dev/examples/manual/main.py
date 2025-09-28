import base64
import json
import os
from datetime import datetime, timedelta

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from opentelemetry import _events, _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# configure tracing
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))

# configure logging and events
_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)
_events.set_event_logger_provider(EventLoggerProvider())


class TokenManager:
    def __init__(
        self, client_id, client_secret, app_key, cache_file=".token.json"
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.app_key = app_key
        self.cache_file = cache_file
        self.token_url = "https://id.cisco.com/oauth2/default/v1/token"

    def _get_cached_token(self):
        if not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, "r") as f:
                cache_data = json.load(f)

            expires_at = datetime.fromisoformat(cache_data["expires_at"])
            if datetime.now() < expires_at - timedelta(minutes=5):
                return cache_data["access_token"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return None

    def _fetch_new_token(self):
        payload = "grant_type=client_credentials"
        value = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode("utf-8")
        ).decode("utf-8")
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {value}",
        }

        response = requests.post(self.token_url, headers=headers, data=payload)
        response.raise_for_status()

        token_data = response.json()
        expires_in = token_data.get("expires_in", 3600)
        expires_at = datetime.now() + timedelta(seconds=expires_in)

        cache_data = {
            "access_token": token_data["access_token"],
            "expires_at": expires_at.isoformat(),
        }

        with open(self.cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        os.chmod(self.cache_file, 0o600)
        return token_data["access_token"]

    def get_token(self):
        token = self._get_cached_token()
        if token:
            return token
        return self._fetch_new_token()

    def cleanup_token_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r+b") as f:
                length = f.seek(0, 2)
                f.seek(0)
                f.write(b"\0" * length)
            os.remove(self.cache_file)


def agent_demo(llm: ChatOpenAI):
    """Demonstrate a LangGraph + LangChain agent with:
    - A tool (get_capital)
    - A subagent specialized for capital questions
    - A simple classifier node routing to subagent or general LLM response

    Tracing & metrics:
      * Each LLM call is instrumented via LangChainInstrumentor.
      * Tool invocation will create its own span.
    """
    try:
        from langchain_core.tools import tool
        from langchain_core.messages import AIMessage
        from langgraph.graph import StateGraph, END
    except ImportError:  # pragma: no cover - optional dependency
        print("LangGraph or necessary LangChain core tooling not installed; skipping agent demo.")
        return

    # ---- Tool Definition ----
    capitals_map = {
        "france": "Paris",
        "germany": "Berlin",
        "italy": "Rome",
        "spain": "Madrid",
        "japan": "Tokyo",
        "canada": "Ottawa",
        "australia": "Canberra",
        "brazil": "Brasília",
        "india": "New Delhi",
        "united states": "Washington, D.C.",
        "united kingdom": "London",
    }

    @tool
    def get_capital(country: str) -> str:
        """Return the capital city for a given country (case-insensitive)."""
        return capitals_map.get(country.strip().lower(), "Unknown")

    # ---- Subagent (Capital Specialist) ----
    # Simple function representing a subagent: it uses the tool explicitly and formats an answer.
    def capital_subagent(state):
        question: str = state["input"]
        # naive extraction: last word 'France?' -> 'France'
        country = question.rstrip("?!. ").split(" ")[-1]
        cap = get_capital.run(country)  # tool invocation spans should appear
        answer = f"The capital of {country.capitalize()} is {cap}."
        return {"messages": [AIMessage(content=answer)], "output": answer}

    # ---- General Node (Fallback) ----
    def general_node(state):
        question: str = state["input"]
        response = llm.invoke([
            SystemMessage(content="You are a helpful, concise assistant."),
            HumanMessage(content=question),
        ])
        return {"messages": [response], "output": getattr(response, "content", str(response))}

    # ---- Classifier Node ----
    def classifier(state):
        q: str = state["input"].lower()
        if "capital" in q or "city" in q:
            return {"route": "capital"}
        return {"route": "general"}

    # LangGraph state: we'll use a minimal dict-like state.
    graph = StateGraph(dict)
    graph.add_node("classify", classifier)
    graph.add_node("capital_agent", capital_subagent)
    graph.add_node("general_agent", general_node)

    # Edges based on classifier route
    def route_decider(state):
        return state.get("route", "general")

    graph.add_conditional_edges("classify", route_decider, {"capital": "capital_agent", "general": "general_agent"})
    graph.add_edge("capital_agent", END)
    graph.add_edge("general_agent", END)

    graph.set_entry_point("classify")
    app = graph.compile()

    demo_questions = [
        "What is the capital of France?",
        "Explain why the sky is blue in one sentence.",
        "What is the capital city of Brazil?",
    ]

    print("\n--- LangGraph Agent Demo ---")
    for q in demo_questions:
        print(f"\nUser Question: {q}")
        result = app.invoke({"input": q})
        # result should contain output & messages set by nodes
        print("Agent Output:", result.get("output"))

    print("--- End Agent Demo ---\n")


def llm_invocation_demo(llm: ChatOpenAI):
    import random

    # List of capital questions to randomly select from
    capital_questions = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of United Kingdom?",
        "What is the capital of Japan?",
        "What is the capital of Canada?",
        "What is the capital of Australia?",
        "What is the capital of Brazil?",
        "What is the capital of India?",
        "What is the capital of United States?",
    ]


    messages = [
        SystemMessage(content="You are a helpful assistant!"),
        HumanMessage(content="What is the capital of France?"),
    ]

    result = llm.invoke(messages)

    print("LLM output:\n", result)

    selected_question = random.choice(capital_questions)
    print(f"Selected question: {selected_question}")

    system_message = "You are a helpful assistant!"

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=selected_question),
    ]

    result = llm.invoke(messages)
    print(f"LLM output: {getattr(result, 'content', result)}")


def main():
    # Set up instrumentation
    LangChainInstrumentor().instrument()

    # Set up Cisco CircuIT credentials from environment
    cisco_client_id = os.getenv("CISCO_CLIENT_ID")
    cisco_client_secret = os.getenv("CISCO_CLIENT_SECRET")
    cisco_app_key = os.getenv("CISCO_APP_KEY")
    token_manager = TokenManager(
        cisco_client_id, cisco_client_secret, cisco_app_key, "/tmp/.token.json"
    )
    api_key = token_manager.get_token()

    # ChatOpenAI setup
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.1,
        max_tokens=100,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop_sequences=["\n", "Human:", "AI:"],
        seed=100,
        api_key=api_key,
        base_url="https://chat-ai.cisco.com/openai/deployments/gpt-4.1",
        default_headers={"api-key": api_key},
        model_kwargs={"user": '{"appkey": "' + cisco_app_key + '"}'},
    )

    # LLM invocation demo (simple)
    # llm_invocation_demo(llm)

    # Run agent demo (tool + subagent). Safe if LangGraph unavailable.
    agent_demo(llm)

    # Un-instrument after use
    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
