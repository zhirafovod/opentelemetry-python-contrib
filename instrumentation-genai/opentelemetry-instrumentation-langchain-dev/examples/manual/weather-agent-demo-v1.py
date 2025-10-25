"""Weather Agent Demo v1 (Real OpenAI + Open-Meteo)

Demonstrates:
    * Real OpenAI chat model usage (if OPENAI_API_KEY set; otherwise echo fallback)
    * Structured output parsing via LangChain provider structured output (WeatherQuery)
    * Multi-agent supervisor pattern (subagents exposed as tools)
    * LangGraph workflow alternative for deterministic orchestration
    * Real external data calls (geocoding + forecast + air quality) via Open-Meteo APIs (no API key required)
    * Activity advisory computation logic
    * Streaming updates (LangGraph stream) and instrumentation via OpenTelemetry spans

Run:
    python weather-agent-demo-v1.py "Weather for Paris tomorrow and is it good for a morning run? Also compare with Berlin"

Assumptions:
    - OPENAI_API_KEY is set for real LLM usage. If absent, a minimal echo model is used.
    - Network access is available for Open-Meteo endpoints.
"""
# pyright: ignore
# mypy: ignore-errors
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict, Tuple, Annotated

import requests

from pydantic import BaseModel, Field

# OpenTelemetry instrumentation (minimal)
try:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)
except Exception:  # pragma: no cover
    class _NoopSpan:
        def __enter__(self) -> "_NoopSpan":
            return self
        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            pass
        def set_attribute(self, key: str, value: Any) -> None:
            pass
        def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
            pass
    class _NoopTracer:
        def start_as_current_span(self, name: str, *a: Any, **kw: Any) -> _NoopSpan:
            return _NoopSpan()
    tracer = _NoopTracer()  # type: ignore

# Try imports for langchain / langgraph; fall back to lightweight stubs where feasible.
try:
    from langchain.tools import tool, ToolRuntime
    from langchain.agents import create_agent, AgentState
    from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    from langchain.chat_models import init_chat_model
    from langchain_core.language_models import BaseChatModel
except Exception:  # Provide minimal shims if langchain not installed.
    def tool(name_or_callable: Any = None, *args: Any, **kwargs: Any):  # type: ignore
        """Fallback @tool decorator supporting both @tool and @tool("name")."""
        if callable(name_or_callable):
            return name_or_callable  # no-op
        def _decorator(fn: Any) -> Any:
            return fn
        return _decorator
    ToolRuntime = Any  # type: ignore

    class AgentState(dict[str, Any]):
        pass
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content
            self.type = "human"
    class AIMessage:
        def __init__(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None):
            self.content = content
            self.tool_calls = tool_calls or []
            self.type = "ai"
        def pretty_print(self) -> None:
            print(self.content)
    class ToolMessage:
        def __init__(self, content: str, tool_call_id: str, name: Optional[str] = None):
            self.content = content
            self.tool_call_id = tool_call_id
            self.name = name or "tool"
            self.type = "tool"
    class SystemMessage:
        def __init__(self, content: str):
            self.content = content
            self.type = "system"

    class BaseChatModel:  # minimal
        def invoke(self, messages: List[Any]) -> AIMessage:  # pylint: disable=unused-argument
            return AIMessage("Stub model response.")
        def stream(self, prompt: Any) -> Any:  # pylint: disable=unused-argument
            for chunk in ["Stub", " ", "stream"]:
                yield AIMessage(chunk)
        def with_structured_output(self, schema: Any) -> "BaseChatModel":
            return self

    def create_agent(model: Any, tools: List[Any], **kwargs: Any):  # type: ignore
        class _Agent:
            def invoke(self, payload: Dict[str, Any], config: Any = None) -> Dict[str, Any]:  # type: ignore
                msgs = payload.get("messages", [])
                last = msgs[-1]["content"] if msgs and isinstance(msgs[-1], dict) else (msgs[-1].content if msgs else "")
                for t in tools:
                    if t.__name__.split("_")[0] in last:
                        return {"messages": [AIMessage(f"Tool {t.__name__} used.")]}  # minimal
                return {"messages": [AIMessage("No tool used.")]}
            def stream(self, payload: Dict[str, Any], stream_mode: str = "updates", config: Any = None):  # type: ignore
                yield {"model": {"messages": [AIMessage("stream start")]}}
        return _Agent()

# LangGraph pieces.
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send
except Exception:
    StateGraph = None  # type: ignore
    START = "__start__"  # type: ignore
    END = "__end__"  # type: ignore
    Send = lambda node, state: (node, state)  # type: ignore

# ---------------------------------------------------------------------------
# Pydantic models for structured output
# ---------------------------------------------------------------------------
class WeatherQuery(BaseModel):
    locations: List[str] = Field(..., description="List of locations requested")
    units: Literal["metric", "imperial"] = Field("metric")
    days: int = Field(1, ge=1, le=5, description="Number of forecast days")
    include_air_quality: bool = Field(True)
    include_sun: bool = Field(True)
    activity: Optional[str] = Field(None, description="Activity of interest (e.g. run)")

class RawWeatherData(BaseModel):
    location: str
    current_temp: float
    forecast: List[Dict[str, Any]]
    air_quality_index: Optional[int] = None
    sunrise: Optional[str] = None
    sunset: Optional[str] = None
    source: str = "open-meteo"

class ActivityAdvisory(BaseModel):
    location: str
    run_ok: bool
    note: str

class ComparativeSummary(BaseModel):
    primary: str
    diff: List[str]
    highlights: List[str]

class WeatherReport(BaseModel):
    query: WeatherQuery
    data: List[RawWeatherData]
    advisories: List[ActivityAdvisory]
    comparison: Optional[ComparativeSummary]
    generated_at: datetime

def get_model() -> BaseChatModel:
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return init_chat_model("openai:gpt-4o-mini")
        except Exception:  # pragma: no cover
            pass
    class _Echo(BaseChatModel):
        def invoke(self, messages):
            txt = " \n".join([getattr(m, "content", str(m)) for m in messages])
            return AIMessage("Echo: " + txt[:400])
        def with_structured_output(self, schema):
            return self
    return _Echo()

MODEL = get_model()

def parse_weather_query(user_text: str) -> WeatherQuery:
    """Parse user query into WeatherQuery using structured output if available; else heuristic fallback."""
    with tracer.start_as_current_span("model.parse_query") as span:
        span.set_attribute("input.len", len(user_text))
        try:
            if hasattr(MODEL, "with_structured_output"):
                pm = MODEL.with_structured_output(WeatherQuery)  # type: ignore[attr-defined]
                msg = pm.invoke([
                    SystemMessage("Extract weather query parameters into WeatherQuery schema."),
                    HumanMessage(user_text),
                ])
                if isinstance(msg, WeatherQuery):
                    return msg
                if isinstance(msg, AIMessage):
                    try:
                        return WeatherQuery(**json.loads(msg.content))
                    except Exception:
                        pass
        except Exception as e:  # pragma: no cover
            span.set_attribute("parse.error", str(e))
        tokens = [t.strip("?,.") for t in user_text.split()]
        locs = []
        composite = []
        for i, t in enumerate(tokens):
            title = t.title()
            if title in ["Paris", "Berlin", "London", "Tokyo", "Rome", "Madrid", "New", "York"]:
                if title == "New" and i + 1 < len(tokens) and tokens[i + 1].title() == "York":
                    composite.append("New York")
                elif title != "New":
                    locs.append(title)
        if composite:
            locs.extend(composite)
        if not locs:
            locs = ["Paris"]
        return WeatherQuery(locations=list(dict.fromkeys(locs)), units="metric", days=1, include_air_quality=True, include_sun=True, activity="run")

CACHE: Dict[str, RawWeatherData] = {}
COORD_CACHE: Dict[str, Tuple[float, float]] = {}

def _safe_request(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} for {url} params={params}")
    return resp.json()

# ---------------------------------------------------------------------------
# Tools (decorated)
# ---------------------------------------------------------------------------
def _call(tool_obj: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a StructuredTool or plain function uniformly.
    Uses .func if present (StructuredTool), else .invoke for single arg / kwargs, else direct call.
    """
    if hasattr(tool_obj, "func") and callable(getattr(tool_obj, "func")):
        return tool_obj.func(*args, **kwargs)  # type: ignore[attr-defined]
    if hasattr(tool_obj, "invoke"):
        if kwargs:
            return tool_obj.invoke(kwargs)  # type: ignore[attr-defined]
        if len(args) == 1:
            return tool_obj.invoke(args[0])  # type: ignore[attr-defined]
    return tool_obj(*args, **kwargs)
@tool
def geocode_location(query: str) -> str:
    """Geocode a location using Open-Meteo geocoding API; cache coordinates and return canonical name."""
    with tracer.start_as_current_span("tool.geocode") as span:
        q = query.strip()
        span.set_attribute("raw", q)
        try:
            data = _safe_request("https://geocoding-api.open-meteo.com/v1/search", {"name": q, "count": 1})
            results = data.get("results") or []
            if results:
                r0 = results[0]
                name = r0.get("name")
                lat = r0.get("latitude")
                lon = r0.get("longitude")
                if isinstance(name, str) and isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    COORD_CACHE[name] = (float(lat), float(lon))
                    span.set_attribute("loc", name)
                    return name
        except Exception as e:  # pragma: no cover
            span.set_attribute("geocode.error", str(e))
        fallback = q.title().split(",")[0]
        COORD_CACHE.setdefault(fallback, (48.8566, 2.3522))  # Paris default
        return fallback

@tool
def fetch_weather(location: str, units: str = "metric", days: int = 1) -> RawWeatherData:
    """Fetch weather forecast (current + daily sunrise/sunset) via Open-Meteo API."""
    with tracer.start_as_current_span("tool.fetch_weather") as span:
        span.set_attribute("location", location)
        if location in CACHE:
            span.set_attribute("cache.hit", True)
            return CACHE[location]
        span.set_attribute("cache.hit", False)
        if location not in COORD_CACHE:
            _call(geocode_location, location)
        lat, lon = COORD_CACHE[location]
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "daily": "sunrise,sunset",
            "forecast_days": days,
            "timezone": "UTC",
        }
        try:
            data = _safe_request("https://api.open-meteo.com/v1/forecast", params)
        except Exception as e:  # pragma: no cover
            span.set_attribute("fetch.error", str(e))
            raise
        current = data.get("current_weather", {}).get("temperature")
        sunrise_list = data.get("daily", {}).get("sunrise", [])
        sunset_list = data.get("daily", {}).get("sunset", [])
        forecast: List[Dict[str, Any]] = []
        for i in range(days):
            forecast.append({
                "day": i + 1,
                "temp": current,
                "sunrise": sunrise_list[i] if i < len(sunrise_list) else None,
                "sunset": sunset_list[i] if i < len(sunset_list) else None,
            })
        w = RawWeatherData(
            location=location,
            current_temp=float(current) if current is not None else 0.0,
            forecast=forecast,
            sunrise=sunrise_list[0] if sunrise_list else None,
            sunset=sunset_list[0] if sunset_list else None,
        )
        CACHE[location] = w
        return w

@tool
def fetch_air_quality(location: str) -> int:
    """Fetch air quality (US AQI) from Open-Meteo air-quality API."""
    with tracer.start_as_current_span("tool.fetch_air_quality") as span:
        if location not in COORD_CACHE:
            _call(geocode_location, location)
        lat, lon = COORD_CACHE[location]
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "us_aqi",
            "timezone": "UTC",
        }
        try:
            data = _safe_request("https://air-quality-api.open-meteo.com/v1/air-quality", params)
            aqi_series = data.get("hourly", {}).get("us_aqi", [])
            aqi = int(aqi_series[0]) if aqi_series else 0
        except Exception as e:  # pragma: no cover
            span.set_attribute("aqi.error", str(e))
            aqi = 0
        span.set_attribute("aqi", aqi)
        return aqi

@tool
def fetch_sun_times(location: str) -> Dict[str, str]:
    """Return sunrise/sunset from cached forecast (avoids extra API calls)."""
    with tracer.start_as_current_span("tool.fetch_sun_times") as span:
        if location not in CACHE:
            _call(fetch_weather, location)
        d = CACHE[location]
        span.set_attribute("has.sunrise", bool(d.sunrise))
        return {"sunrise": d.sunrise or "", "sunset": d.sunset or ""}

@tool
def generate_activity_advice(location: str, temp: float, aqi: Optional[int] = None) -> ActivityAdvisory:
    """Generate running suitability advisory."""
    with tracer.start_as_current_span("tool.generate_activity_advice"):
        run_ok = (temp >= 5 and temp <= 30) and (aqi is None or aqi < 100)
        note = f"Run OK: {run_ok}. Temp={temp}C AQI={aqi}" if aqi is not None else f"Run OK: {run_ok}. Temp={temp}C"
        return ActivityAdvisory(location=location, run_ok=run_ok, note=note)

@tool
def compare_locations(locations: List[str], data: Dict[str, Any]) -> ComparativeSummary:
    """Compare first location with others."""
    with tracer.start_as_current_span("tool.compare_locations"):
        if not locations:
            raise ValueError("No locations provided")
        primary = locations[0]
        diff = locations[1:]
        highlights = [f"{loc}: temp={data[loc].current_temp}" for loc in locations if loc in data]
        return ComparativeSummary(primary=primary, diff=diff, highlights=highlights)

@tool
def summarize_conversation(runtime: ToolRuntime) -> str:  # type: ignore
    """Summarize message counts from runtime state."""
    with tracer.start_as_current_span("tool.summarize_conversation"):
        state_msgs = runtime.state["messages"] if "messages" in runtime.state else []  # type: ignore
        human = sum(1 for m in state_msgs if getattr(m, "type", "") == "human")
        ai = sum(1 for m in state_msgs if getattr(m, "type", "") == "ai")
        tool_msgs = sum(1 for m in state_msgs if getattr(m, "type", "") == "tool")
        return f"Conversation counts human={human} ai={ai} tool={tool_msgs}"

# ---------------------------------------------------------------------------
# Multi-agent supervisor setup (subagents as tools)
# ---------------------------------------------------------------------------
# Each subagent will simply call underlying tools directly; for demo we simulate via wrappers.

# For brevity we treat subagents as simple passthrough callables decorated with @tool
@tool("call_forecast")
def call_forecast(location: str, units: str = "metric", days: int = 1) -> str:
    """Subagent: get weather + sun times for a location"""
    w = _call(fetch_weather, location, units=units, days=days)
    sun = _call(fetch_sun_times, location)
    return json.dumps({"weather": w.dict(), "sun": sun})

@tool("call_air_quality")
def call_air_quality(location: str) -> str:
    """Subagent: get air quality for a location"""
    aqi = _call(fetch_air_quality, location)
    return json.dumps({"aqi": aqi})

@tool("call_recommendation")
def call_recommendation(location: str, temp: float, aqi: Optional[int] = None) -> str:
    """Subagent: running recommendation for location"""
    adv = _call(generate_activity_advice, location, temp=temp, aqi=aqi)
    return adv.json()

SUPERVISOR_TOOLS = [
    geocode_location,
    call_forecast,
    call_air_quality,
    call_recommendation,
    compare_locations,
    summarize_conversation,
]

SUPERVISOR_AGENT = create_agent(  # type: ignore
    model=MODEL,
    tools=SUPERVISOR_TOOLS,
    system_prompt="You are a weather supervisor agent. Use tools to gather data and produce concise results.",
)

# ---------------------------------------------------------------------------
# LangGraph workflow implementation
# ---------------------------------------------------------------------------
def _merge_data_maps(existing: Dict[str, RawWeatherData], new: Dict[str, RawWeatherData]) -> Dict[str, RawWeatherData]:
    """Aggregator for parallel worker_fetch node outputs.
    LangGraph calls this when multiple updates target the same key in the same step.
    """
    if existing is None:
        return dict(new)
    merged = dict(existing)
    merged.update(new)
    return merged

class GraphState(TypedDict):
    input: str
    query: Optional[WeatherQuery]
    locations: List[str]
    # Annotated with merge function so parallel updates accumulate rather than conflict
    data_map: Annotated[Dict[str, RawWeatherData], _merge_data_maps]
    advisories: List[ActivityAdvisory]
    comparison: Optional[ComparativeSummary]
    report: Optional[WeatherReport]

if StateGraph:
    def node_parse(state: GraphState) -> Dict[str, Any]:
        with tracer.start_as_current_span("graph.node.parse"):
            q = parse_weather_query(state["input"])
            return {"query": q, "locations": q.locations}

    def node_parallel_fetch(state: GraphState):
        return [Send("worker_fetch", {"location": loc}) for loc in state["locations"]]

    def worker_fetch(state: Dict[str, Any]):
        loc = state["location"]
        w = _call(fetch_weather, loc, units="metric", days=1)
        aqi = _call(fetch_air_quality, loc)
        sun = _call(fetch_sun_times, loc)
        w.air_quality_index = aqi
        w.sunrise = sun["sunrise"]
        w.sunset = sun["sunset"]
        return {"data_map": {loc: w}}

    def node_advisory(state: GraphState):
        advisories = [
            _call(generate_activity_advice, loc, w.current_temp, w.air_quality_index)
            for loc, w in state["data_map"].items()
        ]
        return {"advisories": advisories}

    def node_compare(state: GraphState):
        if len(state["data_map"]) > 1:
            cmp_summary = _call(compare_locations, list(state["data_map"].keys()), state["data_map"])
            return {"comparison": cmp_summary}
        return {}

    def node_finalize(state: GraphState):
        report = WeatherReport(
            query=state["query"],
            data=list(state["data_map"].values()),
            advisories=state.get("advisories", []),
            comparison=state.get("comparison"),
            generated_at=datetime.now(timezone.utc),
        )
        return {"report": report}

    workflow = StateGraph(GraphState)
    workflow.add_node("parse", node_parse)
    workflow.add_node("worker_fetch", worker_fetch)
    workflow.add_node("advisory", node_advisory)
    workflow.add_node("compare", node_compare)
    workflow.add_node("finalize", node_finalize)
    workflow.add_edge(START, "parse")
    workflow.add_conditional_edges("parse", node_parallel_fetch, ["worker_fetch"])
    workflow.add_edge("worker_fetch", "advisory")
    workflow.add_edge("advisory", "compare")
    workflow.add_edge("compare", "finalize")
    workflow.add_edge("finalize", END)
    GRAPH_APP = workflow.compile()
else:
    GRAPH_APP = None

# ---------------------------------------------------------------------------
# Helper to run supervisor agent (single turn for demo)
# ---------------------------------------------------------------------------

def run_supervisor(user_query: str) -> WeatherReport:
    with tracer.start_as_current_span("run.supervisor") as span:
        parsed = parse_weather_query(user_query)
        span.set_attribute("locs", ",".join(parsed.locations))
        data_items: List[RawWeatherData] = []
        for loc in parsed.locations:
            # Use _call wrapper to handle StructuredTool vs plain function uniformly
            canonical = _call(geocode_location, loc)
            w = _call(fetch_weather, canonical, units=parsed.units, days=parsed.days)
            aqi = _call(fetch_air_quality, canonical) if parsed.include_air_quality else None
            if aqi is not None:
                w.air_quality_index = aqi
            data_items.append(w)
        advisories = [
            _call(
                generate_activity_advice,
                d.location,
                d.current_temp,
                d.air_quality_index if parsed.include_air_quality else None,
            )
            for d in data_items
        ]
        comparison = _call(
            compare_locations,
            parsed.locations,
            {d.location: d for d in data_items},
        ) if len(parsed.locations) > 1 else None
        return WeatherReport(
            query=parsed,
            data=data_items,
            advisories=advisories,
            comparison=comparison,
            generated_at=datetime.now(timezone.utc),
        )

# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    graph_only = False
    if "--graph-only" in args:
        graph_only = True
        args = [a for a in args if a != "--graph-only"]
    query = args[0] if args else "Weather for Paris tomorrow and is it good for a morning run? Also compare with Berlin"

    if not graph_only:
        print("=== Supervisor Deterministic Path ===")
        sup_report = run_supervisor(query)
        print(json.dumps(sup_report.model_dump(), indent=2, default=str))

    if GRAPH_APP:
        print("\n=== LangGraph Workflow Path (stream updates) ===")
        init_state: GraphState = {
            "input": query,
            "query": None,
            "locations": [],
            "data_map": {},
            "advisories": [],
            "comparison": None,
            "report": None,
        }
        try:
            for chunk in GRAPH_APP.stream(init_state, stream_mode="updates"):
                for node_name, node_state in chunk.items():
                    print(f"[update] node={node_name} keys={list(node_state.keys())}")
        except Exception as e:  # pragma: no cover
            print("Stream failed:", e)
        final = GRAPH_APP.invoke(init_state)
        report = final.get("report")
        if report:
            print("\nFinal Graph Report:")
            print(json.dumps(report.model_dump(), indent=2, default=str))
        else:
            print("Graph failed to produce report.")
    else:
        print("LangGraph not available; skipping workflow path.")

    print("\nDone.")


if __name__ == "__main__":
    main()
