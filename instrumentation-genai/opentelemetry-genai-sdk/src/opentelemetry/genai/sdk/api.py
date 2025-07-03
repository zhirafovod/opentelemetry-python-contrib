from datetime import datetime
from threading import Lock
from .types import LLMInvocation, ToolInvocation
from .exporters import SpanMetricEventExporter, SpanMetricExporter

class TelemetryClient:
    """
    High-level client managing GenAI invocation lifecycles and exporting
    them as spans, metrics, and events.
    """
    def __init__(self, tracer=None, meter=None, exporter_type: str = "full"):
        if exporter_type == "full":
            self._exporter = SpanMetricEventExporter(tracer=tracer, meter=meter)
        elif exporter_type == "span_metric":
            self._exporter = SpanMetricExporter(tracer=tracer, meter=meter)
        else:
            raise ValueError(f"Unknown exporter_type: {exporter_type}")

        self._llm_registry: dict[str, LLMInvocation] = {}
        self._tool_registry: dict[str, ToolInvocation] = {}
        self._lock = Lock()

    def start_llm(self, model_name: str, prompt: str, **attrs) -> str:
        invocation = LLMInvocation(model_name=model_name, prompt=prompt, attributes=attrs)
        with self._lock:
            self._llm_registry[invocation.invocation_id] = invocation
        return invocation.invocation_id

    def stop_llm(self, run_id: str, response: str, **attrs) -> LLMInvocation:
        with self._lock:
            invocation = self._llm_registry.pop(run_id)
        invocation.end_time = datetime.utcnow()
        invocation.response = response
        invocation.attributes.update(attrs)
        self._exporter.export(invocation)
        return invocation

    def fail_llm(self, run_id: str, error: str, **attrs) -> LLMInvocation:
        with self._lock:
            invocation = self._llm_registry.pop(run_id)
        invocation.end_time = datetime.utcnow()
        invocation.attributes.update({"error": error, **attrs})
        self._exporter.export(invocation)
        return invocation

    def start_tool(self, tool_name: str, input: dict, **attrs) -> str:
        invocation = ToolInvocation(tool_name=tool_name, input=input, attributes=attrs)
        with self._lock:
            self._tool_registry[invocation.invocation_id] = invocation
        return invocation.invocation_id

    def stop_tool(self, run_id: str, output: dict, **attrs) -> ToolInvocation:
        with self._lock:
            invocation = self._tool_registry.pop(run_id)
        invocation.end_time = datetime.utcnow()
        invocation.output = output
        invocation.attributes.update(attrs)
        self._exporter.export(invocation)
        return invocation

    def fail_tool(self, run_id: str, error: str, **attrs) -> ToolInvocation:
        with self._lock:
            invocation = self._tool_registry.pop(run_id)
        invocation.end_time = datetime.utcnow()
        invocation.attributes.update({"error": error, **attrs})
        self._exporter.export(invocation)
        return invocation

# Singleton accessor
_default_client: TelemetryClient | None = None

def get_telemetry_client(tracer=None, meter=None, exporter_type: str = "full") -> TelemetryClient:
    global _default_client
    if _default_client is None:
        _default_client = TelemetryClient(tracer=tracer, meter=meter, exporter_type=exporter_type)
    return _default_client

# Module‐level convenience functions
def llm_start(model_name: str, prompt: str, **attrs) -> str:
    return get_telemetry_client().start_llm(model_name, prompt, **attrs)

def llm_stop(run_id: str, response: str, **attrs) -> LLMInvocation:
    return get_telemetry_client().stop_llm(run_id, response, **attrs)

def llm_fail(run_id: str, error: str, **attrs) -> LLMInvocation:
    return get_telemetry_client().fail_llm(run_id, error, **attrs)

def tool_start(tool_name: str, input: dict, **attrs) -> str:
    return get_telemetry_client().start_tool(tool_name, input, **attrs)

def tool_stop(run_id: str, output: dict, **attrs) -> ToolInvocation:
    return get_telemetry_client().stop_tool(run_id, output, **attrs)

def tool_fail(run_id: str, error: str, **attrs) -> ToolInvocation:
    return get_telemetry_client().fail_tool(run_id, error, **attrs)
