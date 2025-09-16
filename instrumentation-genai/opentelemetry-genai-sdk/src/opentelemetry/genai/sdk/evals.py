from abc import ABC, abstractmethod
from opentelemetry._events import Event

from .types import LLMInvocation
from opentelemetry import trace
from opentelemetry.trace import (
    Tracer,
)
from opentelemetry import _events
from .deepeval import evaluate_all_metrics
from opentelemetry.trace import SpanContext, Span
from opentelemetry.trace.span import NonRecordingSpan


class EvaluationResult:
    """
    Standardized result for any GenAI evaluation.
    """
    def __init__(self, score: float, details: dict = None):
        self.score = score
        self.details = details or {}


class Evaluator(ABC):
    """
    Abstract base: any evaluation backend must implement.
    """
    @abstractmethod
    def evaluate(self, invocation: LLMInvocation) -> EvaluationResult:
        """
        Evaluate a completed LLMInvocation and return a result.
        """
        pass

class DeepEvalEvaluator(Evaluator):
    """
    Uses DeepEvals library for LLM-as-judge evaluations.
    """
    def __init__(self, event_logger, tracer: Tracer = None, config: dict = None):
        # e.g. load models, setup API keys
        self.config = config or {}
        self._tracer = tracer or trace.get_tracer(__name__)
        self._event_logger = event_logger or _events.get_event_logger(__name__)

    def evaluate(self, invocation: LLMInvocation):
        # stub: integrate with deepevals SDK
        # result = deepevals.judge(invocation.prompt, invocation.response, **self.config)
        human_message = next((msg for msg in invocation.messages if msg.type == "human"), None)
        content = invocation.chat_generations[0].content
        if content is not None and content != "":
            eval_results = evaluate_all_metrics(human_message.content, invocation.chat_generations[0].content, [])
            self._do_telemetry(invocation, eval_results)

    def _do_telemetry(self, invocation: LLMInvocation, eval_results):
        import datetime

        # Create event body with LLM invocation context
        body = {
            "name": invocation.attributes.get("gen_ai.response.model", "unknown"),
            **invocation.attributes,  # Include all LLM attributes
            "kind": "SpanKind.INTERNAL",
            "parent_id": None,
            "start_time": datetime.datetime.fromtimestamp(invocation.start_time).isoformat() + "Z",
            "end_time": datetime.datetime.fromtimestamp(invocation.end_time or invocation.start_time).isoformat() + "Z",
            "status": {"status_code": "UNSET"},
            "events": [],
            "links": [],
            "resource": {
                "attributes": {
                    "telemetry.sdk.language": "python",
                    "telemetry.sdk.name": "opentelemetry",
                    "telemetry.sdk.version": "1.36.0",
                    "service.name": "unknown_service"
                },
                "schema_url": ""
            }
        }
        
        # Event attributes contain evaluation data for all metrics
        attributes = {
            "gen_ai.operation.name": "evaluation"
        }
        
        # Add attributes for each evaluation metric
        for metric_name, metric_data in eval_results.items():
            if metric_name != "error" and isinstance(metric_data, dict):
                # Rename answerrelevancy to relevance
                if metric_name == "answerrelevancy":
                    metric_name = "relevance"
                
                # Add to attributes
                # attributes[f"gen_ai.evaluation.{metric_name}.score"] = metric_data.get("score", 0)
                # attributes[f"gen_ai.evaluation.{metric_name}.label"] = metric_data.get("label", "Unknown")
                # attributes[f"gen_ai.evaluation.{metric_name}.range"] = metric_data.get("range", "[0,1]")
                # attributes[f"gen_ai.evaluation.{metric_name}.reasoning"] = metric_data.get("reason", "")
                # attributes[f"gen_ai.evaluation.{metric_name}.judge_model"] = metric_data.get("judge_model", "unknown")
                
                # Add to body with same naming as attributes
                body[f"gen_ai.evaluation.{metric_name}.score"] = metric_data.get("score", 0)
                body[f"gen_ai.evaluation.{metric_name}.label"] = metric_data.get("label", "Unknown")
                body[f"gen_ai.evaluation.{metric_name}.range"] = metric_data.get("range", "[0,1]")
                body[f"gen_ai.evaluation.{metric_name}.reasoning"] = metric_data.get("reason", "")
                body[f"gen_ai.evaluation.{metric_name}.judge_model"] = metric_data.get("judge_model", "unknown")

        event = Event(
            name="gen_ai.evaluation.message",
            attributes=attributes,
            body=body if body else None,
            span_id=invocation.span_id,
            trace_id=invocation.trace_id,
        )
        self._event_logger.emit(event)

        # create span
        span_context = SpanContext(
            trace_id=invocation.trace_id,
            span_id=invocation.span_id,
            is_remote=False,
        )

        span = NonRecordingSpan(
            context=span_context,
        )

        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("evaluation relevance") as span:
            # do evaluation

            span.add_link(span_context, attributes={
                "gen_ai.operation.name": "evaluation",
            })
            span.set_attribute("gen_ai.operation.name", "evaluation")
            
            for attr_key, attr_value in attributes.items():
                if attr_key != "gen_ai.operation.name":  # Skip duplicate
                    span.set_attribute(attr_key, attr_value)


class OpenLitEvaluator(Evaluator):
    """
    Uses OpenLit or similar OSS evaluation library.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}

    def evaluate(self, invocation: LLMInvocation) -> EvaluationResult:
        # stub: integrate with openlit SDK
        score = 0.0  # placeholder
        details = {"method": "openlit"}
        return EvaluationResult(score=score, details=details)


# Registry for easy lookup
EVALUATORS = {
    "deepeval": DeepEvalEvaluator,
    "openlit": OpenLitEvaluator,
}


def get_evaluator(name: str, event_logger = None, tracer: Tracer = None, config: dict = None) -> Evaluator:
    """
    Factory: return an evaluator by name.
    """
    cls = EVALUATORS.get(name.lower())
    if not cls:
        raise ValueError(f"Unknown evaluator: {name}")
    return cls(event_logger, tracer, config)