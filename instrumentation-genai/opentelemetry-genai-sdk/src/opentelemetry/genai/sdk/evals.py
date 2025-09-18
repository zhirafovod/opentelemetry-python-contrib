from abc import ABC, abstractmethod
from opentelemetry._events import Event

from .types import LLMInvocation
from opentelemetry import trace
from opentelemetry.trace import (
    Tracer,
)
from opentelemetry import metrics
from opentelemetry.metrics import Histogram
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
        meter = metrics.get_meter(__name__)
        self._metric_relevance: Histogram = meter.create_histogram(
            name="gen_ai.evaluation.relevance",
            unit="1",
            description="Evaluation score for relevance in [0,1]",
        )
        self._metric_hallucination: Histogram = meter.create_histogram(
            name="gen_ai.evaluation.hallucination",
            unit="1",
            description="Evaluation score for hallucination in [0,1]",
        )
        self._metric_toxicity: Histogram = meter.create_histogram(
            name="gen_ai.evaluation.toxicity",
            unit="1",
            description="Evaluation score for toxicity in [0,1]",
        )
        self._metric_bias: Histogram = meter.create_histogram(
            name="gen_ai.evaluation.bias",
            unit="1",
            description="Evaluation score for bias in [0,1]",
        )
        self._metric_sentiment: Histogram = meter.create_histogram(
            name="gen_ai.evaluation.sentiment",
            unit="1",
            description="Evaluation score for sentiment in [0,1]",
        )

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

        body = {}
    
        attributes = {
            "gen_ai.operation.name": "evaluation",
            "gen_ai.request.model": invocation.attributes.get("request_model"),
            "gen_ai.provider.name": invocation.attributes.get("provider_name"),
        }
        if invocation.attributes:
            attributes.update(invocation.attributes)

        if isinstance(eval_results, dict) and isinstance(eval_results.get("error"), dict):
            err_type = eval_results.get("error", {}).get("type")
            if err_type:
                attributes["error.type"] = err_type
        
        evaluations: list[dict] = []
        for metric_name, metric_data in eval_results.items():
            if metric_name != "error" and isinstance(metric_data, dict):
                if metric_name == "answerrelevancy":
                    metric_name = "relevance"

                name_lower = metric_name.lower()
                score_val = metric_data.get("score", 0)
                label_val = metric_data.get("label", "Unknown")
                explanation_val = metric_data.get("reason", "")
                # TODO: check if this should be the response id for chat llm invocation or eval LLM-as-judge
                response_id = invocation.attributes.get("response_id") if invocation.attributes else None

                eval_item = {
                    "gen_ai.evaluation.name": name_lower,
                    "gen_ai.evaluation.score.value": score_val,
                    "gen_ai.evaluation.score.label": label_val,
                    "gen_ai.evaluation.explanation": explanation_val,
                }
                if response_id:
                    eval_item["gen_ai.response.id"] = response_id
                # include error.type if present at top-level
                err = eval_results.get("error") if isinstance(eval_results, dict) else None
                if isinstance(err, dict) and err.get("type"):
                    eval_item["error.type"] = err.get("type")

                evaluations.append(eval_item)

                # record metric
                metric_attrs = {
                    "gen_ai.operation.name": "evaluation",
                    "gen_ai.request.model": invocation.attributes.get("request_model"),
                    "gen_ai.provider.name": invocation.attributes.get("provider_name"),
                    "gen_ai.evaluation.score.label": label_val,
                }
                if isinstance(err, dict) and err.get("type"):
                    metric_attrs["error.type"] = err.get("type")
                if name_lower == "relevance":
                    self._metric_relevance.record(score_val, attributes={k: v for k, v in metric_attrs.items() if v is not None})
                elif name_lower == "hallucination":
                    self._metric_hallucination.record(score_val, attributes={k: v for k, v in metric_attrs.items() if v is not None})
                elif name_lower == "toxicity":
                    self._metric_toxicity.record(score_val, attributes={k: v for k, v in metric_attrs.items() if v is not None})
                elif name_lower == "bias":
                    self._metric_bias.record(score_val, attributes={k: v for k, v in metric_attrs.items() if v is not None})
                elif name_lower == "sentiment":
                    self._metric_sentiment.record(score_val, attributes={k: v for k, v in metric_attrs.items() if v is not None})

        if evaluations:
            body["gen_ai.evaluations"] = evaluations

        event = Event(
            name="gen_ai.evaluation.results",
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