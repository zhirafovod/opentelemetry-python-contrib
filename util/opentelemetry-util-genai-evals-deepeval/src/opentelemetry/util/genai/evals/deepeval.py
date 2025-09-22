# DeepEval integration for opentelemetry-utils-genai
# Extracted minimal evaluator to plug into TelemetryHandler via dynamic import

from opentelemetry.util.genai.evaluators.base import Evaluator
from opentelemetry.util.genai.types import (
    Error,
    EvaluationResult,
    LLMInvocation,
)

try:
    import deepeval
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
except ImportError:
    # If deepeval library not installed, will raise during evaluate
    deepeval = None


class DeepEvalEvaluator(Evaluator):
    """
    Uses deepeval library for performing multiple metrics on LLMInvocation.
    Returns a list of EvaluationResult, one per metric.
    """

    def __init__(self, event_logger=None, tracer=None, config: dict = None):
        self.config = config or {}
        self._event_logger = event_logger
        self._tracer = tracer

    def evaluate(self, invocation: LLMInvocation):
        if deepeval is None:
            # deepeval not installed; return error result
            return [
                EvaluationResult(
                    metric_name="deepeval",
                    error=Error(
                        message="deepeval library not installed",
                        type=ImportError,
                    ),
                )
            ]
        # Collect text content from first messages
        try:
            prompt = invocation.input_messages[0].parts[0].content
            output = invocation.output_messages[0].parts[0].content
            test_case = LLMTestCase(
                input=prompt, actual_output=output, retrieval_context=[]
            )
            # Example: run a relevancy metric; extend as needed
            relevancy = AnswerRelevancyMetric(threshold=0.5)
            relevancy.measure(test_case)
            score = getattr(relevancy, "score", None)
            label = getattr(relevancy, "reason", None)
            return [
                EvaluationResult(
                    metric_name="relevance", score=score, label=label
                )
            ]
        except Exception as exc:
            return [
                EvaluationResult(
                    metric_name="deepeval",
                    error=Error(message=str(exc), type=type(exc)),
                )
            ]
