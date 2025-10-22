import sys
import types

from opentelemetry.util.evaluator.nltk import NLTKSentimentEvaluator
from opentelemetry.util.genai.types import AgentInvocation


def _install_stub_analyzer(compound: float = 0.1):
    sentiment_module = types.ModuleType("nltk.sentiment")

    class _Analyzer:
        def polarity_scores(self, text):  # pragma: no cover - simple stub
            return {"compound": compound}

    sentiment_module.SentimentIntensityAnalyzer = _Analyzer
    nltk_module = types.ModuleType("nltk")
    nltk_module.sentiment = sentiment_module
    sys.modules["nltk"] = nltk_module
    sys.modules["nltk.sentiment"] = sentiment_module
    return lambda: (
        sys.modules.pop("nltk", None),
        sys.modules.pop("nltk.sentiment", None),
    )


def test_agent_invocation_sentiment():
    cleanup = _install_stub_analyzer(compound=-0.5)  # negative sentiment
    try:
        agent = AgentInvocation(
            name="demo-agent",
            system_instructions="You are a helpful agent",
            input_context="Process order",
            output_result="This is terrible service!",
        )
        evaluator = NLTKSentimentEvaluator()
        results = evaluator.evaluate_agent(agent)
        assert results
        result = results[0]
        assert result.metric_name == "sentiment"
        assert result.label == "negative"
        assert (result.score or 0) < 0.5  # scaled score
    finally:
        cleanup()
