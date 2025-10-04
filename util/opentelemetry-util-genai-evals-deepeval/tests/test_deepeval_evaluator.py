import importlib
import sys
from unittest.mock import patch

import pytest

from opentelemetry.util.evaluator import deepeval as plugin
from opentelemetry.util.genai.evaluators.registry import (
    clear_registry,
    get_evaluator,
    list_evaluators,
)
from opentelemetry.util.genai.types import LLMInvocation


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    importlib.reload(plugin)
    plugin.register()
    yield
    clear_registry()


def _build_invocation() -> LLMInvocation:
    invocation = LLMInvocation(request_model="test-model")
    return invocation


def test_registration_adds_deepeval() -> None:
    names = list_evaluators()
    assert "deepeval" in names


def test_default_metrics_covered() -> None:
    evaluator = get_evaluator("deepeval")
    assert set(evaluator.metrics) == {
        "bias",
        "toxicity",
        "answer_relevancy",
        "faithfulness",
    }


def test_evaluation_errors_when_dependency_missing() -> None:
    invocation = _build_invocation()
    evaluator = get_evaluator("deepeval")
    with patch.dict(sys.modules, {"deepeval": None}):
        results = evaluator.evaluate(invocation)
    assert len(results) == len(evaluator.metrics)
    assert all(result.error is not None for result in results)


def test_metric_options_used_in_explanation() -> None:
    invocation = _build_invocation()
    evaluator = get_evaluator(
        "deepeval",
        ("bias",),
        invocation_type="LLMInvocation",
        options={"bias": {"threshold": "0.7"}},
    )
    with patch(
        "opentelemetry.util.evaluator.deepeval.DeepevalEvaluator._ensure_dependency",
        return_value=True,
    ):
        results = evaluator.evaluate(invocation)
    assert len(results) == 1
    explanation = results[0].explanation
    assert explanation is not None
    assert "threshold=0.7" in explanation
