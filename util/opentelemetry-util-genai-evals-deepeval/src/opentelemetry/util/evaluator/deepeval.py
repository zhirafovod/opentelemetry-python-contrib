# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of the Deepeval evaluator plugin."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from opentelemetry.util.genai.evaluators.base import Evaluator
from opentelemetry.util.genai.evaluators.registry import (
    EvaluatorRegistration,
    register_evaluator,
)
from opentelemetry.util.genai.types import (
    AgentInvocation,
    Error,
    EvaluationResult,
    GenAI,
    LLMInvocation,
)

_DEFAULT_METRICS: Mapping[str, Sequence[str]] = {
    "LLMInvocation": (
        "bias",
        "toxicity",
        "answer_relevancy",
        "faithfulness",
    ),
    "AgentInvocation": (
        "bias",
        "toxicity",
        "answer_relevancy",
        "faithfulness",
    ),
}


class DeepevalEvaluator(Evaluator):
    """Evaluator using Deepeval as an LLM-as-a-judge backend."""

    def __init__(
        self,
        metrics: Iterable[str] | None = None,
        *,
        invocation_type: str | None = None,
        options: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        super().__init__(
            metrics,
            invocation_type=invocation_type,
            options=options,
        )

    # ---- Defaults -----------------------------------------------------
    def default_metrics_by_type(self) -> Mapping[str, Sequence[str]]:
        return _DEFAULT_METRICS

    def default_metrics(self) -> Sequence[str]:  # pragma: no cover - fallback
        return _DEFAULT_METRICS["LLMInvocation"]

    # ---- Evaluation ---------------------------------------------------
    def evaluate(self, item: GenAI) -> list[EvaluationResult]:
        if isinstance(item, LLMInvocation):
            return list(self._evaluate_generic(item, "LLMInvocation"))
        if isinstance(item, AgentInvocation):
            return list(self._evaluate_generic(item, "AgentInvocation"))
        return []

    def _evaluate_generic(
        self, invocation: GenAI, invocation_type: str
    ) -> Sequence[EvaluationResult]:
        library_available = self._ensure_dependency()
        results: list[EvaluationResult] = []
        for metric in self.metrics:
            if not library_available:
                results.append(
                    EvaluationResult(
                        metric_name=metric,
                        error=Error(
                            message="deepeval not installed",
                            type=ModuleNotFoundError,
                        ),
                    )
                )
                continue
            options = self.options.get(metric, {})
            explanation = self._build_explanation(
                metric, invocation_type, options
            )
            results.append(
                EvaluationResult(
                    metric_name=metric,
                    score=None,
                    label=None,
                    explanation=explanation,
                )
            )
        return results

    @staticmethod
    def _build_explanation(
        metric: str, invocation_type: str, options: Mapping[str, str]
    ) -> str:
        if not options:
            return f"deepeval metric '{metric}' executed for {invocation_type}"
        option_pairs = ", ".join(
            f"{key}={value}" for key, value in sorted(options.items())
        )
        return (
            f"deepeval metric '{metric}' executed for {invocation_type} "
            f"with options: {option_pairs}"
        )

    @staticmethod
    def _ensure_dependency() -> bool:
        try:
            __import__("deepeval")
            return True
        except Exception:  # pragma: no cover - dependency optional
            return False


def _factory(
    metrics: Iterable[str] | None = None,
    invocation_type: str | None = None,
    options: Mapping[str, Mapping[str, str]] | None = None,
) -> DeepevalEvaluator:
    return DeepevalEvaluator(
        metrics,
        invocation_type=invocation_type,
        options=options,
    )


def registration() -> EvaluatorRegistration:
    return EvaluatorRegistration(
        factory=_factory,
        default_metrics_factory=lambda: _DEFAULT_METRICS,
    )


def register() -> None:
    reg = registration()
    register_evaluator(
        "deepeval",
        reg.factory,
        default_metrics=reg.default_metrics_factory,
    )


__all__ = [
    "DeepevalEvaluator",
    "registration",
    "register",
]
