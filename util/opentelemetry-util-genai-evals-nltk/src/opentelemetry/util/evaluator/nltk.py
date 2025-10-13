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
"""NLTK-based sentiment evaluator plug-in."""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

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
    Text,
)


def _extract_text(invocation: LLMInvocation) -> str:
    parts: List[str] = []
    for message in invocation.output_messages:
        for part in getattr(message, "parts", []):
            if isinstance(part, Text):
                parts.append(part.content)
    return "\n".join(part for part in parts if part).strip()


class NLTKSentimentEvaluator(Evaluator):
    """Evaluator that scores sentiment using NLTK's VADER analyser.

    Updated to support both ``LLMInvocation`` and ``AgentInvocation`` so that it
    participates in the new evaluation manager default auto-discovery (which
    consumes per-type default metric mappings). The evaluator remains intentionally
    lightweight; any heavy text extraction should stay minimal to avoid adding
    latency to completion callbacks.
    """

    # ---- Defaults -------------------------------------------------
    def default_metrics(self) -> Sequence[str]:  # pragma: no cover - trivial
        return ("sentiment",)

    def default_metrics_by_type(
        self,
    ) -> Mapping[str, Sequence[str]]:  # pragma: no cover - trivial
        # Provide defaults for both LLM and Agent invocation types so that
        # sentiment is auto-enabled alongside other evaluators (e.g. Deepeval)
        return {
            "LLMInvocation": ("sentiment",),
            "AgentInvocation": ("sentiment",),
        }

    # ---- Unified evaluation dispatch (optional override) ---------
    def evaluate(
        self, item: GenAI
    ) -> list[EvaluationResult]:  # pragma: no cover - exercised indirectly
        if isinstance(item, LLMInvocation):
            return list(self.evaluate_llm(item))
        if isinstance(item, AgentInvocation):
            return list(self.evaluate_agent(item))
        return []

    # ---- Type specific hooks -------------------------------------
    def evaluate_llm(
        self, invocation: LLMInvocation
    ) -> Sequence[EvaluationResult]:
        return self._score_text(_extract_text(invocation))

    def evaluate_agent(
        self, invocation: AgentInvocation
    ) -> Sequence[EvaluationResult]:
        # Prefer explicit output result text; fallback to combining system instructions
        # and input context if present.
        chunks: list[str] = []
        if invocation.output_result:
            chunks.append(str(invocation.output_result))
        else:
            if invocation.system_instructions:
                chunks.append(str(invocation.system_instructions))
            if invocation.input_context:
                chunks.append(str(invocation.input_context))
        text = "\n".join(c for c in chunks if c).strip()
        return self._score_text(text)

    # ---- Internal helper ------------------------------------------
    def _score_text(self, content: str) -> Sequence[EvaluationResult]:
        metric_name = self.metrics[0] if self.metrics else "sentiment"
        try:
            from nltk.sentiment import (
                SentimentIntensityAnalyzer,  # type: ignore
            )

            if not hasattr(
                __import__("nltk.sentiment").sentiment,  # type: ignore
                "SentimentIntensityAnalyzer",
            ):
                raise AttributeError("SentimentIntensityAnalyzer missing")
        except Exception as exc:  # pragma: no cover - defensive fallback
            return [
                EvaluationResult(
                    metric_name=metric_name,
                    error=Error(
                        message="nltk (vader) not installed",
                        type=type(exc),
                    ),
                )
            ]
        if not content:
            return [
                EvaluationResult(
                    metric_name=metric_name,
                    score=0.0,
                    label="neutral",
                )
            ]
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(content)
        compound = scores.get("compound", 0.0)
        score = (compound + 1) / 2
        if compound >= 0.2:
            label = "positive"
        elif compound <= -0.2:
            label = "negative"
        else:
            label = "neutral"
        return [
            EvaluationResult(
                metric_name=metric_name,
                score=score,
                label=label,
                explanation=f"compound={compound}",
            )
        ]


def _factory(
    metrics: Iterable[str] | None = None,
    invocation_type: str | None = None,
    options: Mapping[str, Mapping[str, str]]
    | None = None,  # accepted for compatibility but ignored
) -> NLTKSentimentEvaluator:
    return NLTKSentimentEvaluator(
        metrics,
        invocation_type=invocation_type,
    )


_REGISTRATION = EvaluatorRegistration(
    factory=_factory,
    # Match Deepeval style: supply full per-type defaults so manager auto-enables.
    default_metrics_factory=lambda: {
        "LLMInvocation": ("sentiment",),
        "AgentInvocation": ("sentiment",),
    },
)


def registration() -> EvaluatorRegistration:
    return _REGISTRATION


def register() -> None:
    register_evaluator(
        "nltk_sentiment",
        _REGISTRATION.factory,
        default_metrics=_REGISTRATION.default_metrics_factory,
    )


__all__ = [
    "NLTKSentimentEvaluator",
    "registration",
    "register",
]
