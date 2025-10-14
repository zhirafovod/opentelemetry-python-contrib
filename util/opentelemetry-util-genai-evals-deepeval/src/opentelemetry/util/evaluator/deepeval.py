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

import logging
import os
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

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

_DEFAULT_METRICS: Mapping[str, Sequence[str]] = {
    "LLMInvocation": (
        "bias",
        "toxicity",
        "answer_relevancy",
        "faithfulness",
        "hallucination",
        "sentiment",
    ),
    "AgentInvocation": (
        "bias",
        "toxicity",
        "answer_relevancy",
        "faithfulness",
        "hallucination",
        "sentiment",
    ),
}


_LOGGER = logging.getLogger(__name__)


# Disable Deepeval's internal telemetry (Posthog/New Relic) by default so that
# it does not emit extra spans or events when running inside the GenAI
# instrumentation stack. Users can re-enable it by explicitly setting
# ``DEEPEVAL_TELEMETRY_OPT_OUT`` to ``0`` before importing this module.
if os.environ.get("DEEPEVAL_TELEMETRY_OPT_OUT") is None:
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"


@dataclass(frozen=True)
class _MetricSpec:
    name: str
    options: Mapping[str, Any]


_METRIC_REGISTRY: Mapping[str, str] = {
    # name -> deepeval.metrics class attribute or sentinel for custom build
    "bias": "BiasMetric",
    "toxicity": "ToxicityMetric",
    "answer_relevancy": "AnswerRelevancyMetric",
    # Synonyms for answer relevancy accepted in user configuration; all map to
    # the same underlying Deepeval metric class. Emission canonicalization
    # later will normalize variants (answer relevancy / answer_relevancy / answer relevance)
    "answer_relevance": "AnswerRelevancyMetric",
    "relevance": "AnswerRelevancyMetric",
    "faithfulness": "FaithfulnessMetric",
    "hallucination": "__custom_hallucination__",  # custom GEval metric
    "sentiment": "__custom_sentiment__",  # custom GEval sentiment metric
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
            return list(self._evaluate_llm(item))
        if isinstance(item, AgentInvocation):
            return list(self._evaluate_agent(item))
        return []

    def _evaluate_llm(
        self, invocation: LLMInvocation
    ) -> Sequence[EvaluationResult]:
        return self._evaluate_generic(invocation, "LLMInvocation")

    def _evaluate_agent(
        self, invocation: AgentInvocation
    ) -> Sequence[EvaluationResult]:
        return self._evaluate_generic(invocation, "AgentInvocation")

    def _evaluate_generic(
        self, invocation: GenAI, invocation_type: str
    ) -> Sequence[EvaluationResult]:
        metric_specs = self._build_metric_specs()
        if not metric_specs:
            return []
        test_case = self._build_test_case(invocation, invocation_type)
        if test_case is None:
            return self._error_results(
                "Deepeval requires both input and output text to evaluate",
                ValueError,
            )
        # Ensure OpenAI API key is available for Deepeval metrics that rely on OpenAI.
        # Resolution order:
        # 1. Explicit in invocation.attributes['openai_api_key'] (if provided)
        # 2. Environment OPENAI_API_KEY
        # 3. Environment GENAI_OPENAI_API_KEY (custom fallback)
        # If unavailable we mark all metrics skipped with a clear explanation instead of raising.
        api_key: str | None = None
        try:
            raw_attrs = getattr(invocation, "attributes", None)
            attrs: dict[str, Any] = {}
            if isinstance(raw_attrs, MappingABC):
                for k, v in raw_attrs.items():
                    try:
                        attrs[str(k)] = v
                    except Exception:  # pragma: no cover
                        continue
            candidate_val = attrs.get("openai_api_key") or attrs.get("api_key")
            candidate: str | None = (
                str(candidate_val)
                if isinstance(candidate_val, (str, bytes))
                else None
            )
            env_key = os.getenv("OPENAI_API_KEY") or os.getenv(
                "GENAI_OPENAI_API_KEY"
            )
            api_key = candidate or env_key
            if api_key:
                # Attempt to configure Deepeval/OpenAI client.
                try:  # pragma: no cover - external dependency
                    import openai  # noqa: F401

                    # Support legacy openai<1 and new openai>=1 semantics.
                    if not getattr(openai, "api_key", None):  # type: ignore[attr-defined]
                        try:
                            setattr(openai, "api_key", api_key)  # legacy style
                        except Exception:  # pragma: no cover
                            pass
                    # Ensure env var set for client() style usage.
                    if not os.getenv("OPENAI_API_KEY"):
                        os.environ["OPENAI_API_KEY"] = api_key
                except Exception:
                    pass
        except Exception:  # pragma: no cover - defensive
            api_key = None
        # Do not fail early if API key missing; underlying Deepeval/OpenAI usage
        # will produce an error which we surface as evaluation error results.
        try:
            metrics, skipped_results = self._instantiate_metrics(
                metric_specs, test_case
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._error_results(str(exc), type(exc))
        if not metrics:
            return skipped_results or self._error_results(
                "No Deepeval metrics available", RuntimeError
            )
        try:
            evaluation = self._run_deepeval(test_case, metrics)
        except (
            Exception
        ) as exc:  # pragma: no cover - dependency/runtime failure
            return [
                *skipped_results,
                *self._error_results(str(exc), type(exc)),
            ]
        return [*skipped_results, *self._convert_results(evaluation)]

    # ---- Helpers ------------------------------------------------------
    def _build_metric_specs(self) -> Sequence[_MetricSpec]:
        specs: list[_MetricSpec] = []
        registry = _METRIC_REGISTRY
        import re as _re

        for name in self.metrics:
            raw = (name or "").strip().lower()
            # Normalize any spaces / punctuation to underscores so that
            # variants like "answer relevancy" or "answer-relevance" resolve
            # to the canonical registry key "answer_relevancy".
            normalized = _re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
            key = normalized
            options = self.options.get(name, {})
            if key not in registry:
                specs.append(
                    _MetricSpec(
                        name=name,
                        options={
                            "__error__": f"Unknown Deepeval metric '{name}'",
                        },
                    )
                )
                continue
            parsed_options = {
                opt_key: self._coerce_option(opt_value)
                for opt_key, opt_value in options.items()
            }
            specs.append(_MetricSpec(name=key, options=parsed_options))
        return specs

    def _instantiate_metrics(  # pragma: no cover - exercised via tests
        self, specs: Sequence[_MetricSpec], test_case: Any
    ) -> tuple[Sequence[Any], Sequence[EvaluationResult]]:
        from importlib import import_module

        metrics_module = import_module("deepeval.metrics")
        registry = _METRIC_REGISTRY
        instances: list[Any] = []
        skipped: list[EvaluationResult] = []
        default_model = self._default_model()
        for spec in specs:
            if "__error__" in spec.options:
                raise ValueError(spec.options["__error__"])
            metric_class_name = registry[spec.name]
            if metric_class_name == "__custom_hallucination__":
                try:
                    instances.append(
                        self._build_hallucination_metric(
                            spec.options, default_model
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"Failed to instantiate hallucination metric: {exc}"
                    ) from exc
                continue
            if metric_class_name == "__custom_sentiment__":
                try:
                    instances.append(
                        self._build_sentiment_metric(
                            spec.options, default_model
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuntimeError(
                        f"Failed to instantiate sentiment metric: {exc}"
                    ) from exc
                continue
            metric_cls = getattr(metrics_module, metric_class_name, None)
            if metric_cls is None:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"Deepeval metric class '{metric_class_name}' not found"
                )
            missing = self._missing_required_params(metric_cls, test_case)
            if missing:
                message = (
                    "Missing required Deepeval test case fields "
                    f"{', '.join(missing)} for metric '{spec.name}'."
                )
                _LOGGER.info(
                    "Skipping Deepeval metric '%s': %s", spec.name, message
                )
                skipped.append(
                    EvaluationResult(
                        metric_name=spec.name,
                        label="skipped",
                        explanation=message,
                        error=Error(message=message, type=ValueError),
                        attributes={
                            "deepeval.error": message,
                            "deepeval.skipped": True,
                            "deepeval.missing_params": missing,
                        },
                    )
                )
                continue
            # spec.options already coerced into primitive values
            kwargs = dict(spec.options)
            if default_model and "model" not in kwargs:
                kwargs["model"] = default_model
            try:
                instances.append(metric_cls(**kwargs))
            except TypeError as exc:
                raise TypeError(
                    f"Failed to instantiate Deepeval metric '{spec.name}': {exc}"
                )
        return instances, skipped

    # ---- Custom metric builders ------------------------------------
    def _build_hallucination_metric(
        self, options: Mapping[str, Any], default_model: str | None
    ) -> Any:
        """Create a GEval-based hallucination metric (input/output only).

        Avoids complex dynamic getattr chains by importing the exact classes
        we need; falls back gracefully if signature differences are detected.
        """
        from deepeval.metrics import (
            GEval,  # direct import (simpler & explicit)
        )
        from deepeval.test_case import LLMTestCaseParams

        criteria = (
            "Assess if the output hallucinates by introducing facts, details, or claims not directly supported "
            "or implied by the input. Score 1 for fully grounded outputs (no fabrications) and 0 for severe hallucination."  # noqa: E501
        )
        steps = [
            "Review the input to extract all key facts and scope.",
            "Scan the output for any unsubstantiated additions, contradictions, or extrapolations beyond the input.",
            "Rate factual alignment: 1 = no hallucination, 0 = high hallucination risk.",
        ]
        threshold = options.get("threshold") if options else None
        model_override = options.get("model") if options else None
        strict_mode = options.get("strict_mode") if options else None
        if hasattr(LLMTestCaseParams, "INPUT_OUTPUT"):
            params = getattr(LLMTestCaseParams, "INPUT_OUTPUT")
        else:  # pragma: no cover - legacy fallback
            params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        kwargs: dict[str, Any] = {
            "name": "hallucination",
            "criteria": criteria,
            "evaluation_params": params,
            "threshold": threshold
            if isinstance(threshold, (int, float))
            else 0.7,
            "model": model_override or default_model or "gpt-4o-mini",
        }
        if strict_mode is not None:
            kwargs["strict_mode"] = bool(strict_mode)
        # Attempt primary signature (evaluation_steps), fallback to steps
        try:  # Preferred newer signature
            return GEval(evaluation_steps=steps, **kwargs)
        except TypeError:  # Older signature variant
            return GEval(steps=steps, **kwargs)

    def _build_sentiment_metric(
        self, options: Mapping[str, Any], default_model: str | None
    ) -> Any:
        """Create a GEval-based sentiment metric approximating VADER.

        The GEval prompt will ask the model to classify sentiment and produce a
        normalized score in [-1,1]. We then map to [0,1] for histogram score while
        deriving a pseudo VADER-style distribution (neg/neu/pos) emitted as attributes.
        """
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams

        criteria = "Determine the overall sentiment polarity of the output text: -1 very negative, 0 neutral, +1 very positive."
        steps = [
            "Read the text and note words/phrases indicating sentiment.",
            "Judge if overall tone is negative, neutral, or positive.",
            "Assign a numeric polarity in [-1,1] capturing intensity (e.g. strong positive ~0.8+).",
        ]
        if hasattr(LLMTestCaseParams, "ACTUAL_OUTPUT"):
            params = [LLMTestCaseParams.ACTUAL_OUTPUT]
        else:  # pragma: no cover - legacy fallback
            params = [LLMTestCaseParams.INPUT_OUTPUT]
        model_override = options.get("model") if options else None
        threshold = options.get("threshold") if options else None
        kwargs: dict[str, Any] = {
            "name": "sentiment [geval]",
            "criteria": criteria,
            "evaluation_params": params,
            "threshold": threshold
            if isinstance(threshold, (int, float))
            else 0.0,  # threshold not really used for pass/fail; keep 0
            "model": model_override or default_model or "gpt-4o-mini",
        }
        try:
            return GEval(evaluation_steps=steps, **kwargs)
        except TypeError:  # pragma: no cover - signature variant
            return GEval(steps=steps, **kwargs)

    def _build_test_case(
        self, invocation: GenAI, invocation_type: str
    ) -> Any | None:
        from deepeval.test_case import LLMTestCase

        if isinstance(invocation, LLMInvocation):
            input_text = self._serialize_messages(invocation.input_messages)
            output_text = self._serialize_messages(invocation.output_messages)
            context = self._extract_context(invocation)
            retrieval_context = self._extract_retrieval_context(invocation)
            if not input_text or not output_text:
                return None
            return LLMTestCase(
                input=input_text,
                actual_output=output_text,
                context=context,
                retrieval_context=retrieval_context,
                additional_metadata=invocation.attributes or None,
                name=invocation.request_model,
            )
        if isinstance(invocation, AgentInvocation):
            input_chunks: list[str] = []
            if invocation.system_instructions:
                input_chunks.append(str(invocation.system_instructions))
            if invocation.input_context:
                input_chunks.append(str(invocation.input_context))
            input_text = "\n\n".join(
                chunk
                for chunk in input_chunks
                if isinstance(chunk, str) and chunk
            )
            output_text = invocation.output_result or ""
            if not input_text or not output_text:
                return None
            context: list[str] | None = None
            if invocation.tools:
                context = ["Tools: " + ", ".join(invocation.tools)]
            return LLMTestCase(
                input=input_text,
                actual_output=output_text,
                context=context,
                retrieval_context=self._extract_retrieval_context(invocation),
                additional_metadata={
                    "agent_name": invocation.name,
                    "agent_type": invocation.agent_type,
                    **(invocation.attributes or {}),
                },
                name=invocation.operation,
            )
        return None

    def _run_deepeval(self, test_case: Any, metrics: Sequence[Any]) -> Any:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.evaluate.configs import AsyncConfig, DisplayConfig

        display_config = DisplayConfig(
            show_indicator=False, print_results=False
        )
        async_config = AsyncConfig(run_async=False)
        return deepeval_evaluate(
            [test_case],
            list(metrics),
            async_config=async_config,
            display_config=display_config,
        )

    def _convert_results(self, evaluation: Any) -> Sequence[EvaluationResult]:
        results: list[EvaluationResult] = []
        try:
            test_results = getattr(evaluation, "test_results", [])
        except Exception:  # pragma: no cover - defensive
            return self._error_results(
                "Unexpected Deepeval response", RuntimeError
            )
        for test in test_results:
            metrics_data = getattr(test, "metrics_data", []) or []
            for metric in metrics_data:
                name = getattr(metric, "name", "deepeval")
                raw_score = getattr(metric, "score", None)
                score: float | None
                if isinstance(raw_score, (int, float)):
                    score = float(raw_score)
                elif isinstance(raw_score, str):
                    try:
                        score = float(raw_score.strip())
                    except Exception:  # pragma: no cover - defensive
                        score = None
                else:
                    score = None
                reason = getattr(metric, "reason", None)
                success = getattr(metric, "success", None)
                threshold = getattr(metric, "threshold", None)
                evaluation_model = getattr(metric, "evaluation_model", None)
                evaluation_cost = getattr(metric, "evaluation_cost", None)
                verbose_logs = getattr(metric, "verbose_logs", None)
                strict_mode = getattr(metric, "strict_mode", None)
                error_msg = getattr(metric, "error", None)
                attributes: dict[str, Any] = {
                    "deepeval.success": success,
                }
                if threshold is not None:
                    attributes["deepeval.threshold"] = threshold
                if evaluation_model:
                    attributes["deepeval.evaluation_model"] = evaluation_model
                if evaluation_cost is not None:
                    attributes["deepeval.evaluation_cost"] = evaluation_cost
                if verbose_logs:
                    attributes["deepeval.verbose_logs"] = verbose_logs
                if strict_mode is not None:
                    attributes["deepeval.strict_mode"] = strict_mode
                if getattr(test, "name", None):
                    attributes.setdefault(
                        "deepeval.test_case", getattr(test, "name")
                    )
                if getattr(test, "success", None) is not None:
                    attributes.setdefault(
                        "deepeval.test_success", getattr(test, "success")
                    )
                error = None
                if error_msg:
                    error = Error(message=str(error_msg), type=RuntimeError)
                label: str | None = None
                # Metric-specific labeling overrides generic pass/fail
                metric_lower = str(name).lower()
                if metric_lower in {
                    "relevance",
                    "answer_relevancy",
                    "answer_relevance",
                }:
                    if success is True:
                        label = "Relevant"
                    elif success is False:
                        label = "Irrelevant"
                elif (
                    metric_lower.startswith("hallucination")
                    or metric_lower == "faithfulness"
                ):
                    if success is True:
                        label = "Not hallucinated"
                    elif success is False:
                        label = "Hallucinated"
                elif metric_lower == "toxicity":
                    if success is True:
                        label = "Non toxic"
                    elif success is False:
                        label = "Toxic"
                elif metric_lower == "bias":
                    if success is True:
                        label = "Not biased"
                    elif success is False:
                        label = "Biased"
                elif metric_lower.startswith("sentiment"):
                    # Sentiment multi-class; we derive below from compound/score if not provided
                    pass
                else:
                    # Fallback to generic if no mapping
                    if success is True:
                        label = "pass"
                    elif success is False:
                        label = "fail"
                # Custom sentiment transformation: maintain original compound, map recorded score to [0,1]
                if (
                    name in {"sentiment", "sentiment [geval]"}
                    and score is not None
                ):
                    try:
                        compound = max(-1.0, min(1.0, score))
                        mapped = (compound + 1.0) / 2.0  # [0,1]
                        score = mapped
                        attributes.setdefault(
                            "deepeval.sentiment.compound", round(compound, 6)
                        )
                        # Derive sentiment label if not already set by upstream success mapping
                        if label is None:
                            if compound >= 0.25:
                                label = "Positive"
                            elif compound <= -0.25:
                                label = "Negative"
                            else:
                                label = "Neutral"
                    except Exception:  # pragma: no cover - defensive
                        pass
                results.append(
                    EvaluationResult(
                        metric_name=name,
                        score=score,
                        label=label,
                        explanation=reason,
                        error=error,
                        attributes=attributes,
                    )
                )
                # Post-process custom sentiment distribution (after mapping)
                if name in {"sentiment", "sentiment [geval]"} and isinstance(
                    score, (int, float)
                ):
                    # Score expected in [-1,1]; clamp then transform.
                    try:
                        # Retrieve compound (may have been set above)
                        compound_val = attributes.get(
                            "deepeval.sentiment.compound", (score * 2.0) - 1.0
                        )
                        clamped = max(-1.0, min(1.0, float(compound_val)))
                        # Positive strength now is score itself (mapped)
                        pos_strength = float(score)
                        neg_strength = 1 - pos_strength
                        # Heuristic neutral: proximity to midpoint
                        neu_strength = 1 - abs(clamped)
                        # Normalize (neg+neu+pos) to 1 for safety
                        total = neg_strength + neu_strength + pos_strength
                        if total > 0:
                            neg_strength /= total
                            neu_strength /= total
                            pos_strength /= total
                        attributes.update(
                            {
                                "deepeval.sentiment.neg": round(
                                    neg_strength, 6
                                ),
                                "deepeval.sentiment.neu": round(
                                    neu_strength, 6
                                ),
                                "deepeval.sentiment.pos": round(
                                    pos_strength, 6
                                ),
                                "deepeval.sentiment.compound": round(
                                    clamped, 6
                                ),
                            }
                        )
                    except Exception:  # pragma: no cover - defensive
                        pass
        return results

    def _error_results(
        self, message: str, error_type: type[BaseException]
    ) -> Sequence[EvaluationResult]:
        _LOGGER.warning("Deepeval evaluation failed: %s", message)
        return [
            EvaluationResult(
                metric_name=metric,
                explanation=message,
                error=Error(message=message, type=error_type),
                attributes={"deepeval.error": message},
            )
            for metric in self.metrics
        ]

    @staticmethod
    def _coerce_option(value: Any) -> Any:
        # Best-effort recursive coercion; add explicit types to avoid Unknown complaints
        if isinstance(value, MappingABC):
            out: dict[Any, Any] = {}
            for k, v in value.items():  # type: ignore[assignment]
                out[k] = DeepevalEvaluator._coerce_option(v)
            return out
        if isinstance(value, (int, float, bool)):
            return value
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return text
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            return text

    @staticmethod
    def _serialize_messages(messages: Sequence[Any]) -> str:
        chunks: list[str] = []
        for message in messages or []:
            parts = getattr(message, "parts", [])
            for part in parts:
                if isinstance(part, Text):
                    chunks.append(part.content)
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    @staticmethod
    def _extract_context(invocation: LLMInvocation) -> list[str] | None:
        context_values: list[str] = []
        attr = invocation.attributes or {}
        for key in ("context", "additional_context"):
            context_values.extend(
                DeepevalEvaluator._flatten_to_strings(attr.get(key))
            )
        return [value for value in context_values if value] or None

    @staticmethod
    def _extract_retrieval_context(invocation: GenAI) -> list[str] | None:
        attr = invocation.attributes or {}
        retrieval_values: list[str] = []
        for key in (
            "retrieval_context",
            "retrieved_context",
            "retrieved_documents",
            "documents",
            "sources",
            "evidence",
        ):
            retrieval_values.extend(
                DeepevalEvaluator._flatten_to_strings(attr.get(key))
            )
        return [value for value in retrieval_values if value] or None

    @staticmethod
    def _flatten_to_strings(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, MappingABC):
            for key in ("content", "page_content", "text", "body", "value"):
                try:
                    inner = value.get(key)  # type: ignore[index]
                except Exception:  # pragma: no cover
                    inner = None
                if isinstance(inner, str):
                    return [inner]
                if inner is not None:
                    return DeepevalEvaluator._flatten_to_strings(inner)
            try:
                coerced = str(value)  # type: ignore[arg-type]
                return [coerced]
            except Exception:  # pragma: no cover - defensive
                return []
        if isinstance(value, SequenceABC) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            flattened: list[str] = []
            for item in value:  # type: ignore[assignment]
                flattened.extend(DeepevalEvaluator._flatten_to_strings(item))
            return flattened
        return [str(value)]

    def _missing_required_params(
        self, metric_cls: Any, test_case: Any
    ) -> list[str]:
        required = getattr(metric_cls, "_required_params", [])
        missing: list[str] = []
        for param in required:
            attr_name = getattr(param, "value", str(param))
            value = getattr(test_case, attr_name, None)
            if value is None:
                missing.append(attr_name)
                continue
            if isinstance(value, str) and not value.strip():
                missing.append(attr_name)
                continue
            if isinstance(value, SequenceABC) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                flattened = self._flatten_to_strings(value)
                if not flattened:
                    missing.append(attr_name)
        return missing

    @staticmethod
    def _default_model() -> str | None:
        import os

        model = (
            os.getenv("DEEPEVAL_EVALUATION_MODEL")
            or os.getenv("DEEPEVAL_MODEL")
            or os.getenv("OPENAI_MODEL")
        )
        if model:
            return model
        return "gpt-4o-mini"


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


_REGISTRATION = EvaluatorRegistration(
    factory=_factory,
    default_metrics_factory=lambda: _DEFAULT_METRICS,
)


def registration() -> EvaluatorRegistration:
    return _REGISTRATION


def register() -> None:
    register_evaluator(
        "deepeval",
        _REGISTRATION.factory,
        default_metrics=_REGISTRATION.default_metrics_factory,
    )


__all__ = [
    "DeepevalEvaluator",
    "registration",
    "register",
]
