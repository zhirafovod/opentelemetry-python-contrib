"""Emitters responsible for emitting telemetry derived from evaluation results."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Sequence, Union

from opentelemetry import _events as _otel_events

from ..attributes import (
    GEN_AI_EVALUATION_ATTRIBUTES_PREFIX,
    GEN_AI_EVALUATION_EXPLANATION,
    GEN_AI_EVALUATION_NAME,
    GEN_AI_EVALUATION_SCORE_LABEL,
    GEN_AI_EVALUATION_SCORE_VALUE,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_ID,
)
from ..interfaces import EmitterMeta
from ..types import EvaluationResult, GenAI


def _get_request_model(invocation: GenAI) -> Union[str, None]:
    return getattr(invocation, "request_model", None) or getattr(
        invocation, "model", None
    )


def _get_response_id(invocation: GenAI) -> Union[str, None]:  # best-effort
    return getattr(invocation, "response_id", None)


class _EvaluationEmitterBase(EmitterMeta):
    role = "evaluation"

    def on_start(self, obj: Any) -> None:  # pragma: no cover - default no-op
        return None

    def on_end(self, obj: Any) -> None:  # pragma: no cover - default no-op
        return None

    def on_error(
        self, error, obj: Any
    ) -> None:  # pragma: no cover - default no-op
        return None


def _canonicalize_metric_name(raw_name: str) -> Optional[str]:
    """Map raw evaluator metric names (possibly noisy) to canonical names.

    Handles legacy / provider-specific variants and formatting differences:
    - answer relevancy / answer_relevancy / answer relevance -> relevance
    - faithfulness -> hallucination (legacy synonym)
    - hallucination [geval] / hallucination_geval / hallucination-* -> hallucination
    - direct passthrough for: hallucination, sentiment, toxicity, bias
    Returns None if the metric is unsupported (ignored by emitters).
    """
    if not raw_name:
        return None
    lowered = raw_name.strip().lower()
    # Fast path exact matches first
    if lowered in {"bias", "toxicity", "sentiment", "hallucination"}:
        return lowered
    if lowered == "faithfulness":
        return "hallucination"
    # Normalize punctuation/whitespace to underscores for pattern matching
    import re as _re  # local import to avoid global cost

    normalized = _re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if normalized in {"answer_relevancy", "answer_relevance", "relevance"}:
        return "relevance"
    if normalized.startswith("hallucination"):
        return "hallucination"
    if normalized.startswith("sentiment"):
        # Allow variants like sentiment_geval, sentiment[geval], sentiment-geval
        return "sentiment"
    return None


# Debug logging configuration:
#   OTEL_GENAI_EVAL_DEBUG_SKIPS=1|true|yes  -> one-time logs when a measurement is skipped (already implemented)
#   OTEL_GENAI_EVAL_DEBUG_EACH=1|true|yes   -> verbose log line for every evaluation result processed (attempted measurement)
_EVAL_DEBUG_SKIPS = os.getenv("OTEL_GENAI_EVAL_DEBUG_SKIPS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_EVAL_DEBUG_EACH = os.getenv("OTEL_GENAI_EVAL_DEBUG_EACH", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class EvaluationMetricsEmitter(_EvaluationEmitterBase):
    """Records evaluation scores to metric-specific histograms.

    Instead of a single shared histogram (gen_ai.evaluation.score), we emit to
    gen_ai.evaluation.score.<metric_name>. This improves downstream aggregation
    clarity at the cost of additional instruments. A callable factory provided
    by the handler supplies (and caches) histogram instances.
    """

    role = "evaluation_metrics"
    name = "EvaluationMetrics"

    def __init__(
        self, histogram_factory
    ) -> None:  # callable(metric_name)->Histogram|None OR direct histogram
        # Backward-compatible: tests may pass a histogram instance directly.
        if hasattr(histogram_factory, "record") and not callable(  # type: ignore[arg-type]
            getattr(histogram_factory, "__call__", None)
        ):
            direct_hist = histogram_factory

            def _direct_factory(_name: str):  # ignore metric name, single hist
                return direct_hist

            self._hist_factory = _direct_factory
        else:
            self._hist_factory = histogram_factory

    def on_evaluation_results(  # type: ignore[override]
        self,
        results: Sequence[EvaluationResult],
        obj: Union[Any, None] = None,
    ) -> None:
        invocation = obj if isinstance(obj, GenAI) else None
        if invocation is None:
            if _EVAL_DEBUG_SKIPS:
                logging.getLogger(__name__).debug(
                    "EvaluationMetricsEmitter: skipping all results (no GenAI invocation provided)"
                )
            return
        # Per-emitter set of (reason, key) we have already logged to avoid noise.
        if not hasattr(self, "_logged_skip_keys"):
            self._logged_skip_keys = set()  # type: ignore[attr-defined]

        def _log_skip(
            reason: str,
            metric_raw: Any,
            extra: Optional[Dict[str, Any]] = None,
        ):
            if not _EVAL_DEBUG_SKIPS:
                return
            key = (reason, str(metric_raw))
            try:
                if key in self._logged_skip_keys:  # type: ignore[attr-defined]
                    return
                self._logged_skip_keys.add(key)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                pass
            msg = f"EvaluationMetricsEmitter: skipped metric '{metric_raw}' reason={reason}"
            if extra:
                try:
                    msg += " " + " ".join(
                        f"{k}={v!r}" for k, v in extra.items() if v is not None
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
            logging.getLogger(__name__).debug(msg)

        for res in results:
            canonical = _canonicalize_metric_name(
                getattr(res, "metric_name", "") or ""
            )
            raw_name = getattr(res, "metric_name", None)
            if _EVAL_DEBUG_EACH:
                logging.getLogger(__name__).debug(
                    "EvaluationMetricsEmitter: processing metric raw=%r canonical=%r score=%r type=%s label=%r",
                    raw_name,
                    canonical,
                    getattr(res, "score", None),
                    type(getattr(res, "score", None)).__name__,
                    getattr(res, "label", None),
                )
            if canonical is None:
                _log_skip("unsupported_metric_name", raw_name)
                continue
            if not isinstance(res.score, (int, float)):
                _log_skip(
                    "non_numeric_score",
                    raw_name,
                    {
                        "score_type": type(res.score).__name__,
                        "score_value": getattr(res, "score", None),
                    },
                )
                continue
            try:
                histogram = (
                    self._hist_factory(canonical)
                    if self._hist_factory
                    else None
                )  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive
                histogram = None
                _log_skip(
                    "histogram_factory_error", raw_name, {"error": repr(exc)}
                )
            if histogram is None:
                # Log once per metric name if histogram factory did not provide an instrument.
                try:
                    _once_key = f"_genai_eval_hist_missing_{canonical}"
                    if not getattr(self, _once_key, False):
                        logging.getLogger(__name__).debug(
                            "EvaluationMetricsEmitter: no histogram for canonical metric '%s' (factory returned None)",
                            canonical,
                        )
                        setattr(self, _once_key, True)
                except Exception:
                    pass
                _log_skip(
                    "no_histogram_instrument",
                    raw_name,
                    {"canonical": canonical},
                )
                continue
            elif _EVAL_DEBUG_EACH:
                logging.getLogger(__name__).debug(
                    "EvaluationMetricsEmitter: recording metric canonical=%r score=%r instrument=%s",
                    canonical,
                    getattr(res, "score", None),
                    type(histogram).__name__,
                )
            attrs: Dict[str, Any] = {
                GEN_AI_OPERATION_NAME: "evaluation",
                GEN_AI_EVALUATION_NAME: canonical,
            }
            # If the source invocation carried agent identity, propagate
            agent_name = getattr(invocation, "agent_name", None)
            agent_id = getattr(invocation, "agent_id", None)
            # Fallbacks: if instrumentation didn't populate agent_name/id fields explicitly but
            # the invocation is an AgentInvocation, derive them from core fields to preserve identity.
            try:
                from opentelemetry.util.genai.types import (
                    AgentInvocation as _AI,  # local import to avoid cycle
                )

                if agent_name is None and isinstance(invocation, _AI):  # type: ignore[attr-defined]
                    agent_name = getattr(invocation, "name", None)
                if agent_id is None and isinstance(invocation, _AI):  # type: ignore[attr-defined]
                    agent_id = str(getattr(invocation, "run_id", "")) or None
            except Exception:  # pragma: no cover - defensive
                pass
            workflow_id = getattr(invocation, "workflow_id", None)
            if agent_name:
                attrs["gen_ai.agent.name"] = agent_name
            if agent_id:
                attrs["gen_ai.agent.id"] = agent_id
            if workflow_id:
                attrs["gen_ai.workflow.id"] = workflow_id
            req_model = _get_request_model(invocation)
            if req_model:
                attrs[GEN_AI_REQUEST_MODEL] = req_model
            provider = getattr(invocation, "provider", None)
            if provider:
                attrs[GEN_AI_PROVIDER_NAME] = provider
            if res.label is not None:
                attrs[GEN_AI_EVALUATION_SCORE_LABEL] = res.label
            # Derive boolean gen_ai.evaluation.passed
            passed = None
            if res.label:
                lbl_raw = str(res.label)
                lbl = lbl_raw.lower()
                # Positive (passed) label vocabulary
                passed_positive = {
                    "pass",
                    "success",
                    "ok",
                    "true",
                    "relevant",
                    "not hallucinated",
                    "non toxic",
                    "not biased",
                    "positive",  # sentiment
                    "neutral",  # treat neutral as acceptable pass
                }
                failed_negative = {
                    "fail",
                    "error",
                    "false",
                    "irrelevant",
                    "hallucinated",
                    "toxic",
                    "biased",
                    "negative",  # negative sentiment considered not passed
                }
                if lbl in passed_positive:
                    passed = True
                elif lbl in failed_negative:
                    passed = False
            # NOTE: We deliberately do NOT infer pass/fail purely from numeric score
            # without an accompanying categorical label to avoid accidental cardinality
            # or semantic ambiguities across evaluators. Future extension could allow
            # opt-in heuristic score->pass mapping.
            if passed is not None:
                attrs["gen_ai.evaluation.passed"] = passed
            attrs["gen_ai.evaluation.score.units"] = "score"
            if res.error is not None:
                attrs["error.type"] = res.error.type.__qualname__
            try:
                histogram.record(res.score, attributes=attrs)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive
                _log_skip(
                    "histogram_record_error", raw_name, {"error": repr(exc)}
                )
                if _EVAL_DEBUG_EACH:
                    logging.getLogger(__name__).debug(
                        "EvaluationMetricsEmitter: record failed canonical=%r score=%r error=%r",
                        canonical,
                        getattr(res, "score", None),
                        exc,
                    )
                pass


class EvaluationEventsEmitter(_EvaluationEmitterBase):
    """Emits one event per evaluation result."""

    role = "evaluation_events"
    name = "EvaluationEvents"

    def __init__(
        self, event_logger, *, emit_legacy_event: bool = False
    ) -> None:
        self._event_logger = event_logger
        self._emit_legacy_event = emit_legacy_event
        self._primary_event_name = "gen_ai.evaluation.result"
        self._legacy_event_name = "gen_ai.evaluation"

    def on_evaluation_results(  # type: ignore[override]
        self,
        results: Sequence[EvaluationResult],
        obj: Union[Any, None] = None,
    ) -> None:
        if self._event_logger is None:
            return
        invocation = obj if isinstance(obj, GenAI) else None
        if invocation is None or not results:
            return

        req_model = _get_request_model(invocation)
        provider = getattr(invocation, "provider", None)
        response_id = _get_response_id(invocation)

        span_context = None
        if getattr(invocation, "span", None) is not None:
            try:
                span_context = invocation.span.get_span_context()
            except Exception:  # pragma: no cover - defensive
                span_context = None
        span_id = (
            getattr(span_context, "span_id", None)
            if span_context is not None
            else None
        )
        trace_id = (
            getattr(span_context, "trace_id", None)
            if span_context is not None
            else None
        )

        for res in results:
            canonical = _canonicalize_metric_name(
                getattr(res, "metric_name", "") or ""
            )
            if canonical is None:
                continue
            base_attrs: Dict[str, Any] = {
                GEN_AI_OPERATION_NAME: "evaluation",
                GEN_AI_EVALUATION_NAME: canonical,
            }
            agent_name = getattr(invocation, "agent_name", None)
            agent_id = getattr(invocation, "agent_id", None)
            try:
                from opentelemetry.util.genai.types import (
                    AgentInvocation as _AI,  # local import to avoid cycle
                )

                if agent_name is None and isinstance(invocation, _AI):  # type: ignore[attr-defined]
                    agent_name = getattr(invocation, "name", None)
                if agent_id is None and isinstance(invocation, _AI):  # type: ignore[attr-defined]
                    agent_id = str(getattr(invocation, "run_id", "")) or None
            except Exception:  # pragma: no cover - defensive
                pass
            workflow_id = getattr(invocation, "workflow_id", None)
            if agent_name:
                base_attrs["gen_ai.agent.name"] = agent_name
            if agent_id:
                base_attrs["gen_ai.agent.id"] = agent_id
            if workflow_id:
                base_attrs["gen_ai.workflow.id"] = workflow_id
            if req_model:
                base_attrs[GEN_AI_REQUEST_MODEL] = req_model
            if provider:
                base_attrs[GEN_AI_PROVIDER_NAME] = provider
            if response_id:
                base_attrs[GEN_AI_RESPONSE_ID] = response_id
            if isinstance(res.score, (int, float)):
                base_attrs[GEN_AI_EVALUATION_SCORE_VALUE] = res.score
            if res.label is not None:
                base_attrs[GEN_AI_EVALUATION_SCORE_LABEL] = res.label
            passed = None
            if res.label:
                lbl_raw = str(res.label)
                lbl = lbl_raw.lower()
                passed_positive = {
                    "pass",
                    "success",
                    "ok",
                    "true",
                    "relevant",
                    "not hallucinated",
                    "non toxic",
                    "not biased",
                    "positive",
                    "neutral",
                }
                failed_negative = {
                    "fail",
                    "error",
                    "false",
                    "irrelevant",
                    "hallucinated",
                    "toxic",
                    "biased",
                    "negative",
                }
                if lbl in passed_positive:
                    passed = True
                elif lbl in failed_negative:
                    passed = False
            # Do not infer pass/fail solely from numeric score (see metrics emitter note)
            if passed is not None:
                base_attrs["gen_ai.evaluation.passed"] = passed
            if isinstance(res.score, (int, float)):
                base_attrs["gen_ai.evaluation.score.units"] = "score"
            if res.error is not None:
                base_attrs["error.type"] = res.error.type.__qualname__

            spec_attrs = dict(base_attrs)
            if res.explanation:
                spec_attrs[GEN_AI_EVALUATION_EXPLANATION] = res.explanation
            if res.attributes:
                for key, value in dict(res.attributes).items():
                    key_str = str(key)
                    spec_attrs[
                        f"{GEN_AI_EVALUATION_ATTRIBUTES_PREFIX}{key_str}"
                    ] = value
            if res.error is not None and getattr(res.error, "message", None):
                spec_attrs[
                    f"{GEN_AI_EVALUATION_ATTRIBUTES_PREFIX}error.message"
                ] = res.error.message

            try:
                self._event_logger.emit(
                    _otel_events.Event(
                        name=self._primary_event_name,
                        attributes=spec_attrs,
                        span_id=span_id,
                        trace_id=trace_id,
                    )
                )
            except Exception:  # pragma: no cover - defensive
                pass

            if not self._emit_legacy_event:
                continue

            legacy_attrs = dict(base_attrs)
            legacy_body: Dict[str, Any] = {}
            if res.explanation:
                legacy_body["gen_ai.evaluation.explanation"] = res.explanation
            if res.attributes:
                legacy_body["gen_ai.evaluation.attributes"] = dict(
                    res.attributes
                )
            if res.error is not None and getattr(res.error, "message", None):
                legacy_attrs["error.message"] = res.error.message

            try:
                self._event_logger.emit(
                    _otel_events.Event(
                        name=self._legacy_event_name,
                        attributes=legacy_attrs,
                        body=legacy_body or None,
                        span_id=span_id,
                        trace_id=trace_id,
                    )
                )
            except Exception:  # pragma: no cover - defensive
                pass


__all__ = [
    "EvaluationMetricsEmitter",
    "EvaluationEventsEmitter",
]
