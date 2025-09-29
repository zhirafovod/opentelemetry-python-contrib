from __future__ import annotations

import importlib
import time
from typing import List, Optional

from opentelemetry import _events as _otel_events
from opentelemetry.trace import Tracer

from ..config import Settings
from ..types import Error, EvaluationResult, LLMInvocation
from .base import Evaluator
from .evaluation_emitters import (
    CompositeEvaluationEmitter,
    EvaluationEventsEmitter,
    EvaluationMetricsEmitter,
    EvaluationSpansEmitter,
)
from .registry import get_evaluator, register_evaluator

# NOTE: Type checker warns about heterogeneous list (metrics + events + spans) passed
# to CompositeEvaluationEmitter due to generic inference; safe at runtime.


class EvaluationManager:
    """Coordinates evaluator discovery, execution, and telemetry emission."""

    def __init__(
        self,
        settings: Settings,
        tracer: Tracer,
        event_logger: _otel_events.EventLogger,  # type: ignore[attr-defined]
        histogram,  # opentelemetry.metrics.Histogram
    ) -> None:
        self._settings = settings
        self._tracer = tracer
        self._event_logger = event_logger
        self._histogram = histogram
        emitters = [
            EvaluationMetricsEmitter(histogram),
            EvaluationEventsEmitter(event_logger),
        ]
        if settings.evaluation_span_mode in ("aggregated", "per_metric"):
            emitters.append(
                EvaluationSpansEmitter(
                    tracer=tracer, span_mode=settings.evaluation_span_mode
                )
            )
        self._emitter = CompositeEvaluationEmitter(emitters)  # type: ignore[arg-type]

    def evaluate(
        self, invocation: LLMInvocation, evaluators: Optional[List[str]] = None
    ) -> List[EvaluationResult]:
        if not self._settings.evaluation_enabled:
            return []
        if evaluators is None:
            evaluators = list(self._settings.evaluation_evaluators)
        if not evaluators:
            return []
        # Removed automatic builtin import to allow dynamic 'deepeval' override before placeholder registration.
        if invocation.end_time is None:
            invocation.end_time = time.time()
        results: List[EvaluationResult] = []
        for name in evaluators:
            results.extend(self._run_single(name, invocation))
        if results:
            self._emitter.emit(results, invocation)
        return results

    def _run_single(
        self, name: str, invocation: LLMInvocation
    ) -> List[EvaluationResult]:
        evaluator: Optional[Evaluator] = None
        lower = name.lower()
        # 1. Try dynamic external (deepeval) first (overrides any placeholder)
        if lower == "deepeval":
            try:
                import sys
                import types

                if "opentelemetry.util.genai.evals" not in sys.modules:
                    pkg = types.ModuleType("opentelemetry.util.genai.evals")
                    pkg.__path__ = []  # type: ignore[attr-defined]
                    sys.modules["opentelemetry.util.genai.evals"] = pkg
                ext_mod = importlib.import_module(
                    "opentelemetry.util.genai.evals.deepeval"
                )
                if hasattr(ext_mod, "DeepEvalEvaluator"):
                    register_evaluator(
                        "deepeval",
                        lambda: ext_mod.DeepEvalEvaluator(
                            self._event_logger, self._tracer
                        ),
                    )
            except Exception:  # pragma: no cover
                pass
        # 2. Try to obtain evaluator
        try:
            evaluator = get_evaluator(name)
        except Exception:
            evaluator = None
        # 3. If still missing, load builtins (length, sentiment, deepeval placeholder)
        if evaluator is None and lower in {"length", "sentiment", "deepeval"}:
            try:
                import importlib as _imp
                import sys

                mod_name = "opentelemetry.util.genai.evaluators.builtins"
                if mod_name in sys.modules:
                    _imp.reload(sys.modules[mod_name])
                else:
                    _imp.import_module(mod_name)
                evaluator = get_evaluator(name)
            except Exception:  # pragma: no cover
                evaluator = None
        # 4. If deepeval placeholder loaded but external now available, attempt one more override
        if (
            lower == "deepeval"
            and evaluator is not None
            and evaluator.__class__.__name__ == "DeepevalEvaluator"
        ):
            try:
                ext_mod = importlib.import_module(
                    "opentelemetry.util.genai.evals.deepeval"
                )
                if hasattr(ext_mod, "DeepEvalEvaluator"):
                    register_evaluator(
                        "deepeval",
                        lambda: ext_mod.DeepEvalEvaluator(
                            self._event_logger, self._tracer
                        ),
                    )
                    evaluator = get_evaluator(name)
            except Exception:  # pragma: no cover
                pass
        if evaluator is None:
            return [
                EvaluationResult(
                    metric_name=name,
                    error=Error(
                        message=f"Unknown evaluator: {name}", type=LookupError
                    ),
                )
            ]
        # 5. Execute evaluator safely
        try:
            eval_out = evaluator.evaluate(invocation)
            if isinstance(eval_out, EvaluationResult):
                return [eval_out]
            if isinstance(eval_out, list):
                out: List[EvaluationResult] = []
                for item in eval_out:
                    if isinstance(item, EvaluationResult):
                        out.append(item)
                    else:
                        out.append(
                            EvaluationResult(
                                metric_name=name,
                                error=Error(
                                    message="Evaluator returned non-EvaluationResult item",
                                    type=TypeError,
                                ),
                            )
                        )
                return out
            return [
                EvaluationResult(
                    metric_name=name,
                    error=Error(
                        message="Evaluator returned unsupported type",
                        type=TypeError,
                    ),
                )
            ]
        except Exception as exc:  # pragma: no cover - runtime safety
            return [
                EvaluationResult(
                    metric_name=name,
                    error=Error(message=str(exc), type=type(exc)),
                )
            ]

    def _try_dynamic_register(
        self, name: str
    ) -> Optional[Evaluator]:  # legacy path no longer used
        return None


__all__ = ["EvaluationManager"]
