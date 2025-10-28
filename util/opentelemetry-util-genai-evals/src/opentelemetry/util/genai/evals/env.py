"""Environment helpers for evaluation configuration."""

from __future__ import annotations

import os
from typing import Mapping

from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS,
    OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL,
    OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION,
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE,
)

_TRUTHY = {"1", "true", "yes", "on"}


def _get_env(name: str, source: Mapping[str, str] | None = None) -> str | None:
    env = source if source is not None else os.environ
    return env.get(name)


def read_raw_evaluators(
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Return raw evaluator configuration text."""

    return _get_env(OTEL_INSTRUMENTATION_GENAI_EVALS_EVALUATORS, env)


def read_interval(
    env: Mapping[str, str] | None = None,
    *,
    default: float | None = 5.0,
) -> float | None:
    raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALS_INTERVAL, env)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def read_aggregation_flag(
    env: Mapping[str, str] | None = None,
) -> bool | None:
    raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALS_RESULTS_AGGREGATION, env)
    if raw is None:
        return None
    return raw.strip().lower() in _TRUTHY


def read_sample_rate(
    env: Mapping[str, str] | None = None,
    *,
    default: float = 1.0,
) -> float:
    raw = _get_env(OTEL_INSTRUMENTATION_GENAI_EVALUATION_SAMPLE_RATE, env)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


__all__ = [
    "read_raw_evaluators",
    "read_interval",
    "read_aggregation_flag",
    "read_sample_rate",
]
