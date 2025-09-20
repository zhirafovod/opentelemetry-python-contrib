"""Deprecated shim.

This file will be removed in a future release. Import from the package instead:
    from opentelemetry.util.genai.generators import SpanGenerator
"""

from warnings import warn

warn(
    "opentelemetry.util.genai.generators (module) is deprecated; use the package 'opentelemetry.util.genai.generators' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .generators import (  # type: ignore  # noqa: E402
    BaseTelemetryGenerator,
    SpanGenerator,
    SpanMetricEventGenerator,
    SpanMetricGenerator,
)

__all__ = [
    "BaseTelemetryGenerator",
    "SpanGenerator",
    "SpanMetricGenerator",
    "SpanMetricEventGenerator",
]
