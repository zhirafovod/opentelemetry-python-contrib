"""Emitter package consolidating all telemetry signal emitters.

Exports:
    SpanEmitter
    MetricsEmitter
    ContentEventsEmitter
    CompositeGenerator (composition orchestrator; legacy name retained)

NOTE: CompositeGenerator name retained for backward compatibility with
previous documentation. Future rename to CompositeEmitter may introduce
an alias first.
"""
from __future__ import annotations

from .span import SpanEmitter  # noqa: F401
from .metrics import MetricsEmitter  # noqa: F401
from .content_events import ContentEventsEmitter  # noqa: F401
from .composite import CompositeGenerator  # noqa: F401

__all__ = [
    "SpanEmitter",
    "MetricsEmitter",
    "ContentEventsEmitter",
    "CompositeGenerator",
]

