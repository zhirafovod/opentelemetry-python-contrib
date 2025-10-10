from __future__ import annotations

from opentelemetry.util.genai.types import ContentCapturingMode
from opentelemetry.util.genai.utils import get_content_capturing_mode


def test_event_only_mode(monkeypatch):  # type: ignore[no-untyped-def]
    """Setting capture messages to 'events' yields EVENT_ONLY."""
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES", "events")
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.EVENT_ONLY


def test_span_only_mode(monkeypatch):  # type: ignore[no-untyped-def]
    """Setting capture messages to 'span' yields SPAN_ONLY."""
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES", "span")
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.SPAN_ONLY


def test_both_mode(monkeypatch):  # type: ignore[no-untyped-def]
    """Setting capture messages to 'both' yields SPAN_AND_EVENT."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES", "both")
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.SPAN_AND_EVENT


def test_none_mode(monkeypatch):  # type: ignore[no-untyped-def]
    """Setting capture messages to 'none' yields NO_CONTENT."""
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES", "none")
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.NO_CONTENT


def test_invalid_mode_defaults(monkeypatch):  # type: ignore[no-untyped-def]
    """Invalid value falls back to NO_CONTENT."""
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGES", "garbage-value"
    )
    mode = get_content_capturing_mode()
    assert mode == ContentCapturingMode.NO_CONTENT
