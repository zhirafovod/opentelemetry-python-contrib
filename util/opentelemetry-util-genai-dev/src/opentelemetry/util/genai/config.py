import os
from dataclasses import dataclass

from .environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_ENABLE,
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_SPAN_MODE,
    OTEL_INSTRUMENTATION_GENAI_EVALUATORS,
    OTEL_INSTRUMENTATION_GENAI_GENERATOR,
)
from .types import ContentCapturingMode
from .utils import get_content_capturing_mode


@dataclass(frozen=True)
class Settings:
    """
    Configuration for GenAI telemetry based on environment variables.
    """

    generator_kind: str
    capture_content_span: bool
    capture_content_events: bool
    evaluation_enabled: bool
    evaluation_evaluators: list[str]
    evaluation_span_mode: str


def parse_env() -> Settings:
    """
    Parse relevant environment variables into a Settings object.
    """
    # Generator flavor: span, span_metric, span_metric_event
    gen_choice = (
        os.environ.get(OTEL_INSTRUMENTATION_GENAI_GENERATOR, "span")
        .strip()
        .lower()
    )
    # Content capturing mode (span vs event vs both)
    try:
        mode = get_content_capturing_mode()
    except Exception:
        # If experimental mode not enabled or parsing fails, default to NO_CONTENT
        mode = ContentCapturingMode.NO_CONTENT
    if gen_choice == "span_metric_event":
        capture_content_events = mode in (
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
        capture_content_span = False
    else:
        capture_content_events = False
        capture_content_span = mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
    return Settings(
        generator_kind=gen_choice,
        capture_content_span=capture_content_span,
        capture_content_events=capture_content_events,
        evaluation_enabled=(
            os.environ.get(
                OTEL_INSTRUMENTATION_GENAI_EVALUATION_ENABLE, "false"
            )
            .strip()
            .lower()
            in ("true", "1", "yes")
        ),
        evaluation_evaluators=[
            n.strip()
            for n in os.environ.get(
                OTEL_INSTRUMENTATION_GENAI_EVALUATORS, ""
            ).split(",")
            if n.strip()
        ],
        evaluation_span_mode=(
            lambda v: v if v in ("off", "aggregated", "per_metric") else "off"
        )(
            os.environ.get(
                OTEL_INSTRUMENTATION_GENAI_EVALUATION_SPAN_MODE, "off"
            )
            .strip()
            .lower()
        ),
    )
