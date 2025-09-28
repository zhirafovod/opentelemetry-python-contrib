import os
from dataclasses import dataclass

from .environment_variables import (
    # OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_EMITTERS,
    OTEL_INSTRUMENTATION_GENAI_EVALUATION_ENABLE,
    OTEL_INSTRUMENTATION_GENAI_EVALUATORS,
)
from .types import ContentCapturingMode
from .utils import get_content_capturing_mode


@dataclass(frozen=True)
class Settings:
    """
    Configuration for GenAI telemetry based on environment variables.
    """

    generator_kind: str
    evaluation_enabled: bool
    evaluation_evaluators: list[str]
    capture_content_span: bool
    capture_content_events: bool


def parse_env() -> Settings:
    """
    Parse relevant environment variables into a Settings object.
    """
    # Generator flavor: span, span_metric, span_metric_event
    gen_choice = (
        os.environ.get(OTEL_INSTRUMENTATION_GENAI_EMITTERS, "span")
        .strip()
        .lower()
    )

    # capture_content = os.environ.get(
    #     os.environ.get(
    #         OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
    #     )
    #     .strip()
    #     .lower()
    # )

    # # Content capturing mode (span vs event vs both)
    try:
        # capture_content = os.environ.get(
        #     OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT
        # )
        mode = get_content_capturing_mode()
    except Exception:
        # If experimental mode not enabled or parsing fails, default to NO_CONTENT
        mode = ContentCapturingMode.NO_CONTENT
    # capture_content_events = False
    # capture_content_span = False
    # if capture_content == "true":
    #     if gen_choice == "span_metric_event":
    #         capture_content_events = True
    #     else:
    #         capture_content_span = True

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
        # evaluation_span_mode=(
        #     lambda v: v if v in ("off", "aggregated", "per_metric") else "off"
        # )(
        #     os.environ.get(
        #         OTEL_INSTRUMENTATION_GENAI_EVALUATION_SPAN_MODE, "off"
        #     )
        #     .strip()
        #     .lower()
        # ),
    )
