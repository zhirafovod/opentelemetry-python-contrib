from __future__ import annotations

import os
from typing import List, Optional

from opentelemetry._logs import Logger, get_logger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

from ..generators.utils import (
    _collect_finish_reasons,
    _emit_chat_generation_logs,
    _message_to_log_record,
    _set_response_and_usage_attributes,
)
from ..types import Error, LLMInvocation

_ENV_VAR = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"


class ContentEventsEmitter:
    """Emits input/output content as events (log records) instead of span attributes.

    Role: content_event
    Behavior mirrors existing SpanMetricEventGenerator for content emission, but
    isolated so composite can combine with metrics + span emitters.
    """

    role = "content_event"
    name = "semconv_content_events"

    def __init__(
        self,
        logger: Optional[Logger] = None,
        capture_content: bool = False,
    ):
        self._logger: Logger = logger or get_logger(__name__)
        self._capture_content = capture_content

    # Lifecycle -------------------------------------------------------------
    def start(self, invocation: LLMInvocation) -> None:
        if not os.getenv(_ENV_VAR):  # gating same as legacy generator
            return
        if not invocation.input_messages:
            return
        for msg in invocation.input_messages:
            record = _message_to_log_record(
                msg,
                provider_name=invocation.provider,
                framework=invocation.attributes.get("framework"),
                capture_content=self._capture_content,
            )
            if record:
                try:  # pragma: no cover
                    self._logger.emit(record)
                except Exception:
                    pass

    def finish(self, invocation: LLMInvocation) -> None:
        if invocation.span is None:
            return
        span = invocation.span
        # Always set finish reasons / usage attrs here (event flavor keeps span lean)
        finish_reasons: List[str] = []
        if invocation.output_messages:
            finish_reasons = _collect_finish_reasons(
                invocation.output_messages
            )
            if finish_reasons:
                span.set_attribute(
                    GenAI.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons
                )
        _set_response_and_usage_attributes(
            span,
            invocation.response_model_name,
            invocation.response_id,
            invocation.input_tokens,
            invocation.output_tokens,
        )
        if not os.getenv(_ENV_VAR):
            return
        if not invocation.output_messages:
            return
        try:
            _emit_chat_generation_logs(
                self._logger,
                invocation.output_messages,
                provider_name=invocation.provider,
                framework=invocation.attributes.get("framework"),
                capture_content=self._capture_content,
            )
        except Exception:  # pragma: no cover
            pass

    def error(
        self, error: Error, invocation: LLMInvocation
    ) -> None:  # no event emission on error (simpler; can add input later)
        return None
