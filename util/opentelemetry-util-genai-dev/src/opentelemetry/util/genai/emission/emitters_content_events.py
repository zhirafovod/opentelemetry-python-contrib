from __future__ import annotations

from typing import Any, Optional

from opentelemetry._logs import Logger, get_logger

from ..generators.utils import (
    _chat_generation_to_log_record,
    _message_to_log_record,
)
from ..types import Error, LLMInvocation


class ContentEventsEmitter:
    """Emits input/output content as events (log records) instead of span attributes.

    Ignores objects that are not LLMInvocation (embeddings produce no content events yet).
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
    def start(self, obj: Any) -> None:
        if not isinstance(obj, LLMInvocation) or not self._capture_content:
            return
        invocation = obj
        if not invocation.input_messages:
            return
        for msg in invocation.input_messages:
            try:
                record = _message_to_log_record(
                    msg,
                    provider_name=invocation.provider,
                    framework=invocation.attributes.get("framework"),
                    capture_content=self._capture_content,
                )
                if record and self._logger:
                    self._logger.emit(record)
            except Exception:
                pass

    def finish(self, obj: Any) -> None:
        if not isinstance(obj, LLMInvocation) or not self._capture_content:
            return
        invocation = obj
        if invocation.span is None or not invocation.output_messages:
            return
        # Use chat-generation log records for output messages
        for index, msg in enumerate(invocation.output_messages):
            try:
                record = _chat_generation_to_log_record(
                    msg,
                    index,
                    invocation.provider,
                    invocation.attributes.get("framework"),
                    self._capture_content,
                )
                if record:
                    try:
                        self._logger.emit(record)
                    except Exception:
                        pass
            except Exception:
                pass

    def error(self, error: Error, obj: Any) -> None:
        # No content events on error
        return None

    def handles(self, obj: Any) -> bool:
        return isinstance(obj, LLMInvocation)
