# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TraceloopSpanProcessor - A span processor that automatically transforms spans
using Traceloop transformation rules.
"""

from typing import Any, Callable, Dict, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Span

from .span_transformer import transform_existing_span_to_telemetry
from .traceloop_span_generator import TraceloopSpanGenerator


class TraceloopSpanProcessor(SpanProcessor):
    """
    A span processor that automatically applies Traceloop transformations to spans.

    This processor can be added to your TracerProvider to automatically transform
    all spans according to your transformation rules.
    """

    def __init__(
        self,
        attribute_transformations: Optional[Dict[str, Any]] = None,
        name_transformations: Optional[Dict[str, str]] = None,
        traceloop_attributes: Optional[Dict[str, Any]] = None,
        span_filter: Optional[Callable[[ReadableSpan], bool]] = None,
        generator: Optional[TraceloopSpanGenerator] = None,
    ):
        """
        Initialize the Traceloop span processor.

        Args:
            attribute_transformations: Rules for transforming span attributes
            name_transformations: Rules for transforming span names
            traceloop_attributes: Additional Traceloop-specific attributes to add
            span_filter: Optional filter function to determine which spans to transform
            generator: Optional custom TraceloopSpanGenerator
        """
        self.attribute_transformations = attribute_transformations or {}
        self.name_transformations = name_transformations or {}
        self.traceloop_attributes = traceloop_attributes or {}
        self.span_filter = span_filter or self._default_span_filter
        self.generator = generator or TraceloopSpanGenerator(
            capture_content=True
        )

    def _default_span_filter(self, span: ReadableSpan) -> bool:
        """Default filter: Transform spans that look like LLM/AI calls."""
        if not span.name or not span.attributes:
            return False

        # Check for common LLM/AI span indicators
        llm_indicators = [
            "chat",
            "completion",
            "llm",
            "ai",
            "gpt",
            "claude",
            "gemini",
            "openai",
            "anthropic",
            "cohere",
            "huggingface",
        ]

        span_name_lower = span.name.lower()
        for indicator in llm_indicators:
            if indicator in span_name_lower:
                return True

        # Check attributes for AI/LLM markers
        for attr_key in span.attributes.keys():
            attr_key_lower = str(attr_key).lower()
            if any(
                marker in attr_key_lower
                for marker in ["llm", "ai", "gen_ai", "model"]
            ):
                return True

        return False

    def on_start(
        self, span: Span, parent_context: Optional[Context] = None
    ) -> None:
        """Called when a span is started."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span is ended.
        """
        try:
            # Check if this span should be transformed
            if not self.span_filter(span):
                return

            # Apply transformations and generate new telemetry
            transform_existing_span_to_telemetry(
                existing_span=span,
                attribute_transformations=self.attribute_transformations,
                name_transformations=self.name_transformations,
                traceloop_attributes=self.traceloop_attributes,
                generator=self.generator,
            )

        except Exception as e:
            # Don't let transformation errors break the original span processing
            import logging

            logging.warning(
                f"TraceloopSpanProcessor failed to transform span: {e}"
            )

    def shutdown(self) -> None:
        """Called when the tracer provider is shutdown."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        return True
