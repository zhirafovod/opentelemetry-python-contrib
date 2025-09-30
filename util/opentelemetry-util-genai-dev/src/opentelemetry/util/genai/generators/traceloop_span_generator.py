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

import re
import time
from typing import Any, Dict, Optional, Union

from opentelemetry import trace
from opentelemetry.trace import Tracer
from opentelemetry.trace.status import Status, StatusCode

from ..types import Error, LLMInvocation, TraceloopInvocation
from .base_span_generator import BaseSpanGenerator


class TraceloopSpanGenerator(BaseSpanGenerator):
    """
    Generator for Traceloop-compatible spans using util-genai infrastructure.

    Instead of modifying existing spans, this creates new telemetry from
    TraceloopInvocation data types that contain all proprietary attributes
    and transformation rules.
    """

    def __init__(
        self,
        tracer: Optional[Tracer] = None,
        capture_content: bool = False,
        default_attribute_rules: Optional[Dict[str, Any]] = None,
        default_name_mappings: Optional[Dict[str, str]] = None,
    ):
        super().__init__(tracer, capture_content)
        self.default_attribute_rules = default_attribute_rules or {}
        self.default_name_mappings = default_name_mappings or {}

    def _apply_name_transformations(
        self, invocation: TraceloopInvocation, span_name: str
    ) -> str:
        """Apply name transformations based on the invocation's rules."""
        # Use invocation-specific rules, fall back to defaults
        name_mappings = {
            **self.default_name_mappings,
            **invocation.name_transformations,
        }

        # Apply direct mappings first
        if span_name in name_mappings:
            return name_mappings[span_name]

        # Apply pattern-based transformations
        for pattern, replacement in name_mappings.items():
            if "*" in pattern:
                regex_pattern = pattern.replace("*", ".*")
                if re.match(regex_pattern, span_name):
                    return replacement

        return span_name

    def _apply_attribute_transformations(
        self, invocation: TraceloopInvocation
    ):
        """Apply attribute transformations to the invocation's attributes."""
        if invocation.span is None:
            return

        # Use invocation-specific rules, merged with defaults
        attribute_rules = {
            **self.default_attribute_rules,
            **invocation.attribute_transformations,
        }

        # Start with the base attributes from invocation
        attributes = dict(invocation.attributes)

        # Apply transformation rules
        for rule_key, rule_value in attribute_rules.items():
            if rule_key == "remove":
                # Remove specified attributes
                for attr_to_remove in rule_value:
                    attributes.pop(attr_to_remove, None)
            elif rule_key == "rename":
                # Rename attributes
                for old_name, new_name in rule_value.items():
                    if old_name in attributes:
                        attributes[new_name] = attributes.pop(old_name)
            elif rule_key == "add":
                # Add new attributes (traceloop-specific ones)
                attributes.update(rule_value)

        # Add traceloop-specific attributes
        attributes.update(invocation.traceloop_attributes)

        # Update the invocation's attributes
        invocation.attributes = attributes

    def _on_after_start(self, invocation: LLMInvocation):
        """Hook called after span start - apply traceloop transformations."""
        if not isinstance(invocation, TraceloopInvocation):
            # If not a TraceloopInvocation, just call the parent implementation
            super()._on_after_start(invocation)
            return

        if invocation.span is None:
            return

        # Apply attribute transformations
        self._apply_attribute_transformations(invocation)

        # Re-apply attributes after transformation
        for k, v in invocation.attributes.items():
            invocation.span.set_attribute(k, v)

    def start(
        self, invocation: Union[LLMInvocation, TraceloopInvocation]
    ) -> None:
        """Start a new span with Traceloop-specific handling."""
        if isinstance(invocation, TraceloopInvocation):
            # Generate the base span name
            base_span_name = f"chat {invocation.request_model}"

            # Apply name transformations
            span_name = self._apply_name_transformations(
                invocation, base_span_name
            )

            # Create span with transformed name
            span = self._tracer.start_span(
                name=span_name, kind=trace.SpanKind.CLIENT
            )
            invocation.span = span

            # Set up context management
            from opentelemetry.trace import use_span

            cm = use_span(span, end_on_exit=False)
            cm.__enter__()
            invocation.context_token = cm

            # Apply base attributes first
            self._apply_start_attrs(invocation)

            # Apply traceloop-specific transformations
            self._on_after_start(invocation)
        else:
            # Handle regular LLMInvocation
            super().start(invocation)

    def finish(
        self, invocation: Union[LLMInvocation, TraceloopInvocation]
    ) -> None:
        """Finish the span with any final transformations."""
        if isinstance(invocation, TraceloopInvocation):
            if invocation.span is None:
                return

            invocation.end_time = time.time()

            # Apply any final attribute transformations
            self._apply_attribute_transformations(invocation)

            # Apply finish attributes
            self._apply_finish_attrs(invocation)

            # End the span
            token = invocation.context_token
            if token is not None and hasattr(token, "__exit__"):
                try:
                    token.__exit__(None, None, None)
                except Exception:
                    pass
            invocation.span.end()
        else:
            # Handle regular LLMInvocation
            super().finish(invocation)

    def error(
        self,
        error: Error,
        invocation: Union[LLMInvocation, TraceloopInvocation],
    ) -> None:
        """Handle error cases with Traceloop-specific handling."""
        if isinstance(invocation, TraceloopInvocation):
            if invocation.span is None:
                return

            invocation.end_time = time.time()

            # Set error status
            invocation.span.set_status(Status(StatusCode.ERROR, error.message))
            if invocation.span.is_recording():
                from opentelemetry.semconv.attributes import (
                    error_attributes as ErrorAttributes,
                )

                invocation.span.set_attribute(
                    ErrorAttributes.ERROR_TYPE, error.type.__qualname__
                )

            # Apply transformations even on error
            self._apply_attribute_transformations(invocation)
            self._apply_finish_attrs(invocation)

            # End the span
            token = invocation.context_token
            if token is not None and hasattr(token, "__exit__"):
                try:
                    token.__exit__(None, None, None)
                except Exception:
                    pass
            invocation.span.end()
        else:
            # Handle regular LLMInvocation
            super().error(error, invocation)
