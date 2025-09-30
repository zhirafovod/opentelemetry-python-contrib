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
Utilities for transforming existing spans into TraceloopInvocations
based on transformation rules.
"""

from typing import Any, Dict, Optional

from opentelemetry.sdk.trace import ReadableSpan

from ..types import TraceloopInvocation
from .traceloop_span_generator import TraceloopSpanGenerator


def create_traceloop_invocation_from_span(
    existing_span: ReadableSpan,
    attribute_transformations: Optional[Dict[str, Any]] = None,
    name_transformations: Optional[Dict[str, str]] = None,
    traceloop_attributes: Optional[Dict[str, Any]] = None,
    request_model: Optional[str] = None,
) -> TraceloopInvocation:
    """
    Create a TraceloopInvocation from an existing span, applying transformation rules.

    Args:
        existing_span: The original span to extract data from
        attribute_transformations: Rules for transforming attributes
        name_transformations: Rules for transforming span name
        traceloop_attributes: Additional Traceloop-specific attributes
        request_model: Override model name (extracted from span if not provided)

    Returns:
        TraceloopInvocation with transformed data
    """

    # Extract data from existing span
    span_attributes = (
        dict(existing_span.attributes) if existing_span.attributes else {}
    )
    span_name = existing_span.name

    # Determine request_model
    if request_model is None:
        # Try to extract from span attributes
        request_model = (
            span_attributes.get("gen_ai.request.model")
            or span_attributes.get("llm.request.model")
            or span_attributes.get("ai.model.name")
            or "unknown"
        )

    # Create TraceloopInvocation with extracted data
    invocation = TraceloopInvocation(
        request_model=request_model,
        attribute_transformations=attribute_transformations or {},
        name_transformations=name_transformations or {},
        traceloop_attributes=traceloop_attributes or {},
        attributes=span_attributes.copy(),  # Start with original attributes
        # Copy timing information if available
        start_time=existing_span.start_time / 1_000_000_000
        if existing_span.start_time
        else 0,  # Convert from nanoseconds
        end_time=existing_span.end_time / 1_000_000_000
        if existing_span.end_time
        else None,
    )

    return invocation


def transform_existing_span_to_telemetry(
    existing_span: ReadableSpan,
    attribute_transformations: Optional[Dict[str, Any]] = None,
    name_transformations: Optional[Dict[str, str]] = None,
    traceloop_attributes: Optional[Dict[str, Any]] = None,
    generator: Optional[TraceloopSpanGenerator] = None,
) -> TraceloopInvocation:
    """
    Transform an existing span into new telemetry using Traceloop transformation rules.

    Args:
        existing_span: The span to transform
        attribute_transformations: Transformation rules for attributes
        name_transformations: Transformation rules for span names
        traceloop_attributes: Additional Traceloop-specific attributes
        generator: Optional custom generator (creates default if not provided)

    Returns:
        TraceloopInvocation with new span created based on transformation rules
    """

    # Create TraceloopInvocation from existing span data
    invocation = create_traceloop_invocation_from_span(
        existing_span=existing_span,
        attribute_transformations=attribute_transformations,
        name_transformations=name_transformations,
        traceloop_attributes=traceloop_attributes,
    )

    # Create generator if not provided
    if generator is None:
        generator = TraceloopSpanGenerator(capture_content=True)

    # Generate new telemetry with transformations applied
    generator.start(invocation)

    if existing_span.end_time is not None:
        generator.finish(invocation)

    return invocation
