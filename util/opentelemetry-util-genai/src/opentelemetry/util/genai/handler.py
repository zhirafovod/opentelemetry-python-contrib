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
Telemetry handler for GenAI invocations.

This module exposes the `TelemetryHandler` class, which manages the lifecycle of
GenAI (Generative AI) invocations and emits telemetry data (spans and related attributes).
It supports starting, stopping, and failing LLM invocations.

Classes:
    - TelemetryHandler: Manages GenAI invocation lifecycles and emits telemetry.

Functions:
    - get_telemetry_handler: Returns a singleton `TelemetryHandler` instance.

Usage:
    handler = get_telemetry_handler()

    # Create a request object with your input data
    request = LLMRequest(
        request_model="my-model",
        input_messages=[...],
        provider="my-provider",
        attributes={"custom": "attr"},
    )

    # Start the invocation (opens a span) - returns a new invocation with telemetry
    invocation = handler.start_llm(request)

    # Create response data after LLM call completes
    response = LLMResponse(
        output_messages=[...],
        response_model_name="my-model",
        input_tokens=100,
        output_tokens=50,
    )

    # Stop the invocation (closes the span) - returns final invocation
    final_invocation = handler.stop_llm(invocation, response)

    # Or, in case of error
    handler.fail_llm(invocation, Error(type=Exception, message="..."))
"""

import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import (
    SpanKind,
    Tracer,
    get_tracer,
    set_span_in_context,
)
from opentelemetry.util.genai.span_utils import (
    _apply_error_attributes,
    _apply_finish_attributes,
)
from opentelemetry.util.genai.types import Error, LLMInvocation, LLMRequest, LLMResponse
from opentelemetry.util.genai.version import __version__


class TelemetryHandler:
    """
    High-level handler managing GenAI invocation lifecycles and emitting
    them as spans, metrics, and events.
    """

    def __init__(self, **kwargs: Any):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_36_0.value,
        )
        self._tracer: Tracer = tracer or trace.get_tracer(__name__)

    def start_llm(self, request: LLMRequest) -> LLMInvocation:
        """Start an LLM invocation from a request and create a pending span."""
        # Create a span and attach it as current; keep the token to detach later
        span = self._tracer.start_span(
            name=f"{GenAI.GenAiOperationNameValues.CHAT.value} {request.request_model}",
            kind=SpanKind.CLIENT,
        )
        context_token = otel_context.attach(set_span_in_context(span))

        # Return a new immutable invocation with the span and context_token set
        return LLMInvocation(
            request=request,
            span=span,
            context_token=context_token
        )

    def stop_llm(self, invocation: LLMInvocation, response: LLMResponse) -> LLMInvocation:
        """Finalize an LLM invocation successfully with response data and end its span."""
        end_time = time.time()
        if invocation.context_token is None or invocation.span is None:
            # TODO: Provide feedback that this invocation was not started
            return invocation

        # Create the final invocation with response data
        final_invocation = LLMInvocation(
            request=invocation.request,
            response=response,
            start_time=invocation.start_time,
            end_time=end_time,
            span=invocation.span,
            context_token=invocation.context_token
        )

        _apply_finish_attributes(invocation.span, final_invocation)
        # Detach context and end span
        otel_context.detach(invocation.context_token)
        invocation.span.end()

        return final_invocation

    def fail_llm(self, invocation: LLMInvocation, error: Error) -> LLMInvocation:
        """Fail an LLM invocation and end its span with error status."""
        end_time = time.time()
        if invocation.context_token is None or invocation.span is None:
            # TODO: Provide feedback that this invocation was not started
            return invocation

        _apply_error_attributes(invocation.span, error)
        # Detach context and end span
        otel_context.detach(invocation.context_token)
        invocation.span.end()

        # Return a new immutable invocation with the end_time set
        return LLMInvocation(
            request=invocation.request,
            response=invocation.response,
            start_time=invocation.start_time,
            end_time=end_time,
            span=invocation.span,
            context_token=invocation.context_token
        )

    @contextmanager
    def llm(self, request: LLMRequest) -> Generator[LLMInvocation, None, None]:
        """
        Context manager for LLM invocations that automatically handles span lifecycle.

        Usage:
            request = LLMRequest(request_model="my-model", input_messages=[...])
            with handler.llm(request) as invocation:
                # Make your LLM call here
                # ... your LLM logic ...

                # To add response data, call stop_llm explicitly:
                response = LLMResponse(output_messages=[...])
                handler.stop_llm(invocation, response)

        Args:
            request: The LLM request containing model and input data

        Yields:
            LLMInvocation: The started invocation with an active span
        """
        invocation = self.start_llm(request)
        span_ended = False

        try:
            yield invocation
        except Exception as e:
            # Handle any exception that occurs during the context
            error = Error(type=type(e), message=str(e))
            self.fail_llm(invocation, error)
            span_ended = True
            raise
        finally:
            # Only end the span if it wasn't already ended by stop_llm or fail_llm
            if not span_ended and invocation.context_token is not None and invocation.span is not None:
                # Check if the span is still recording (not ended by stop_llm)
                if invocation.span.is_recording():
                    otel_context.detach(invocation.context_token)
                    invocation.span.end()


def get_telemetry_handler(**kwargs: Any) -> TelemetryHandler:
    """
    Returns a singleton TelemetryHandler instance.
    """
    handler: Optional[TelemetryHandler] = getattr(
        get_telemetry_handler, "_default_handler", None
    )
    if handler is None:
        handler = TelemetryHandler(**kwargs)
        setattr(get_telemetry_handler, "_default_handler", handler)
    return handler
