"""
Generator for emitting spans with semantic conventions for GenAI data types.

Implements the BaseGenerator interface, providing extensibility hooks for subclasses
and allowing hooks to be specified for modifying telemetry before and after main operations.

Note: This class only defines the interface and hook call order. Actual span emission logic
should be implemented in subclasses or future versions.
"""

from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
)
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import (
    SpanKind,
    Status,
    StatusCode,
    get_tracer,
    set_span_in_context,
)
from opentelemetry.util.genai.generators.base_generator import BaseGenerator
from opentelemetry.util.genai.span_utils import _maybe_set_span_messages
from opentelemetry.util.genai.types.generic import GenAI
from opentelemetry.util.genai.types.invocations import LLMInvocation


class SemConvGenerator(BaseGenerator):
    """
    Generator for emitting spans with semantic conventions for GenAI data types.

    Implements the BaseGenerator interface, providing extensibility hooks for subclasses
    and allowing hooks to be specified for modifying telemetry before and after main operations.
    """

    def __init__(self, tracer_provider: Any = None):
        schema_url = Schemas.V1_36_0.value
        self._tracer = get_tracer(
            __name__,
            None,
            tracer_provider,
            schema_url=schema_url,
        )

    def start(self, data: GenAI) -> None:
        """
        Start telemetry emission for a GenAI data type (e.g., LLMInvocation).
        Calls protected hooks before and after the main logic.
        """
        self._on_before_start(data)
        self._start(data)
        self._on_after_start(data)

    def stop(self, data: GenAI) -> None:
        """
        Stop telemetry emission for a GenAI data type.
        Calls protected hooks before and after the main logic.
        """
        self._on_before_stop(data)
        self._stop(data)
        self._on_after_stop(data)

    def fail(self, data: GenAI, error: Exception) -> None:
        """
        Emit telemetry for a failed GenAI operation.
        Calls protected hooks before and after the main logic.
        """
        self._on_before_fail(data, error)
        self._fail(data, error)
        self._on_after_fail(data, error)

    def _set_llm_attributes(self, span, data: LLMInvocation) -> None:
        """
        Set LLMInvocation-specific attributes on the span.
        """
        span.set_attribute(
            gen_ai_attributes.GEN_AI_OPERATION_NAME,
            gen_ai_attributes.GenAiOperationNameValues.CHAT.value,
        )
        if data.provider:
            span.set_attribute(
                gen_ai_attributes.GEN_AI_PROVIDER_NAME, data.provider
            )
        if data.response_model_name:
            span.set_attribute(
                gen_ai_attributes.GEN_AI_RESPONSE_MODEL,
                data.response_model_name,
            )
        if data.response_id:
            span.set_attribute(
                gen_ai_attributes.GEN_AI_RESPONSE_ID, data.response_id
            )
        if data.input_tokens is not None:
            span.set_attribute(
                gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS, data.input_tokens
            )
        if data.output_tokens is not None:
            span.set_attribute(
                gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
                data.output_tokens,
            )
        _maybe_set_span_messages(
            span, data.input_messages, data.output_messages
        )

    def _start(self, data: GenAI) -> None:
        """
        Main logic for starting telemetry. Creates a span and attaches it to the GenAI data.
        """
        model = getattr(data, "request_model", None)
        span_name = (
            f"{gen_ai_attributes.GenAiOperationNameValues.CHAT.value} {data.request_model}",
        )
        span = self._tracer.start_span(
            name=span_name,
            kind=SpanKind.CLIENT,
        )
        # Set basic semantic attributes from GenAI data
        if model:
            span.set_attribute(gen_ai_attributes.GEN_AI_REQUEST_MODEL, model)
        if hasattr(data, "attributes") and isinstance(data.attributes, dict):
            for k, v in data.attributes.items():
                span.set_attribute(k, v)
        data.span = span
        data.context_token = otel_context.attach(set_span_in_context(span))
        if isinstance(data, LLMInvocation):
            self._set_llm_attributes(span, data)

    def _stop(self, data: GenAI) -> None:
        """
        Main logic for stopping telemetry. Ends the span and detaches context.
        Also updates the span name and attributes if they changed after start.
        """
        if (
            getattr(data, "context_token", None) is not None
            and getattr(data, "span", None) is not None
        ):
            span = data.span
            # Update span name if model name was set after start
            model = getattr(data, "request_model", None)
            if model:
                name = f"{gen_ai_attributes.GenAiOperationNameValues.CHAT.value} {data.request_model}"
                span.update_name(name)
                span.set_attribute(
                    gen_ai_attributes.GEN_AI_REQUEST_MODEL, model
                )
            if hasattr(data, "attributes") and isinstance(
                data.attributes, dict
            ):
                for k, v in data.attributes.items():
                    span.set_attribute(k, v)
            if isinstance(data, LLMInvocation):
                self._set_llm_attributes(span, data)
            otel_context.detach(data.context_token)
            span.end()

    def _fail(self, data: GenAI, error: Exception) -> None:
        """
        Main logic for handling telemetry on failure. Sets error attributes, ends span, detaches context.
        Also updates the span name and attributes if they changed after start.
        """
        if (
            getattr(data, "context_token", None) is not None
            and getattr(data, "span", None) is not None
        ):
            span = data.span
            # Update span name if model name was set after start
            model = getattr(data, "request_model", None)
            if model:
                name = f"{gen_ai_attributes.GenAiOperationNameValues.CHAT.value} {data.request_model}"
                span.update_name(name)
                span.set_attribute(
                    gen_ai_attributes.GEN_AI_REQUEST_MODEL, model
                )
            # Update attributes
            if hasattr(data, "attributes") and isinstance(
                data.attributes, dict
            ):
                for k, v in data.attributes.items():
                    span.set_attribute(k, v)
            span.set_status(Status(StatusCode.ERROR, error.message))
            if span.is_recording():
                span.set_attribute(
                    error_attributes.ERROR_TYPE, error.type.__qualname__
                )
            if isinstance(data, LLMInvocation):
                self._set_llm_attributes(span, data)
            otel_context.detach(data.context_token)
            span.end()

    def _on_before_start(self, data: GenAI) -> None:
        """
        Hook called before start logic. Can be overridden by subclasses.
        """
        pass

    def _on_after_start(self, data: GenAI) -> None:
        """
        Hook called after start logic. Can be overridden by subclasses.
        """
        pass

    def _on_before_stop(self, data: GenAI) -> None:
        """
        Hook called before stop logic. Can be overridden by subclasses.
        """
        pass

    def _on_after_stop(self, data: GenAI) -> None:
        """
        Hook called after stop logic. Can be overridden by subclasses.
        """
        pass

    def _on_before_fail(self, data: GenAI, error: Exception) -> None:
        """
        Hook called before fail logic. Can be overridden by subclasses.
        """
        pass

    def _on_after_fail(self, data: GenAI, error: Exception) -> None:
        """
        Hook called after fail logic. Can be overridden by subclasses.
        """
        pass
