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

    # Create an invocation object with your request data
    invocation = LLMInvocation(
        request_model="my-model",
        input_messages=[...],
        provider="my-provider",
        attributes={"custom": "attr"},
    )

    # Start the invocation (opens a span)
    handler.start_llm(invocation)

    # Populate outputs and any additional attributes, then stop (closes the span)
    invocation.output_messages = [...]
    invocation.attributes.update({"more": "attrs"})
    handler.stop_llm(invocation)

    # Or, in case of error
    # handler.fail_llm(invocation, Error(type="...", message="..."))
"""

import time
from typing import Any, Optional

from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import get_tracer

from .data import ChatGeneration, Error, Message
from .generators import SpanMetricEventGenerator, SpanMetricGenerator
from .types import LLMInvocation, EmbeddingInvocation
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

# TODO: Get the tool version for emitting spans, use GenAI Utils for now
from .version import __version__


def _apply_known_attrs_to_invocation(
    invocation: LLMInvocation, attributes: dict[str, Any]
) -> None:
    """Pop known fields from attributes and set them on the invocation.

    Mutates the provided attributes dict by popping known keys, leaving
    only unknown/custom attributes behind for the caller to persist into
    invocation.attributes.
    """
    if "provider" in attributes:
        invocation.provider = attributes.pop("provider")
    if "response_model_name" in attributes:
        invocation.response_model_name = attributes.pop("response_model_name")
    if "response_id" in attributes:
        invocation.response_id = attributes.pop("response_id")
    if "input_tokens" in attributes:
        invocation.input_tokens = attributes.pop("input_tokens")
    if "output_tokens" in attributes:
        invocation.output_tokens = attributes.pop("output_tokens")


class TelemetryHandler:
    """
    High-level handler managing GenAI invocation lifecycles and emitting
    them as spans, metrics, and events.
    """

    def __init__(self, **kwargs: Any):
        tracer_provider = kwargs.get("tracer_provider")
        self._tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_36_0.value,
        )

        meter_provider = kwargs.get("meter_provider")
        self._meter = get_meter(
            __name__,
            __version__,
            meter_provider,
            schema_url=Schemas.V1_36_0.value,
        )

        event_logger_provider = kwargs.get("event_logger_provider")
        self._event_logger = get_event_logger(
            __name__,
            __version__,
            event_logger_provider=event_logger_provider,
            schema_url=Schemas.V1_36_0.value,
        )

        logger_provider = kwargs.get("logger_provider")
        self._logger = get_logger(
            __name__,
            __version__,
            logger_provider=logger_provider,
            schema_url=Schemas.V1_36_0.value,
        )

        self._generator = (
            SpanMetricEventGenerator(
                tracer=self._tracer,
                meter=self._meter,
                logger=self._logger,
                capture_content=self._should_collect_content(),
            )
            if emitter_type_full
            else SpanMetricGenerator(
                tracer=self._tracer,
                meter=self._meter,
                capture_content=self._should_collect_content(),
            )
        )

        self._llm_registry: dict[UUID, LLMInvocation] = {}
        self._embedding_registry: dict[UUID, EmbeddingInvocation] = {}
        self._lock = Lock()

    @staticmethod
    def _should_collect_content() -> bool:
        return True  # Placeholder for future config

    def start_llm(
        self,
        invocation: LLMInvocation,
    ) -> LLMInvocation:
        """Start an LLM invocation and create a pending span entry."""
        self._generator.start(invocation)
        return invocation

    def stop_llm(self, invocation: LLMInvocation) -> LLMInvocation:
        """Finalize an LLM invocation successfully and end its span."""
        invocation.end_time = time.time()
        self._generator.finish(invocation)
        return invocation

    def fail_llm(
        self, invocation: LLMInvocation, error: Error
    ) -> LLMInvocation:
        """Fail an LLM invocation and end its span with error status."""
        invocation.end_time = time.time()
        self._generator.error(error, invocation)
        return invocation

    def start_embedding(
        self,
        run_id: UUID,
        model_name: str,
        parent_run_id: Optional[UUID] = None,
        **attributes: Any,
    ) -> None:
        """Start an embedding invocation."""
        # Create span attributes
        span_attributes = {
            gen_ai_attributes.GEN_AI_OPERATION_NAME: gen_ai_attributes.GenAiOperationNameValues.EMBEDDINGS.value,
            gen_ai_attributes.GEN_AI_REQUEST_MODEL: model_name,
        }
        span_attributes.update(attributes)

        invocation = EmbeddingInvocation(
            run_id=run_id,
            parent_run_id=parent_run_id,
            attributes=span_attributes,
            input=attributes.get("input", None),
        )

        with self._lock:
            self._embedding_registry[invocation.run_id] = invocation

        self._generator.start(invocation)

    def stop_embedding(
        self,
        run_id: UUID,
        dimension_count: int,
        output: List[float],
        **attributes: Any,
    ) -> EmbeddingInvocation:
        """Stop an embedding invocation with results."""
        with self._lock:
            invocation = self._embedding_registry.pop(run_id)
        invocation.end_time = time.time()
        invocation.dimension_count = dimension_count
        invocation.attributes.update(attributes)
        invocation.output = output
        self._generator.finish(invocation)
        return invocation

    def fail_embedding(
        self, run_id: UUID, error: Error, **attributes: Any
    ) -> EmbeddingInvocation:
        """Fail an embedding invocation with error."""
        with self._lock:
            invocation = self._embedding_registry.pop(run_id)
        invocation.end_time = time.time()
        invocation.attributes.update(**attributes)
        self._generator.error(error, invocation)
        return invocation


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


# Module‐level convenience functions
def llm_start(
    prompts: List[Message],
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **attributes: Any,
) -> None:
    return get_telemetry_handler().start_llm(
        prompts=prompts,
        run_id=run_id,
        parent_run_id=parent_run_id,
        **attributes,
    )


def llm_stop(
    run_id: UUID, chat_generations: List[ChatGeneration], **attributes: Any
) -> LLMInvocation:
    return get_telemetry_handler().stop_llm(
        run_id=run_id, chat_generations=chat_generations, **attributes
    )


def llm_fail(run_id: UUID, error: Error, **attributes: Any) -> LLMInvocation:
    return get_telemetry_handler().fail_llm(
        run_id=run_id, error=error, **attributes
    )


def embedding_start(
    model_name: str,
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **attributes: Any,
) -> None:
    """Start an embedding invocation."""
    return get_telemetry_handler().start_embedding(
        run_id=run_id,
        model_name=model_name,
        parent_run_id=parent_run_id,
        **attributes,
    )


def embedding_stop(
    run_id: UUID,
    dimension_count: int,
    output: List[float],
    **attributes: Any,
) -> EmbeddingInvocation:
    """Stop an embedding invocation with results."""
    return get_telemetry_handler().stop_embedding(
        run_id=run_id,
        dimension_count=dimension_count,
        output=output,
        **attributes,
    )


def embedding_fail(
    run_id: UUID, error: Error, **attributes: Any
) -> EmbeddingInvocation:
    """Fail an embedding invocation with error."""
    return get_telemetry_handler().fail_embedding(
        run_id=run_id, error=error, **attributes
    )
