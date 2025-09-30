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

import time
from threading import Lock
from typing import Any, List, Optional, Union
from uuid import UUID, uuid4

from generators import SpanMetricEventGenerator, SpanMetricGenerator

from opentelemetry._events import get_event_logger
from opentelemetry._logs import get_logger
from opentelemetry.metrics import get_meter
from opentelemetry.semconv.schemas import Schemas
from opentelemetry.trace import get_tracer

from .data import ChatGeneration, Error, Message, ToolFunction, ToolOutput
from .types import LLMInvocation, ToolInvocation, TraceloopInvocation
from .version import __version__


class TelemetryClient:
    """
    High-level client managing GenAI invocation lifecycles and exporting
    them as spans, metrics, and events.
    """

    def __init__(self, exporter_type_full: bool = True, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        self._tracer = get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_28_0.value,
        )

        meter_provider = kwargs.get("meter_provider")
        self._meter = get_meter(
            __name__,
            __version__,
            meter_provider,
            schema_url=Schemas.V1_28_0.value,
        )

        event_logger_provider = kwargs.get("event_logger_provider")
        self._event_logger = get_event_logger(
            __name__,
            __version__,
            event_logger_provider=event_logger_provider,
            schema_url=Schemas.V1_28_0.value,
        )

        logger_provider = kwargs.get("logger_provider")
        self._logger = get_logger(
            __name__,
            __version__,
            logger_provider=logger_provider,
            schema_url=Schemas.V1_28_0.value,
        )

        self._generator = (
            SpanMetricEventGenerator(
                tracer=self._tracer,
                meter=self._meter,
                logger=self._logger,
            )
            if exporter_type_full
            else SpanMetricGenerator(tracer=self._tracer, meter=self._meter)
        )

        self._llm_registry: dict[
            UUID, Union[LLMInvocation, TraceloopInvocation]
        ] = {}
        self._tool_registry: dict[UUID, ToolInvocation] = {}
        self._lock = Lock()

    def start_llm(
        self,
        prompts: Optional[List[Message]] = None,
        tool_functions: Optional[List[ToolFunction]] = None,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        invocation: Optional[Union[LLMInvocation, TraceloopInvocation]] = None,
        **attributes: Any,
    ):
        """
        Start an LLM invocation.

        Can accept either:
        1. Traditional parameters (prompts, tool_functions, etc.) to create LLMInvocation
        2. Pre-built invocation object (LLMInvocation or TraceloopInvocation)
        """
        if invocation is not None:
            # Use the provided invocation (could be TraceloopInvocation)
            actual_invocation = invocation
            # Set run_id if not already set
            if run_id is not None:
                actual_invocation.run_id = run_id
            if parent_run_id is not None:
                actual_invocation.parent_run_id = parent_run_id
            # Merge any additional attributes
            actual_invocation.attributes.update(attributes)
        else:
            # Create traditional LLMInvocation
            actual_run_id = run_id or uuid4()
            actual_invocation = LLMInvocation(
                request_model=attributes.get("request_model", "unknown"),
                messages=prompts or [],
                run_id=actual_run_id,
                parent_run_id=parent_run_id,
                attributes=attributes,
            )
            # Handle tool_functions if provided
            if tool_functions:
                # Store tool functions in attributes for now
                actual_invocation.attributes["tool_functions"] = tool_functions

        with self._lock:
            self._llm_registry[actual_invocation.run_id] = actual_invocation
        self._generator.start(actual_invocation)
        return actual_invocation

    def stop_llm(
        self,
        run_id: UUID,
        chat_generations: Optional[List[ChatGeneration]] = None,
        **attributes: Any,
    ) -> Union[LLMInvocation, TraceloopInvocation]:
        with self._lock:
            invocation = self._llm_registry.pop(run_id)
        invocation.end_time = time.time()
        # Convert ChatGeneration to OutputMessage if needed (for now, store as-is)
        if chat_generations:
            # Store in attributes for compatibility
            invocation.attributes["chat_generations"] = chat_generations
        invocation.attributes.update(attributes)
        self._generator.finish(invocation)
        return invocation

    def fail_llm(
        self, run_id: UUID, error: Error, **attributes: Any
    ) -> Union[LLMInvocation, TraceloopInvocation]:
        with self._lock:
            invocation = self._llm_registry.pop(run_id)
        invocation.end_time = time.time()
        invocation.attributes.update(**attributes)
        self._generator.error(error, invocation)
        return invocation

    def start_tool(
        self,
        input_str: str,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **attributes,
    ):
        invocation = ToolInvocation(
            input_str=input_str,
            run_id=run_id,
            parent_run_id=parent_run_id,
            attributes=attributes,
        )
        with self._lock:
            self._tool_registry[invocation.run_id] = invocation
        self._generator.init_tool(invocation)

    def stop_tool(
        self, run_id: UUID, output: ToolOutput, **attributes
    ) -> ToolInvocation:
        with self._lock:
            invocation = self._tool_registry.pop(run_id)
        invocation.end_time = time.time()
        invocation.output = output
        self._generator.export_tool(invocation)
        return invocation

    def fail_tool(
        self, run_id: UUID, error: Error, **attributes
    ) -> ToolInvocation:
        with self._lock:
            invocation = self._tool_registry.pop(run_id)
        invocation.end_time = time.time()
        invocation.attributes.update(**attributes)
        self._generator.error_tool(error, invocation)
        return invocation


# Singleton accessor
_default_client: TelemetryClient | None = None


def get_telemetry_client(
    exporter_type_full: bool = True, **kwargs
) -> TelemetryClient:
    global _default_client
    if _default_client is None:
        _default_client = TelemetryClient(
            exporter_type_full=exporter_type_full, **kwargs
        )
    return _default_client


# Module‐level convenience functions
def llm_start(
    prompts: List[Message],
    run_id: UUID,
    parent_run_id: Optional[UUID] = None,
    **attributes,
):
    return get_telemetry_client().start_llm(
        prompts=prompts,
        run_id=run_id,
        parent_run_id=parent_run_id,
        **attributes,
    )


def llm_stop(
    run_id: UUID, chat_generations: List[ChatGeneration], **attributes
) -> LLMInvocation:
    return get_telemetry_client().stop_llm(
        run_id=run_id, chat_generations=chat_generations, **attributes
    )


def llm_fail(run_id: UUID, error: Error, **attributes) -> LLMInvocation:
    return get_telemetry_client().fail_llm(
        run_id=run_id, error=error, **attributes
    )
