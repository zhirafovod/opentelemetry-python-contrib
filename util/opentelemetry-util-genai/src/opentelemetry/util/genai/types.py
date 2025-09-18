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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

import time
from contextvars import Token
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, Union

from typing_extensions import TypeAlias

from opentelemetry.context import Context
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

ContextToken: TypeAlias = Token[Context]


@dataclass
class LLMInvocation:
    """
    Represents a single LLM call invocation.
    """

    run_id: UUID
    parent_run_id: Optional[UUID] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    input_messages: List[InputMessage] = field(default_factory=list)
    output_messages: List[OutputMessage] = field(default_factory=list)
    provider: Optional[str] = None
    response_model_name: Optional[str] = None
    response_id: Optional[str] = None
    input_tokens: Optional[AttributeValue] = None
    output_tokens: Optional[AttributeValue] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    span_id: int = 0
    trace_id: int = 0

@dataclass
class EmbeddingInvocation:
    """
    Represents a single Embedding call invocation.
    """
    run_id: UUID
    parent_run_id: Optional[UUID] = None
    start_time: float = field(default_factory=time.time)
    end_time: float = None
    dimension_count : int = 0
    input: Optional[List[str]] = None
    output: Optional[List[float]] = None
    attributes: dict = field(default_factory=dict)
