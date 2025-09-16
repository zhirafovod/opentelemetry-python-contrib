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

import inspect
from typing import Awaitable, Callable, Optional, TypeVar, Union

from typing_extensions import ParamSpec

from opentelemetry.util.genai.decorators import (
    entity_class,
    entity_method,
)
from opentelemetry.util.genai.types import (
    ObserveSpanKindValues,
)

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[P, Union[R, Awaitable[R]]])


def tool(
    name: Optional[str] = None,
    method_name: Optional[str] = None,
    tlp_span_kind: Optional[
        ObserveSpanKindValues
    ] = ObserveSpanKindValues.TOOL,
) -> Callable[[F], F]:
    def decorator(target):
        # Check if target is a class
        if inspect.isclass(target):
            return entity_class(
                name=name,
                method_name=method_name,
                tlp_span_kind=tlp_span_kind,
            )(target)
        # Target is a function/method
        return entity_method(
            name=name,
            tlp_span_kind=tlp_span_kind,
        )(target)

    return decorator


def llm(
    name: Optional[str] = None,
    model_name: Optional[str] = None,
    method_name: Optional[str] = None,
) -> Callable[[F], F]:
    def decorator(target):
        # Check if target is a class
        if inspect.isclass(target):
            return entity_class(
                name=name,
                model_name=model_name,
                method_name=method_name,
                tlp_span_kind=ObserveSpanKindValues.LLM,
            )(target)
        # Target is a function/method
        return entity_method(
            name=name,
            model_name=model_name,
            tlp_span_kind=ObserveSpanKindValues.LLM,
        )(target)

    return decorator
