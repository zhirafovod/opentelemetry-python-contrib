import inspect
from typing import Optional, Union, TypeVar, Callable, Awaitable

from typing_extensions import ParamSpec

from opentelemetry.genai.sdk.decorators.base import (
    entity_class,
    entity_method,
)
from opentelemetry.genai.sdk.utils.const import (
    ObserveSpanKindValues,
)

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[P, Union[R, Awaitable[R]]])


def task(
    name: Optional[str] = None,
    method_name: Optional[str] = None,
    tlp_span_kind: Optional[ObserveSpanKindValues] = ObserveSpanKindValues.TASK,
) -> Callable[[F], F]:
    def decorator(target):
        # Check if target is a class
        if inspect.isclass(target):
            return entity_class(
                name=name,
                method_name=method_name,
                tlp_span_kind=tlp_span_kind,
            )(target)
        else:
            # Target is a function/method
            return entity_method(
                name=name,
                tlp_span_kind=tlp_span_kind,
            )(target)
    return decorator


def workflow(
    name: Optional[str] = None,
    method_name: Optional[str] = None,
    tlp_span_kind: Optional[
        Union[ObserveSpanKindValues, str]
    ] = ObserveSpanKindValues.WORKFLOW,
) -> Callable[[F], F]:
    def decorator(target):
        # Check if target is a class
        if inspect.isclass(target):
            return entity_class(
                name=name,
                method_name=method_name,
                tlp_span_kind=tlp_span_kind,
            )(target)
        else:
            # Target is a function/method
            return entity_method(
                name=name,
                tlp_span_kind=tlp_span_kind,
            )(target)

    return decorator


def agent(
    name: Optional[str] = None,
    method_name: Optional[str] = None,
) -> Callable[[F], F]:
    return workflow(
        name=name,
        method_name=method_name,
        tlp_span_kind=ObserveSpanKindValues.AGENT,
    )


def tool(
    name: Optional[str] = None,
    method_name: Optional[str] = None,
) -> Callable[[F], F]:
    return task(
        name=name,
        method_name=method_name,
        tlp_span_kind=ObserveSpanKindValues.TOOL,
    )


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
        else:
            # Target is a function/method
            return entity_method(
                name=name,
                model_name=model_name,
                tlp_span_kind=ObserveSpanKindValues.LLM,
            )(target)
    return decorator
