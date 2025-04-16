"""
OpenTelemetry instrumentation for LangChain.
"""

import logging
from typing import Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.instrumentation.langchain.version import __version__
from opentelemetry.instrumentation.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import get_meter
from opentelemetry.trace import get_tracer
from opentelemetry._events import get_event_logger

# todo: fix me! newer versions are not backward compatible
_instruments = (
    "langchain >= 0.0.346",
    "langchain-core > 0.1.0",
)


class LangChainInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentor for LangChain.

    This adds a custom callback handler to the LangChain callback manager
    to capture chain, LLM, and tool events. It also wraps the internal
    OpenAI invocation points (BaseOpenAI) to inject W3C trace headers
    for downstream calls to OpenAI (or other providers).
    """

    def __init__(self, disable_trace_injection: bool = False):
        """
        :param disable_trace_injection: If True, do not wrap OpenAI invocation
                                        for trace-context injection.
        """
        super().__init__()
        self._disable_trace_injection = disable_trace_injection
        self._logger = logging.getLogger(__name__)

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        event_logger_provider = kwargs.get("event_logger_provider")

        tracer = get_tracer(__name__, __version__, tracer_provider)
        meter = get_meter(__name__, __version__, meter_provider)
        event_logger_provider = get_event_logger(__name__, __version__, meter_provider)

        # Create shared metrics: a duration histogram and a token-usage histogram
        duration_histogram = meter.create_histogram(
            name="gen_ai.client.operation_duration",
            unit="s",
            description="Duration of LLM/Chain/Tool operations within LangChain",
        )
        token_histogram = meter.create_histogram(
            name="gen_ai.client.token_usage",
            unit="token",
            description="Number of input and output tokens used by LLMs in LangChain",
        )

        otel_callback_handler = OpenTelemetryLangChainCallbackHandler(
            tracer=tracer,
            duration_histogram=duration_histogram,
            token_histogram=token_histogram,
        )

        # Hook BaseCallbackManager init so that our OTel handler is added
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(otel_callback_handler),
        )

        # Optionally wrap LangChain's "BaseOpenAI" methods to inject trace context
        if not self._disable_trace_injection:
            # todo: fixme! The below code is not compatible with the latest version. Need to split to different instrumentors
            # wrap_function_wrapper(
            #     module="langchain_community.llms.openai",
            #     name="BaseOpenAI._generate",
            #     wrapper=_OpenAITraceInjectionWrapper(otel_callback_handler),
            # )
            # wrap_function_wrapper(
            #     module="langchain_community.llms.openai",
            #     name="BaseOpenAI._agenerate",
            #     wrapper=_OpenAITraceInjectionWrapper(otel_callback_handler),
            # )
            # wrap_function_wrapper(
            #     module="langchain_community.llms.openai",
            #     name="BaseOpenAI._stream",
            #     wrapper=_OpenAITraceInjectionWrapper(otel_callback_handler),
            # )
            # wrap_function_wrapper(
            #     module="langchain_community.llms.openai",
            #     name="BaseOpenAI._astream",
            #     wrapper=_OpenAITraceInjectionWrapper(otel_callback_handler),
            # )
            #
            # If you also have "langchain_openai" installed:
            wrap_function_wrapper(
                module="langchain_openai.llms.base",
                name="BaseOpenAI._generate",
                wrapper=_OpenAITraceInjectionWrapper(otel_callback_handler),
            )
            wrap_function_wrapper(
                module="langchain_openai.llms.base",
                name="BaseOpenAI._agenerate",
                wrapper=_OpenAITraceInjectionWrapper(otel_callback_handler),
            )
            wrap_function_wrapper(
                module="langchain_openai.llms.base",
                name="BaseOpenAI._stream",
                wrapper=_OpenAITraceInjectionWrapper(otel_callback_handler),
            )
            wrap_function_wrapper(
                module="langchain_openai.llms.base",
                name="BaseOpenAI._astream",
                wrapper=_OpenAITraceInjectionWrapper(otel_callback_handler),
            )

            wrap_function_wrapper(
                module="langchain_openai.chat_models.base",
                name="BaseChatOpenAI._generate",
                wrapper=_OpenAITraceInjectionWrapper(otel_callback_handler),
            )
            wrap_function_wrapper(
                module="langchain_openai.chat_models.base",
                name="BaseChatOpenAI._agenerate",
                wrapper=_OpenAITraceInjectionWrapper(otel_callback_handler),
            )

    def _uninstrument(self, **kwargs):
        """
        Cleanup instrumentation (unwrap).
        """
        unwrap("langchain_core.callbacks.base", "BaseCallbackManager.__init__")
        if not self._disable_trace_injection:
            # unwrap("langchain_community.llms.openai", "BaseOpenAI._generate")
            # unwrap("langchain_community.llms.openai", "BaseOpenAI._agenerate")
            # unwrap("langchain_community.llms.openai", "BaseOpenAI._stream")
            # unwrap("langchain_community.llms.openai", "BaseOpenAI._astream")

            unwrap("langchain_openai.llms.base", "BaseOpenAI._generate")
            unwrap("langchain_openai.llms.base", "BaseOpenAI._agenerate")
            unwrap("langchain_openai.llms.base", "BaseOpenAI._stream")
            unwrap("langchain_openai.llms.base", "BaseOpenAI._astream")

            unwrap("langchain_openai.chat_models.base", "BaseChatOpenAI._generate")
            unwrap("langchain_openai.chat_models.base", "BaseChatOpenAI._agenerate")


class _BaseCallbackManagerInitWrapper:
    """
    Wrap the BaseCallbackManager __init__ so we can insert
    our custom callback handler in the manager's handlers list.
    """

    def __init__(self, callback_handler):
        self._otel_handler = callback_handler

    def __call__(self, wrapped, instance, args, kwargs):
        wrapped(*args, **kwargs)
        # Ensure our OTel callback is present if not already.
        for handler in instance.inheritable_handlers:
            if isinstance(handler, type(self._otel_handler)):
                break
        else:
            instance.add_handler(self._otel_handler, inherit=True)


class _OpenAITraceInjectionWrapper:
    """
    A wrapper that intercepts calls to the underlying LLM code in LangChain
    to inject W3C trace headers into upstream requests (if possible).
    """

    def __init__(self, callback_manager):
        self._otel_handler = callback_manager

    def __call__(self, wrapped, instance, args, kwargs):
        """
        We look up the run_id in the `kwargs["run_manager"]` to find
        the active span from the callback handler. Then we inject
        that span context into the 'extra_headers' for the openai call.
        """
        run_manager = kwargs.get("run_manager")
        if run_manager is not None:
            run_id = run_manager.run_id
            span_holder = self._otel_handler.spans.get(run_id)
            if span_holder and span_holder.span.is_recording():
                extra_headers = kwargs.get("extra_headers", {})
                context = span_holder.span_context
                # Use standard W3C injection
                from opentelemetry.trace.propagation.tracecontext import (
                    TraceContextTextMapPropagator,
                )
                from opentelemetry.trace import set_span_in_context

                ctx = set_span_in_context(span_holder.span)
                TraceContextTextMapPropagator().inject(extra_headers, context=ctx)
                kwargs["extra_headers"] = extra_headers

        return wrapped(*args, **kwargs)