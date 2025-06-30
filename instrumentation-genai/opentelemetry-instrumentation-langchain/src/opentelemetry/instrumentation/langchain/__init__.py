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
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics

# === New imports from our GenAI SDK ===
from opentelemetry.genai.sdk.api import (
    llm_start,
    llm_stop,
    llm_fail,
    tool_start,
    tool_stop,
)
from opentelemetry.genai.sdk.exporters import SpanMetricEventExporter

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
    OpenAI invocation points to inject W3C trace headers.
    """

    def __init__(self, disable_trace_injection: bool = False):
        super().__init__()
        self._disable_trace_injection = disable_trace_injection
        self._logger = logging.getLogger(__name__)
        # will hold our SDK exporter
        self._sdk_exporter = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        event_logger_provider = kwargs.get("event_logger_provider")

        tracer = get_tracer(__name__, __version__, tracer_provider)
        meter = get_meter(__name__, __version__, meter_provider)
        event_logger = get_event_logger(
            __name__, __version__, event_logger_provider=event_logger_provider
        )

        # Create shared metrics: duration + token histograms
        duration_histogram = meter.create_histogram(
            name=gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION,
            unit="s",
            description="GenAI operation duration",
        )
        token_histogram = meter.create_histogram(
            name=gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE,
            unit="{token}",
            description="Measures number of input and output tokens used",
        )

        # === Instantiate our GenAI SDK exporter ===
        self._sdk_exporter = SpanMetricEventExporter(tracer=tracer, meter=meter)

        # Create OTel callback handler and attach our exporter to it:
        otel_callback_handler = OpenTelemetryLangChainCallbackHandler(
            tracer=tracer,
            duration_histogram=duration_histogram,
            token_histogram=token_histogram,
            event_logger=event_logger,
        )
        # monkey-patch so handler can call into sdk_exporter
        otel_callback_handler.sdk_exporter = self._sdk_exporter

        # Hook BaseCallbackManager so our handler is registered:
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(otel_callback_handler),
        )

        # Optionally wrap LangChain's OpenAI entrypoints for trace injection:
        if not self._disable_trace_injection:
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
        unwrap("langchain_core.callbacks.base", "BaseCallbackManager.__init__")
        if not self._disable_trace_injection:
            unwrap("langchain_openai.llms.base", "BaseOpenAI._generate")
            unwrap("langchain_openai.llms.base", "BaseOpenAI._agenerate")
            unwrap("langchain_openai.llms.base", "BaseOpenAI._stream")
            unwrap("langchain_openai.llms.base", "BaseOpenAI._astream")
            unwrap("langchain_openai.chat_models.base", "BaseChatOpenAI._generate")
            unwrap("langchain_openai.chat_models.base", "BaseChatOpenAI._agenerate")


class _BaseCallbackManagerInitWrapper:
    def __init__(self, callback_handler):
        self._otel_handler = callback_handler

    def __call__(self, wrapped, instance, args, kwargs):
        wrapped(*args, **kwargs)
        for handler in instance.inheritable_handlers:
            if isinstance(handler, type(self._otel_handler)):
                break
        else:
            instance.add_handler(self._otel_handler, inherit=True)


class _OpenAITraceInjectionWrapper:
    def __init__(self, callback_manager):
        self._otel_handler = callback_manager

    def __call__(self, wrapped, instance, args, kwargs):
        run_manager = kwargs.get("run_manager")
        if run_manager is not None:
            run_id = run_manager.run_id
            span_holder = self._otel_handler.spans.get(run_id)
            if span_holder and span_holder.span.is_recording():
                extra_headers = kwargs.get("extra_headers", {})
                from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
                from opentelemetry.trace import set_span_in_context

                ctx = set_span_in_context(span_holder.span)
                TraceContextTextMapPropagator().inject(extra_headers, context=ctx)
                kwargs["extra_headers"] = extra_headers

        return wrapped(*args, **kwargs)
