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

# === Now we delegate to our GenAI SDK ===
from opentelemetry.genai.sdk.api import get_telemetry_client
from opentelemetry.genai.sdk.api import TelemetryClient

# todo: fix me! newer versions are not backward compatible
_instruments = (
    "langchain >= 0.0.346",
    "langchain-core > 0.1.0",
)


class LangChainInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentor for LangChain.

    Delegates all GenAI spans/metrics/events to the GenAI SDK's TelemetryClient.
    """

    def __init__(self, disable_trace_injection: bool = False):
        super().__init__()
        self._disable_trace_injection = disable_trace_injection
        self._logger = logging.getLogger(__name__)

        self._telemetry: TelemetryClient | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        # TODO: need to move telemetry provider instantiation to the TelemetryClient client we should pass kwargs context into the TelemetryClient._instrument(class_name ,kwargs) additionally in TelemetryClient we need to check if these providers are not instantiated by instrumentation frameworks higher in the stack call and instantiate it them (use case - when TelemetryClient is used from LiteLLM Opentelemetry Integration
        ## move from here ->
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        event_logger_provider = kwargs.get("event_logger_provider")

        tracer = get_tracer(__name__, __version__, tracer_provider)
        meter = get_meter(__name__, __version__, meter_provider)
        event_logger = get_event_logger(
            __name__, __version__, event_logger_provider=event_logger_provider
        )
        ### <- to here

        # Instantiate a singleton TelemetryClient bound to our tracer & meter
        self._telemetry = get_telemetry_client(tracer=tracer, meter=meter, exporter_type="full")

        # Create OTel callback handler and attach our telemetry client
        handler = OpenTelemetryLangChainCallbackHandler(
            tracer=tracer,
            duration_histogram=meter.create_histogram(
                name=gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION,
                unit="s",
                description="GenAI operation duration",
            ),
            token_histogram=meter.create_histogram(
                name=gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE,
                unit="{token}",
                description="Number of tokens used",
            ),
            event_logger=event_logger,
        )
        # allow the handler to call telemetry.start/stop
        handler.telemetry_client = self._telemetry

        # Hook BaseCallbackManager so our handler is registered
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInitWrapper(handler),
        )

        # Trace-context injection wrappers (unchanged)…
        if not self._disable_trace_injection:
            for mod, fn in [
                ("langchain_openai.llms.base", "BaseOpenAI._generate"),
                ("langchain_openai.llms.base", "BaseOpenAI._agenerate"),
                ("langchain_openai.llms.base", "BaseOpenAI._stream"),
                ("langchain_openai.llms.base", "BaseOpenAI._astream"),
                ("langchain_openai.chat_models.base", "BaseChatOpenAI._generate"),
                ("langchain_openai.chat_models.base", "BaseChatOpenAI._agenerate"),
            ]:
                wrap_function_wrapper(module=mod, name=fn, wrapper=_OpenAITraceInjectionWrapper(handler))

    def _uninstrument(self, **kwargs):
        unwrap("langchain_core.callbacks.base", "BaseCallbackManager.__init__")
        if not self._disable_trace_injection:
            for mod, fn in [
                ("langchain_openai.llms.base", "BaseOpenAI._generate"),
                ("langchain_openai.llms.base", "BaseOpenAI._agenerate"),
                ("langchain_openai.llms.base", "BaseOpenAI._stream"),
                ("langchain_openai.llms.base", "BaseOpenAI._astream"),
                ("langchain_openai.chat_models.base", "BaseChatOpenAI._generate"),
                ("langchain_openai.chat_models.base", "BaseChatOpenAI._agenerate"),
            ]:
                unwrap(mod, fn)


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
