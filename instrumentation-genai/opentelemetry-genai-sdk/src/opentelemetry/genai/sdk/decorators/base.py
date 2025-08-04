import json
from functools import wraps
import os
from typing import Optional, TypeVar, Callable, Awaitable, Any, Union
import inspect
import traceback

from opentelemetry.genai.sdk.decorators.helpers import (
    _is_async_method,
    _get_original_function_name,
    _is_async_generator,
)

from opentelemetry.genai.sdk.decorators.util import camel_to_snake
from opentelemetry import trace
from opentelemetry import context as context_api
from typing_extensions import ParamSpec
from ..version import __version__

from opentelemetry.genai.sdk.utils.const import (
    ObserveSpanKindValues,
)

from opentelemetry.genai.sdk.data import Message, ChatGeneration
from opentelemetry.genai.sdk.exporters import _get_property_value

from opentelemetry.genai.sdk.api import get_telemetry_client

from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Configure OpenTelemetry providers (add this after imports)
def _configure_telemetry():
    """Configure OpenTelemetry providers if not already configured"""
    # Check if providers are already configured
    try:
        # Configure tracing
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        
        # Check environment variable to decide which exporter to use
        span_exporter = OTLPSpanExporter()
        
        span_processor = BatchSpanProcessor(span_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Configure metrics
        metric_exporter = OTLPMetricExporter()
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        # configure logging and events
        # _logs.set_logger_provider(LoggerProvider())
        # _logs.get_logger_provider().add_log_record_processor(
        #     BatchLogRecordProcessor(OTLPLogExporter())
        # )
        # _events.set_event_logger_provider(EventLoggerProvider())
    except Exception as e:
        print(f"Warning: Failed to configure OpenTelemetry providers - {e}")

_configure_telemetry()


P = ParamSpec("P")

R = TypeVar("R")
F = TypeVar("F", bound=Callable[P, Union[R, Awaitable[R]]])

OTEL_INSTRUMENTATION_GENAI_EXPORTER = (
    "OTEL_INSTRUMENTATION_GENAI_EXPORTER"
)


def should_emit_events() -> bool:
    val = os.getenv(OTEL_INSTRUMENTATION_GENAI_EXPORTER, "SpanMetricEventExporter")
    if val.strip().lower() == "spanmetriceventexporter":
        return True
    elif val.strip().lower() == "spanmetricexporter":
        return False
    else:
        raise ValueError(f"Unknown exporter_type: {val}")

exporter_type_full = should_emit_events()

# Instantiate a singleton TelemetryClient bound to our tracer & meter
telemetry = get_telemetry_client(exporter_type_full)


def _should_send_prompts():
    return (
        os.getenv("OBSERVE_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _handle_llm_span_attributes(tlp_span_kind, args, kwargs, res=None):
    """Add GenAI-specific attributes to span for LLM operations by delegating to TelemetryClient logic."""
    if tlp_span_kind != ObserveSpanKindValues.LLM:
        return

    # Import here to avoid circular import issues
    from uuid import uuid4
    import contextlib

    # Extract messages and attributes as before
    messages = _extract_messages_from_args_kwargs(args, kwargs)
    attributes = _extract_llm_attributes_from_args_kwargs(args, kwargs, res)
    run_id = uuid4()

    # Pass the current span to TelemetryClient via context
    # context_api.set_value("active_llm_span", span)

    try:
        telemetry.start_llm(prompts=messages, run_id=run_id, **attributes)
    except Exception as e:
        print(f"Warning: TelemetryClient.start_llm failed: {e}")
        return

    if res:
        chat_generations = _extract_chat_generations_from_response(res)
        try:
            with contextlib.suppress(Exception):
                telemetry.stop_llm(run_id=run_id, chat_generations=chat_generations)
        except Exception as e:
            print(f"Warning: TelemetryClient.stop_llm failed: {e}")


def _extract_messages_from_args_kwargs(args, kwargs):
    """Extract messages from function arguments using patterns similar to exporters"""
    messages = []
    
    # Try different patterns to find messages
    raw_messages = None
    if kwargs.get('messages'):
        raw_messages = kwargs['messages']
    elif kwargs.get('inputs'):  # Sometimes messages are in inputs
        inputs = kwargs['inputs']
        if isinstance(inputs, dict) and 'messages' in inputs:
            raw_messages = inputs['messages']
    elif len(args) > 0:
        # Try to find messages in args
        for arg in args:
            if hasattr(arg, 'messages'):
                raw_messages = arg.messages
                break
            elif isinstance(arg, list) and arg and hasattr(arg[0], 'content'):
                raw_messages = arg
                break
    
    # Convert to Message objects using similar logic as exporters
    if raw_messages:
        for msg in raw_messages:
            content = _get_property_value(msg, "content")
            msg_type = _get_property_value(msg, "type") or _get_property_value(msg, "role")
            # Convert 'human' to 'user' like in exporters
            msg_type = "user" if msg_type == "human" else msg_type
            
            if content and msg_type:
                messages.append(Message(content=str(content), type=str(msg_type)))
    
    return messages


def _extract_llm_attributes_from_args_kwargs(args, kwargs, res=None):
    """Extract LLM attributes from function arguments"""
    attributes = {}
    
    # Extract model information
    model = None
    if kwargs.get('model'):
        model = kwargs['model']
    elif kwargs.get('model_name'):
        model = kwargs['model_name']
    elif len(args) > 0 and hasattr(args[0], 'model'):
        model = getattr(args[0], 'model', None)
    elif len(args) > 0 and isinstance(args[0], str):
        model = args[0]  # Sometimes model is the first string argument
    
    if model:
        attributes['request_model'] = str(model)
    
    # Extract system/framework information
    system = None
    framework = None
    
    if kwargs.get('system'):
        system = kwargs['system']
    elif hasattr(args[0] if args else None, '__class__'):
        # Try to infer system from class name
        class_name = args[0].__class__.__name__.lower()
        if 'openai' in class_name or 'gpt' in class_name:
            system = 'openai'
        elif 'anthropic' in class_name or 'claude' in class_name:
            system = 'anthropic'
        elif 'google' in class_name or 'gemini' in class_name:
            system = 'google'
        elif 'langchain' in class_name:
            system = 'langchain'
            framework = 'langchain'
    
    if system is not None:
        attributes['system'] = system

    if 'framework' in kwargs and kwargs['framework'] is not None:
        framework = kwargs['framework']
    else:
        framework = "unknown"
        
    if framework:
        attributes['framework'] = framework
    
    # Extract response attributes if available
    if res:
        _extract_response_attributes(res, attributes)
    
    return attributes


def _extract_response_attributes(res, attributes):
    """Extract attributes from response similar to exporter logic"""
    try:
        # Check if res has response_metadata attribute directly
        metadata = None
        if hasattr(res, 'response_metadata'):
            metadata = res.response_metadata
        elif isinstance(res, str):
            # If res is a string, try to parse it as JSON
            try:
                parsed_res = json.loads(res)
                metadata = parsed_res.get('response_metadata')
            except:
                pass
        
        # Extract token usage if available
        if metadata and 'token_usage' in metadata:
            usage = metadata['token_usage']
            if 'prompt_tokens' in usage:
                attributes['input_tokens'] = usage['prompt_tokens']
            if 'completion_tokens' in usage:
                attributes['output_tokens'] = usage['completion_tokens']
        
        # Extract response model
        if metadata and 'model_name' in metadata:
            attributes['response_model_name'] = metadata['model_name']
        
        # Extract response ID
        if hasattr(res, 'id'):
            attributes['response_id'] = res.id
            
    except Exception:
        # Silently ignore errors in extracting response attributes
        pass


def _extract_chat_generations_from_response(res):
    """Extract chat generations from response similar to exporter logic"""
    chat_generations = []
    
    try:
        # Handle OpenAI-style responses with choices
        if hasattr(res, 'choices') and res.choices:
            for choice in res.choices:
                content = None
                finish_reason = None
                msg_type = "assistant"
                
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content
                    if hasattr(choice.message, 'role'):
                        msg_type = choice.message.role
                
                if hasattr(choice, 'finish_reason'):
                    finish_reason = choice.finish_reason
                
                if content:
                    chat_generations.append(ChatGeneration(
                        content=str(content),
                        finish_reason=finish_reason,
                        type=str(msg_type)
                    ))
        
        # Handle responses with direct content attribute (e.g., some LangChain responses)
        elif hasattr(res, 'content'):
            msg_type = "assistant"
            if hasattr(res, 'type'):
                msg_type = res.type
            
            chat_generations.append(ChatGeneration(
                content=str(res.content),
                finish_reason="stop",  # May not be available
                type=str(msg_type)
            ))
            
    except Exception:
        # Silently ignore errors in extracting chat generations
        pass
    
    return chat_generations


def _unwrap_structured_tool(fn):
    # Unwraps StructuredTool or similar wrappers to get the underlying function
    if hasattr(fn, "func") and callable(fn.func):
        return fn.func
    return fn


def entity_method(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[int] = None,
    protocol: Optional[str] = None,
    tlp_span_kind: Optional[ObserveSpanKindValues] = ObserveSpanKindValues.TASK,
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        # Unwrap StructuredTool if present
        fn = _unwrap_structured_tool(fn)
        is_async = _is_async_method(fn)
        entity_name = name or _get_original_function_name(fn)
        if is_async:
            if _is_async_generator(fn):

                @wraps(fn)
                async def async_gen_wrap(*args: Any, **kwargs: Any) -> Any:
                    
                    # add entity_name to kwargs
                    kwargs["system"] = entity_name
                    _handle_llm_span_attributes(tlp_span_kind, args, kwargs)

                return async_gen_wrap
            else:

                @wraps(fn)
                async def async_wrap(*args, **kwargs):
                    try:
                        res = await fn(*args, **kwargs)

                        # Add GenAI-specific attributes from response for LLM spans
                        _handle_llm_span_attributes(tlp_span_kind, args, kwargs, res)                        
                    except Exception as e:
                        print(traceback.format_exc())
                        raise e
                    return res

            decorated = async_wrap
        else:

            @wraps(fn)
            def sync_wrap(*args: Any, **kwargs: Any) -> Any:
                try:
                    res = fn(*args, **kwargs)
                    # Add entity_name to kwargs
                    kwargs["system"] = entity_name
                    # Add GenAI-specific attributes from response for LLM spans
                    _handle_llm_span_attributes(tlp_span_kind, args, kwargs, res)
                    
                except Exception as e:
                    print(traceback.format_exc())
                    raise e
                return res

            decorated = sync_wrap
        # # If the original fn was a StructuredTool, re-wrap
        if hasattr(fn, "func") and callable(fn.func):
            fn.func = decorated
            return fn
        return decorated

    return decorate


def entity_class(
    name: Optional[str],
    description: Optional[str],
    version: Optional[int],
    protocol: Optional[str],
    method_name: Optional[str],
    tlp_span_kind: Optional[ObserveSpanKindValues] = ObserveSpanKindValues.TASK,
):
    def decorator(cls):
        task_name = name if name else camel_to_snake(cls.__qualname__)

        methods_to_wrap = []

        if method_name:
            # Specific method specified - existing behavior
            methods_to_wrap = [method_name]
        else:
            # No method specified - wrap all public methods defined in this class
            for attr_name in dir(cls):
                if (
                    not attr_name.startswith("_")  # Skip private/built-in methods
                    and attr_name != "mro"  # Skip class method
                    and hasattr(cls, attr_name)
                ):
                    attr = getattr(cls, attr_name)
                    # Only wrap functions defined in this class (not inherited methods or built-ins)
                    if (
                        inspect.isfunction(attr)  # Functions defined in the class
                        and not isinstance(attr, (classmethod, staticmethod, property))
                        and hasattr(attr, "__qualname__")  # Has qualname attribute
                        and attr.__qualname__.startswith(
                            cls.__name__ + "."
                        )  # Defined in this class
                    ):
                        # Additional check: ensure the function has a proper signature with 'self' parameter
                        try:
                            sig = inspect.signature(attr)
                            params = list(sig.parameters.keys())
                            if params and params[0] == "self":
                                methods_to_wrap.append(attr_name)
                        except (ValueError, TypeError):
                            # Skip methods that can't be inspected
                            continue

        # Wrap all detected methods
        for method_to_wrap in methods_to_wrap:
            if hasattr(cls, method_to_wrap):
                original_method = getattr(cls, method_to_wrap)
                # Only wrap actual functions defined in this class
                unwrapped_method = _unwrap_structured_tool(original_method)
                if inspect.isfunction(unwrapped_method):
                    try:
                        # Verify the method has a proper signature
                        sig = inspect.signature(unwrapped_method)
                        wrapped_method = entity_method(
                            name=f"{task_name}.{method_to_wrap}",
                            description=description,
                            version=version,
                            protocol=protocol,
                            tlp_span_kind=tlp_span_kind,
                        )(unwrapped_method)
                        # Set the wrapped method on the class
                        setattr(cls, method_to_wrap, wrapped_method)
                    except Exception:
                        # Don't wrap methods that can't be properly decorated
                        continue

        return cls

    return decorator

