import dataclasses
import datetime
import json
import logging
import os
import traceback
from time import time_ns

from langchain_core.documents import Document
from opentelemetry._events import Event
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as GenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# By default, we do not record prompt or completion content. Set this
# environment variable to "true" to enable collection of message text.
OTEL_INSTRUMENTATION_LANGCHAIN_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_LANGCHAIN_CAPTURE_MESSAGE_CONTENT"
)


def should_collect_content() -> bool:
    val = os.getenv(OTEL_INSTRUMENTATION_LANGCHAIN_CAPTURE_MESSAGE_CONTENT, "false")
    return val.strip().lower() == "true"


def dont_throw(func):
    """
    Decorator that catches and logs exceptions, rather than re-raising them,
    to avoid interfering with user code if instrumentation fails.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "OpenTelemetry instrumentation for LangChain encountered an error in %s: %s",
                func.__name__,
                traceback.format_exc(),
            )
            from opentelemetry.instrumentation.langchain.config import Config
            if Config.exception_logger:
                Config.exception_logger(e)
            return None
    return wrapper

def get_property_value(obj, property_name):
    if isinstance(obj, dict):
        return obj.get(property_name, None)

    return getattr(obj, property_name, None)

def message_to_event(message, tools, content, type):
    if should_collect_content() and content is not None:
        attributes = {
            GenAI.GEN_AI_SYSTEM: "langchain"
        }
        body = {}
        if type == "tool":
            name = get_property_value(message, "name")
            tool_call_id = get_property_value(message, "tool_call_id")
            body.update([
                ("content",content),
                ("name",name),
                ("tool_call_id",tool_call_id)]
            )
        elif type == "ai":
            tool_calls = message.tool_calls
            body.update([
                ("content",content),
                ("tool_calls",str(tool_calls))
            ])
        elif type == "human":
            body.update([
                ("content",content)
            ])

        if tools is not None:
            for index, tool in enumerate(tools):
                function = tool.get("function")
                if function is not None:
                    attributes.update([
                        (f"gen_ai.request.function.{index}.name", function.get("name")),
                        (f"gen_ai.request.function.{index}.description", function.get("description")),
                        (f"gen_ai.request.function.{index}.parameters", str(function.get("parameters"))),
                    ])

        return Event(
            name=f"gen_ai.{type}.message",
            attributes=attributes,
            body=body if body else None,
        )

def chat_generation_to_event(chat_generation, index, prefix):
    if should_collect_content() and chat_generation.message:
        content = get_property_value(chat_generation.message, "content")
        if content is not None:
            attributes = {
                GenAI.GEN_AI_SYSTEM: "langchain"
            }

            finish_reason = None
            generation_info = chat_generation.generation_info
            if generation_info is not None:
                finish_reason = generation_info.get("finish_reason")

            message = {
                "content": content,
                "type": chat_generation.type
            }
            body = {
                "index": index,
                "finish_reason": finish_reason or "error",
                "message": message
            }

            tool_calls = chat_generation.message.additional_kwargs.get("tool_calls")
            if tool_calls is not None:
                attributes.update(
                    chat_generation_tool_calls_attributes(tool_calls, prefix)
                )

            return Event(
                name="gen_ai.choice",
                attributes=attributes,
                body=body,
            )

def query_to_event(query):
    if should_collect_content() and query is not None:
        body = {}
        body["content"] = query
        attributes = {
            GenAI.GEN_AI_SYSTEM: "langchain",
            GenAI.GEN_AI_PROMPT: query
        }

        return Event(
            name="gen_ai.content.prompt",
            attributes=attributes,
            body=body if body else None,
        )

def document_to_event(document: Document, index: int) -> Event:
    # Build the semantic attributes you want on the event:
    attrs = {
        "retrieval.document.rank": index,
        "retrieval.document.content": document.page_content,
        **{f"doc.meta.{k}": v for k, v in document.metadata.items()},
    }
    return Event(
        name="langchain.retriever.document",
        timestamp=time_ns(),
        attributes=attrs,
    )

def input_to_event(input):
    if should_collect_content() and input is not None:
        body = {
            "content" : input,
            "role": "tool",
        }
        attributes = {
            GenAI.GEN_AI_SYSTEM: "langchain",
        }

        return Event(
            name="gen_ai.tool.message",
            attributes=attributes,
            body=body if body else None,
        )

def output_to_event(output):
    if should_collect_content() and output is not None:
        body = {
            "content":output.content,
            "id":output.tool_call_id,
            "role":"tool",
        }
        attributes = {
            GenAI.GEN_AI_SYSTEM: "langchain",
        }

        return Event(
            name="gen_ai.tool.message",
            attributes=attributes,
            body=body if body else None,
        )

def chat_generation_tool_calls_attributes(tool_calls, prefix):
    attributes = {}
    for idx, tool_call in enumerate(tool_calls):
        tool_call_prefix = f"{prefix}.tool_calls.{idx}"
        attributes[f"{tool_call_prefix}.id"] = tool_call.get("id")
        attributes[f"{tool_call_prefix}.name"] = tool_call.get("function").get("name")
        attributes[f"{tool_call_prefix}.arguments"] = tool_call.get("function").get("arguments")
    return attributes

class CallbackFilteredJSONEncoder(json.JSONEncoder):
    """
    Example JSON encoder that removes a "callbacks" field if present,
    and tries to handle typical dataclass or pydantic objects.
    """

    def default(self, o):
        if isinstance(o, dict):
            if "callbacks" in o:
                # Remove any "callbacks" from the dict to avoid massive recursion
                new_o = dict(o)
                del new_o["callbacks"]
                return new_o
            return dict(o)

        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if isinstance(o, BaseModel):
            # For pydantic v2, "model_dump_json" is sometimes available
            if hasattr(o, "model_dump_json"):
                return o.model_dump_json()
            return o.dict()

        if isinstance(o, datetime.datetime):
            return o.isoformat()

        # fallback:
        return str(o)
