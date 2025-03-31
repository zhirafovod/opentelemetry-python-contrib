import json
import logging
import os
import traceback
import dataclasses
import datetime

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
