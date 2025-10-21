from __future__ import annotations

import os
import re
import json
from typing import Dict, Any

from opentelemetry.util.genai.emitters.spec import EmitterFactoryContext, EmitterSpec
from opentelemetry.util.genai.interfaces import EmitterMeta
from opentelemetry.util.genai.types import LLMInvocation, Error

# https://github.com/traceloop/openllmetry/blob/main/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py
# TODO: rename appropriately 
_TRACELOOP_TO_SEMCONV: Dict[str, str] = {
    # Workflow / entity hierarchy (proposed extension namespace)
    "traceloop.workflow.name": "gen_ai.workflow.name",
    "traceloop.entity.name": "gen_ai.agent.name",
    "traceloop.entity.path": "gen_ai.workflow.path",
    "traceloop.association.properties": "gen_ai.association.properties",
    "traceloop.entity.version": "gen_ai.workflow.version",
    # Prompt registry / template metadata (proposed extensions)
    "traceloop.prompt.managed": "gen_ai.prompt.managed",
    "traceloop.prompt.key": "gen_ai.prompt.key",
    "traceloop.prompt.version": "gen_ai.prompt.version",
    "traceloop.prompt.version_name": "gen_ai.prompt.version_name",
    "traceloop.prompt.version_hash": "gen_ai.prompt.version_hash",
    "traceloop.prompt.template": "gen_ai.prompt.template",
    "traceloop.prompt.template_variables": "gen_ai.prompt.template_variables",
    # Correlation -> conversation (conditional)
    "traceloop.correlation.id": "gen_ai.conversation.id",
}

_CONTENT_MAPPING = {
    "traceloop.entity.input": "gen_ai.input.messages",
    "traceloop.entity.output": "gen_ai.output.messages",
}

# Explicit direction map
ENTITY_CONTENT = {
    "traceloop.entity.input": "input",
    "traceloop.entity.output": "output",
}

_STRIP_FLAG = "OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY"
# NOTE: Previous implementation evaluated strip flag at import time and never applied.
# We now evaluate dynamically for each translation to honor late env changes and actually strip.
def _is_strip_legacy_enabled() -> bool:
    return os.getenv(_STRIP_FLAG, "true") not in ("0", "false", "False")

# Content capture opt-in: default off for sensitive prompt/template + messages
_CONTENT_CAPTURE_FLAG = "OTEL_GENAI_CONTENT_CAPTURE"

def _is_content_capture_enabled() -> bool:
    """Return True if content capture is enabled (evaluated at call time).

    Previously this was computed at import time, which meant tests (or user code)
    that modified the environment variable *after* importing this module could not
    disable content capture. Making it a function keeps backward compatibility for
    callers while honoring late env changes.
    """
    val = os.getenv(_CONTENT_CAPTURE_FLAG, "1")
    return val not in ("0", "false", "False")

# Correlation->conversation mapping strictness toggle
_MAP_CORRELATION_FLAG = "OTEL_GENAI_MAP_CORRELATION_TO_CONVERSATION"
_map_correlation = os.getenv(_MAP_CORRELATION_FLAG, "true") not in ("0", "false", "False")

# Regex to validate conversation id (avoid mapping arbitrary large/high-cardinality values)
_CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")

from .content_normalizer import normalize_traceloop_content, maybe_truncate_template


class TraceloopTranslatorEmitter(EmitterMeta):
    role = "span"
    name = "traceloop_translator"

    def handles(self, obj: object) -> bool:
        return isinstance(obj, LLMInvocation)

    # EmitterProtocol expects on_start/on_end/on_error with a generic object param; keep
    # internal specificity via isinstance checks.
    def on_start(self, obj: Any) -> None:
        if isinstance(obj, LLMInvocation):
            self._translate_attributes(obj)

    def _translate_attributes(self, invocation: LLMInvocation) -> None:
        """Translate traceloop.* attributes to gen_ai.* equivalents."""
        attrs = getattr(invocation, "attributes", None)
        if not attrs:
            return
        legacy_keys_to_strip: list[str] = []  # track keys we successfully mapped; only strip those
        # Capture original presence to decide on fallback later
        had_traceloop_input = "traceloop.entity.input" in attrs
        had_traceloop_output = "traceloop.entity.output" in attrs
        for key in list(attrs.keys()):
            value = attrs.get(key)
            is_prefixed = key.startswith("traceloop.")
            raw_key = key
            if not (is_prefixed or raw_key in _TRACELOOP_TO_SEMCONV or raw_key in _CONTENT_MAPPING):
                continue

            # Content mapping (entity input/output)
            if raw_key in _CONTENT_MAPPING:
                if _is_content_capture_enabled():
                    target = _CONTENT_MAPPING[raw_key]
                    direction = ENTITY_CONTENT.get(raw_key)
                    if direction is None:
                        continue
                    try:
                        norm = normalize_traceloop_content(value, direction)
                        attrs.setdefault(target, json.dumps(norm))
                        legacy_keys_to_strip.append(raw_key)
                    except Exception:
                        attrs.setdefault(target, value)
                        legacy_keys_to_strip.append(raw_key)
            else:
                mapped = _TRACELOOP_TO_SEMCONV.get(raw_key)
                if mapped:
                    # Conditional conversation id mapping
                    if mapped == "gen_ai.conversation.id":
                        if not _map_correlation:
                            mapped = None
                        elif not isinstance(value, str) or not _CONVERSATION_ID_PATTERN.match(value):
                            mapped = None
                    if mapped:
                        if mapped == "gen_ai.prompt.template" and not _is_content_capture_enabled():
                            # Skip sensitive template unless opted-in
                            mapped = None
                        elif mapped == "gen_ai.prompt.template" and isinstance(value, str):
                            value = maybe_truncate_template(value)
                        if mapped:
                            attrs.setdefault(mapped, value)
                            legacy_keys_to_strip.append(raw_key)
                            # Also push onto span immediately (order-independent visibility)
                            span = getattr(invocation, "span", None)
                            if span is not None and mapped not in getattr(span, "attributes", {}):
                                try:
                                    span.set_attribute(mapped, value)
                                except Exception:  # pragma: no cover
                                    pass


        # Heuristic: infer operation name for tool/workflow invocations if absent
        if attrs.get("gen_ai.operation.name") is None:
            span_kind = attrs.get("traceloop.span.kind")
            if span_kind == "tool":
                attrs.setdefault("gen_ai.operation.name", "execute_tool")
            elif span_kind in ("workflow", "agent", "chain"):
                attrs.setdefault("gen_ai.operation.name", "invoke_agent")

        # Fallback: if caller provided invocation.input_messages but no traceloop.entity.input
        # and translator didn't already set gen_ai.input.messages, derive it now.
        if _is_content_capture_enabled() and not had_traceloop_input and "gen_ai.input.messages" not in attrs:
            input_messages = getattr(invocation, "input_messages", None)
            if input_messages:
                try:
                    norm = normalize_traceloop_content(input_messages, "input")
                    serialized = json.dumps(norm)
                    attrs.setdefault("gen_ai.input.messages", serialized)
                    span = getattr(invocation, "span", None)
                    if span is not None and "gen_ai.input.messages" not in getattr(span, "attributes", {}):
                        try:
                            span.set_attribute("gen_ai.input.messages", serialized)
                        except Exception:  # pragma: no cover
                            pass
                except Exception:
                    # Best effort; store raw repr
                    fallback = json.dumps([getattr(m, "__dict__", str(m)) for m in input_messages])
                    attrs.setdefault("gen_ai.input.messages", fallback)
                    span = getattr(invocation, "span", None)
                    if span is not None and "gen_ai.input.messages" not in getattr(span, "attributes", {}):
                        try:
                            span.set_attribute("gen_ai.input.messages", fallback)
                        except Exception:  # pragma: no cover
                            pass

        # Fallback for output: only after model produced output_messages
        if _is_content_capture_enabled() and not had_traceloop_output and "gen_ai.output.messages" not in attrs:
            output_messages = getattr(invocation, "output_messages", None)
            if output_messages:
                try:
                    norm = normalize_traceloop_content(output_messages, "output")
                    serialized = json.dumps(norm)
                    attrs.setdefault("gen_ai.output.messages", serialized)
                    span = getattr(invocation, "span", None)
                    if span is not None and "gen_ai.output.messages" not in getattr(span, "attributes", {}):
                        try:
                            span.set_attribute("gen_ai.output.messages", serialized)
                        except Exception:  # pragma: no cover
                            pass
                except Exception:
                    fallback = json.dumps([getattr(m, "__dict__", str(m)) for m in output_messages])
                    attrs.setdefault("gen_ai.output.messages", fallback)
                    span = getattr(invocation, "span", None)
                    if span is not None and "gen_ai.output.messages" not in getattr(span, "attributes", {}):
                        try:
                            span.set_attribute("gen_ai.output.messages", fallback)
                        except Exception:  # pragma: no cover
                            pass

        # Strip legacy keys if enabled (only those we mapped to avoid data loss)
        if _is_strip_legacy_enabled():
            for lk in legacy_keys_to_strip:
                attrs.pop(lk, None)

    def on_end(self, obj: Any) -> None:
        """Also translate attributes at end, in case new ones were added after start."""
        if isinstance(obj, LLMInvocation):
            self._translate_attributes(obj)

    def on_error(self, error: Error, obj: Any) -> None:  # pragma: no cover
        # Translator does not emit error-specific attributes currently.
        return None


def traceloop_translator_emitters() -> list[EmitterSpec]:
    def _factory(ctx: EmitterFactoryContext) -> TraceloopTranslatorEmitter:
        return TraceloopTranslatorEmitter()

    return [
        EmitterSpec(
            name="TraceloopTranslator",
            category="span",
            factory=_factory,
            mode="append"
        )
    ]


__all__ = ["TraceloopTranslatorEmitter", "traceloop_translator_emitters"]