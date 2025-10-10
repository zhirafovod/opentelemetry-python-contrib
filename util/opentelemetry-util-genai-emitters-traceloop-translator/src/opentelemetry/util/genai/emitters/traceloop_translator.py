from __future__ import annotations

import os
import re
import json
from typing import Dict, Any

from opentelemetry.util.genai.emitters.spec import EmitterFactoryContext, EmitterSpec
from opentelemetry.util.genai.interfaces import EmitterMeta
from opentelemetry.util.genai.types import LLMInvocation

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
strip_legacy = bool(os.getenv(_STRIP_FLAG, "true"))

# Content capture opt-in: default off for sensitive prompt/template + messages
_CONTENT_CAPTURE_FLAG = "OTEL_GENAI_CONTENT_CAPTURE"
_content_capture_enabled = os.getenv(_CONTENT_CAPTURE_FLAG, "1") not in ("0", "false", "False")

# Message normalization (wrap blobs into structured schema) toggle & limits
_INPUT_MAX = 100  # deprecated local constant (kept for backward compat if referenced)
_OUTPUT_MAX = 100
_MSG_CONTENT_MAX = 16000

# Correlation->conversation mapping strictness toggle
_MAP_CORRELATION_FLAG = "OTEL_GENAI_MAP_CORRELATION_TO_CONVERSATION"
_map_correlation = os.getenv(_MAP_CORRELATION_FLAG, "true") not in ("0", "false", "False")

# Maximum size for prompt template before truncation
_PROMPT_TEMPLATE_MAX = 4096

# Regex to validate conversation id (avoid mapping arbitrary large/high-cardinality values)
_CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")

from .content_normalizer import normalize_traceloop_content, maybe_truncate_template


class TraceloopTranslatorEmitter(EmitterMeta):
    role = "span"
    name = "traceloop_translator"

    def handles(self, obj: object) -> bool:
        return isinstance(obj, LLMInvocation)

    def on_start(self, invocation: LLMInvocation) -> None:
        attrs = getattr(invocation, "attributes", None)
        if not attrs:
            return
        for key in list(attrs.keys()):
            value = attrs.get(key)
            is_prefixed = key.startswith("traceloop.")
            raw_key = key
            if not (is_prefixed or raw_key in _TRACELOOP_TO_SEMCONV or raw_key in _CONTENT_MAPPING):
                continue

            # Content mapping (entity input/output)
            if raw_key in _CONTENT_MAPPING:
                if _content_capture_enabled:
                    target = _CONTENT_MAPPING[raw_key]
                    direction = ENTITY_CONTENT.get(raw_key)
                    if direction is None:
                        continue
                    try:
                        norm = normalize_traceloop_content(value, direction)
                        attrs.setdefault(target, json.dumps(norm))
                    except Exception:
                        attrs.setdefault(target, value)
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
                        if mapped == "gen_ai.prompt.template" and not _content_capture_enabled:
                            # Skip sensitive template unless opted-in
                            mapped = None
                        elif mapped == "gen_ai.prompt.template" and isinstance(value, str):
                            value = maybe_truncate_template(value)
                        attrs.setdefault(mapped, value)

            # Legacy strip: keep traceloop.span.kind even when stripping (diagnostic)
            if strip_legacy and is_prefixed and raw_key != "traceloop.span.kind":
                # Remove only if we have mapped something or mapping not needed
                attrs.pop(key, None)

        # Heuristic: infer operation name for tool/workflow invocations if absent
        if attrs.get("gen_ai.operation.name") is None:
            span_kind = attrs.get("traceloop.span.kind")
            if span_kind == "tool":
                attrs.setdefault("gen_ai.operation.name", "execute_tool")
            elif span_kind in ("workflow", "agent", "chain"):
                attrs.setdefault("gen_ai.operation.name", "invoke_agent")

    def on_end(self, invocation: LLMInvocation) -> None:  # pragma: no cover
        return

    def on_error(self, error, invocation: LLMInvocation) -> None:  # pragma: no cover
        return


def traceloop_translator_emitters() -> list[EmitterSpec]:
    def _factory(ctx: EmitterFactoryContext) -> TraceloopTranslatorEmitter:
        return TraceloopTranslatorEmitter()

    return [
        EmitterSpec(
            name="TraceloopTranslator",
            category="span",
            factory=_factory,
            mode="prepend",
        )
    ]


__all__ = ["TraceloopTranslatorEmitter", "traceloop_translator_emitters"]
