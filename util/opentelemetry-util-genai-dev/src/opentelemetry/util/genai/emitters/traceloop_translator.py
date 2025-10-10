"""Traceloop -> GenAI Semantic Convention translation emitter.

This emitter runs early in the span category chain and *mutates* the invocation
attributes in-place, translating a subset of legacy ``traceloop.*`` attributes
into semantic convention (``gen_ai.*``) or structured invocation fields so that
subsequent emitters (e.g. the primary semconv span emitter) naturally record
the standardized form.

It intentionally does NOT emit its own span. It simply rewrites data.

If both the original TraceloopCompatEmitter and this translator are enabled,
the pipeline order should be: translator -> semconv span -> traceloop compat span.
The translator only promotes data; it does not delete the legacy attributes by
default (configurable via env var in future if needed).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .spec import EmitterFactoryContext, EmitterSpec
from ..interfaces import EmitterMeta
from ..types import LLMInvocation

# Mapping from traceloop attribute key (without prefix) to either:
# - a gen_ai semantic convention attribute key
# - a special handler function name (prefixed with "@") for structured placement.
_TRACELOOP_TO_SEMCONV: Dict[str, str] = {
    "workflow.name": "gen_ai.workflow.name",   # custom (not in spec yet)
    "entity.name": "gen_ai.agent.name",        # approximate: treat entity as agent name
    "entity.path": "gen_ai.workflow.path",     # custom placeholder (maps from traceloop.entity.path or entity.path)
    # callback metadata (custom placeholders until standardized)
    "callback.name": "gen_ai.callback.name",
    "callback.id": "gen_ai.callback.id",
    # span.kind is redundant (SpanKind already encodes); omitted
}

# Input/output content attributes – when present we map them to message serialization
# helpers by copying into invocation.attributes under semconv-like provisional keys.
_CONTENT_MAPPING = {
    "entity.input": "gen_ai.input.messages",
    "entity.output": "gen_ai.output.messages",
}


_STRIP_FLAG = "OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY"


class TraceloopTranslatorEmitter(EmitterMeta):
    role = "span"
    name = "traceloop_translator"

    def __init__(self) -> None:  # no tracer needed – we do not create spans
        pass

    def handles(self, obj: object) -> bool:  # only care about LLM invocations
        return isinstance(obj, LLMInvocation)

    def on_start(self, invocation: LLMInvocation) -> None:  # mutate attributes
        attrs = getattr(invocation, "attributes", None)
        if not attrs:
            return
        strip_legacy = bool(os.getenv(_STRIP_FLAG))
        for key in list(attrs.keys()):
            value = attrs.get(key)
            is_prefixed = False
            if key.startswith("traceloop."):
                raw_key = key[len("traceloop.") :]
                is_prefixed = True
            elif key in _TRACELOOP_TO_SEMCONV or key in _CONTENT_MAPPING:
                raw_key = key
            else:
                continue

            # Content mapping
            if raw_key in _CONTENT_MAPPING:
                target = _CONTENT_MAPPING[raw_key]
                attrs.setdefault(target, value)
            else:
                mapped = _TRACELOOP_TO_SEMCONV.get(raw_key)
                if mapped:
                    attrs.setdefault(mapped, value)
                if raw_key == "callback.name" and isinstance(value, str):
                    attrs.setdefault("gen_ai.operation.source", value)

            # Optionally remove legacy prefixed variant after promotion
            if strip_legacy and is_prefixed:
                try:
                    attrs.pop(key, None)
                except Exception:  # pragma: no cover - defensive
                    pass

    # No-op finish & error hooks – translation is only needed once.
    def on_end(self, invocation: LLMInvocation) -> None:  # pragma: no cover - trivial
        return

    def on_error(self, error, invocation: LLMInvocation) -> None:  # pragma: no cover - trivial
        return


def traceloop_translator_emitters() -> list[EmitterSpec]:
    def _factory(ctx: EmitterFactoryContext) -> TraceloopTranslatorEmitter:
        return TraceloopTranslatorEmitter()

    return [
        EmitterSpec(
            name="TraceloopTranslator",
            category="span",
            factory=_factory,
            mode="prepend",  # ensure earliest so promotion happens before SemanticConvSpan is added
            after=(),
        )
    ]


__all__ = [
    "TraceloopTranslatorEmitter",
    "traceloop_translator_emitters",
]
