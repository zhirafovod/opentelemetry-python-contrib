# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Span

from opentelemetry.util.genai.types import LLMInvocation
from opentelemetry.util.genai.handler import (
    get_telemetry_handler,
    TelemetryHandler,
)

_ENV_RULES = "OTEL_GENAI_SPAN_TRANSFORM_RULES"


@dataclass
class TransformationRule:
    """Represents a single conditional transformation rule.

    Fields map closely to the JSON structure accepted via the environment
    variable. All fields are optional; empty rule never matches.
    """

    match_name: Optional[str] = None  # glob pattern (e.g. "chat *")
    match_scope: Optional[str] = None  # regex or substring (case-insensitive)
    match_attributes: Dict[str, Optional[str]] = field(default_factory=dict)

    attribute_transformations: Dict[str, Any] = field(default_factory=dict)
    name_transformations: Dict[str, str] = field(default_factory=dict)
    traceloop_attributes: Dict[str, Any] = field(default_factory=dict)

    def matches(self, span: ReadableSpan) -> bool:  # pragma: no cover - simple logic
        if self.match_name:
            if not fnmatch.fnmatch(span.name, self.match_name):
                return False
        if self.match_scope:
            scope = getattr(span, "instrumentation_scope", None)
            scope_name = getattr(scope, "name", "") if scope else ""
            pattern = self.match_scope
            # Accept either regex (contains meta chars) or simple substring
            try:
                if any(ch in pattern for ch in ".^$|()[]+?\\"):
                    if not re.search(pattern, scope_name, re.IGNORECASE):
                        return False
                else:
                    if pattern.lower() not in scope_name.lower():
                        return False
            except re.error:
                # Bad regex – treat as non-match but log once
                logging.warning("Invalid regex in match_scope: %s", pattern)
                return False
        if self.match_attributes:
            for k, expected in self.match_attributes.items():
                if k not in span.attributes:
                    return False
                if expected is not None and str(span.attributes.get(k)) != str(expected):
                    return False
        return True


def _load_rules_from_env() -> List[TransformationRule]:
    raw = os.getenv(_ENV_RULES)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        rules_spec = data.get("rules") if isinstance(data, dict) else None
        if not isinstance(rules_spec, list):
            logging.warning("%s must contain a 'rules' list", _ENV_RULES)
            return []
        rules: List[TransformationRule] = []
        for r in rules_spec:
            if not isinstance(r, dict):
                continue
            match = r.get("match", {}) if isinstance(r.get("match"), dict) else {}
            rules.append(
                TransformationRule(
                    match_name=match.get("name"),
                    match_scope=match.get("scope"),
                    match_attributes=match.get("attributes", {}) or {},
                    attribute_transformations=r.get("attribute_transformations", {}) or {},
                    name_transformations=r.get("name_transformations", {}) or {},
                    traceloop_attributes=r.get("traceloop_attributes", {}) or {},
                )
            )
        return rules
    except Exception as exc:  # broad: we never want to break app startup
        logging.warning("Failed to parse %s: %s", _ENV_RULES, exc)
        return []


class TraceloopSpanProcessor(SpanProcessor):
    """
    A span processor that automatically applies transformation rules to spans.

    This processor can be added to your TracerProvider to automatically transform
    all spans according to your transformation rules.
    """

    def __init__(
        self,
        attribute_transformations: Optional[Dict[str, Any]] = None,
        name_transformations: Optional[Dict[str, str]] = None,
        traceloop_attributes: Optional[Dict[str, Any]] = None,
        span_filter: Optional[Callable[[ReadableSpan], bool]] = None,
        rules: Optional[List[TransformationRule]] = None,
        load_env_rules: bool = True,
    telemetry_handler: Optional[TelemetryHandler] = None,
        # Legacy synthetic span duplication removed – always emit via handler
    ):
        """
        Initialize the Traceloop span processor.

        Args:
            attribute_transformations: Rules for transforming span attributes
            name_transformations: Rules for transforming span names
            traceloop_attributes: Additional Traceloop-specific attributes to add
            span_filter: Optional filter function to determine which spans to transform
        """
        self.attribute_transformations = attribute_transformations or {}
        self.name_transformations = name_transformations or {}
        self.traceloop_attributes = traceloop_attributes or {}
        self.span_filter = span_filter or self._default_span_filter
        # Load rule set (env + explicit). Explicit rules first for precedence.
        env_rules = _load_rules_from_env() if load_env_rules else []
        self.rules: List[TransformationRule] = list(rules or []) + env_rules
        self.telemetry_handler = telemetry_handler
        if self.rules:
            logging.getLogger(__name__).debug(
                "TraceloopSpanProcessor loaded %d transformation rules (explicit=%d env=%d)",
                len(self.rules), len(rules or []), len(env_rules)
            )

    def _default_span_filter(self, span: ReadableSpan) -> bool:
        """Default filter: Transform spans that look like LLM/AI calls.

        Previously this required both a name and at least one attribute. Some
        tests (and real-world scenarios) emit spans with meaningful names like
        "chat gpt-4" before any model/provider attributes are recorded. We now
        allow name-only detection; attributes merely increase confidence.
        """
        if not span.name:
            return False

        # Check for common LLM/AI span indicators
        llm_indicators = [
            "chat",
            "completion",
            "llm",
            "ai",
            "gpt",
            "claude",
            "gemini",
            "openai",
            "anthropic",
            "cohere",
            "huggingface",
        ]

        span_name_lower = span.name.lower()
        for indicator in llm_indicators:
            if indicator in span_name_lower:
                return True

        # Check attributes for AI/LLM markers (if any attributes present)
        if span.attributes:
            for attr_key in span.attributes.keys():
                attr_key_lower = str(attr_key).lower()
                if any(
                    marker in attr_key_lower
                    for marker in ["llm", "ai", "gen_ai", "model"]
                ):
                    return True
        return False

    def on_start(
        self, span: Span, parent_context: Optional[Context] = None
    ) -> None:
        """Called when a span is started."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span is ended.
        """
        try:
            # Check if this span should be transformed (cheap heuristic first)
            if not self.span_filter(span):
                return
            # Skip spans we already produced (recursion guard)
            if span.attributes and "_traceloop_processed" in span.attributes:
                return

            # Determine which transformation set to use
            applied_rule: Optional[TransformationRule] = None
            for rule in self.rules:
                try:
                    if rule.matches(span):
                        applied_rule = rule
                        break
                except Exception as match_err:  # pragma: no cover - defensive
                    logging.warning("Rule match error ignored: %s", match_err)

            sentinel = {"_traceloop_processed": True}
            # Decide which transformation config to apply
            if applied_rule is not None:
                attr_tx = applied_rule.attribute_transformations
                name_tx = applied_rule.name_transformations
                extra_tl_attrs = {**applied_rule.traceloop_attributes, **sentinel}
            else:
                attr_tx = self.attribute_transformations
                name_tx = self.name_transformations
                extra_tl_attrs = {**self.traceloop_attributes, **sentinel}

            # Always emit via TelemetryHandler
            invocation = self._build_invocation(
                span,
                attribute_transformations=attr_tx,
                name_transformations=name_tx,
                traceloop_attributes=extra_tl_attrs,
            )
            invocation.attributes.setdefault("_traceloop_processed", True)
            handler = self.telemetry_handler or get_telemetry_handler()
            try:
                handler.start_llm(invocation)
                handler.stop_llm(invocation)
            except Exception as emit_err:  # pragma: no cover - defensive
                logging.getLogger(__name__).warning(
                    "Telemetry handler emission failed: %s", emit_err
                )

        except Exception as e:
            # Don't let transformation errors break the original span processing
            import logging

            logging.warning(
                f"TraceloopSpanProcessor failed to transform span: {e}"
            )

    def shutdown(self) -> None:
        """Called when the tracer provider is shutdown."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_attribute_transformations(
        self, base: Dict[str, Any], transformations: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not transformations:
            return base
        remove_keys = transformations.get("remove") or []
        for k in remove_keys:
            base.pop(k, None)
        rename_map = transformations.get("rename") or {}
        for old, new in rename_map.items():
            if old in base:
                base[new] = base.pop(old)
        add_map = transformations.get("add") or {}
        for k, v in add_map.items():
            base[k] = v
        return base

    def _derive_new_name(
        self, original_name: str, name_transformations: Optional[Dict[str, str]]
    ) -> Optional[str]:
        if not name_transformations:
            return None
        import fnmatch

        for pattern, new_name in name_transformations.items():
            try:
                if fnmatch.fnmatch(original_name, pattern):
                    return new_name
            except Exception:
                continue
        return None

    def _build_invocation(
        self,
        existing_span: ReadableSpan,
        *,
        attribute_transformations: Optional[Dict[str, Any]] = None,
        name_transformations: Optional[Dict[str, str]] = None,
        traceloop_attributes: Optional[Dict[str, Any]] = None,
    ) -> LLMInvocation:
        base_attrs: Dict[str, Any] = (
            dict(existing_span.attributes) if existing_span.attributes else {}
        )
        base_attrs = self._apply_attribute_transformations(
            base_attrs, attribute_transformations
        )
        if traceloop_attributes:
            base_attrs.update(traceloop_attributes)
        new_name = self._derive_new_name(
            existing_span.name, name_transformations
        )
        if new_name:
            # Provide override for SpanEmitter (we extended it to honor this)
            base_attrs.setdefault("gen_ai.override.span_name", new_name)
        request_model = (
            base_attrs.get("gen_ai.request.model")
            or base_attrs.get("llm.request.model")
            or base_attrs.get("ai.model.name")
            or "unknown"
        )
        invocation = LLMInvocation(
            request_model=str(request_model),
            attributes=base_attrs,
            messages=[],
        )
        # Mark operation heuristically from original span name
        lowered = existing_span.name.lower()
        if lowered.startswith("embed"):
            invocation.operation = "embedding"  # type: ignore[attr-defined]
        elif lowered.startswith("chat"):
            invocation.operation = "chat"  # type: ignore[attr-defined]
        return invocation