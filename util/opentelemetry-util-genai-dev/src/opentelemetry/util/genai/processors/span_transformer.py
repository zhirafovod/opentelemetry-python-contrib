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

from typing import Any, Dict, Optional

from opentelemetry.sdk.trace import ReadableSpan

from .traceloop_span_generator import TraceloopSpanGenerator
from ..types import LLMInvocation


def _apply_attribute_transformations(
    base: Dict[str, Any], transformations: Optional[Dict[str, Any]]
) -> Dict[str, Any]:  # pragma: no cover - trivial helpers
    if not transformations:
        return base
    # Order: remove -> rename -> add (so add always wins)
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
    original_name: str, name_transformations: Optional[Dict[str, str]]
) -> Optional[str]:  # pragma: no cover - simple matching
    if not name_transformations:
        return None
    import fnmatch

    for pattern, new_name in name_transformations.items():
        try:
            if fnmatch.fnmatch(original_name, pattern):
                return new_name
        except Exception:  # defensive
            continue
    return None


def transform_existing_span_to_telemetry(
    existing_span: ReadableSpan,
    attribute_transformations: Optional[Dict[str, Any]] = None,
    name_transformations: Optional[Dict[str, str]] = None,
    traceloop_attributes: Optional[Dict[str, Any]] = None,
    generator: Optional[TraceloopSpanGenerator] = None,
) -> LLMInvocation:
    """Create a synthetic LLMInvocation span from an ended (or ending) span.

    Returns the synthetic ``LLMInvocation`` used purely as a carrier for the new span.
    """
    base_attrs: Dict[str, Any] = (
        dict(existing_span.attributes) if existing_span.attributes else {}
    )

    # Apply transformations
    base_attrs = _apply_attribute_transformations(
        base_attrs, attribute_transformations
    )
    if traceloop_attributes:
        base_attrs.update(traceloop_attributes)

    # Span name rewrite (store so generator can use & remove later)
    new_name = _derive_new_name(existing_span.name, name_transformations)
    if new_name:
        base_attrs["_traceloop_new_name"] = new_name

    # Determine request_model (best-effort, fallback to unknown)
    request_model = (
        base_attrs.get("gen_ai.request.model")
        or base_attrs.get("llm.request.model")
        or base_attrs.get("ai.model.name")
        or "unknown"
    )

    invocation = LLMInvocation(
        request_model=str(request_model),
        attributes=base_attrs,
        messages=[],  # empty; original content not reconstructed here
    )

    if generator is None:
        generator = TraceloopSpanGenerator(capture_content=True)
    generator.start(invocation)
    if existing_span.end_time is not None:
        generator.finish(invocation)
    return invocation
