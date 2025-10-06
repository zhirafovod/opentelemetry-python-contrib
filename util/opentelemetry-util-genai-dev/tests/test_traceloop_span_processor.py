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

import json
from typing import List

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
try:  # Prefer direct export if available (older versions)
    from opentelemetry.sdk.trace.export import InMemorySpanExporter  # type: ignore
except ImportError:  # pragma: no cover - fallback path for newer versions
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # type: ignore
        InMemorySpanExporter,
    )
from opentelemetry import trace

from opentelemetry.util.genai.processors.traceloop_span_processor import (
    TraceloopSpanProcessor,
    TransformationRule,
)
from opentelemetry.util.genai.processors.traceloop_span_generator import (
    TraceloopSpanGenerator,
)


@pytest.fixture
def tracer_provider():
    # Provide a fresh provider per test to avoid cross-test processor leakage
    # (we intentionally DO NOT set it globally to keep isolation).
    return SDKTracerProvider()


@pytest.fixture
def in_memory_exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer(tracer_provider, in_memory_exporter):
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))
    return tracer_provider.get_tracer(__name__)


def _find_transformed_spans(spans: List[ReadableSpan]):
    # Heuristic: transformed spans have the sentinel attribute
    return [s for s in spans if s.attributes.get("_traceloop_processed")]


def test_fallback_single_rule(tracer_provider, tracer, in_memory_exporter):
    # Rename an existing attribute instead of adding a new one.
    processor = TraceloopSpanProcessor(
        attribute_transformations={"rename": {"llm.provider": "service.name"}},
        name_transformations={"chat *": "genai.chat"},
        generator=TraceloopSpanGenerator(tracer=tracer),
    )
    tracer_provider.add_span_processor(processor)

    with tracer.start_as_current_span("chat gpt-4") as span:
        span.set_attribute("llm.provider", "openai")

    spans = in_memory_exporter.get_finished_spans()
    transformed = _find_transformed_spans(spans)
    # Original + transformed
    assert len(transformed) == 1
    t = transformed[0]
    assert t.name == "genai.chat"
    # Value preserved from original attribute
    assert t.attributes["service.name"] == "openai"
    assert t.attributes["_traceloop_processed"] is True


def test_rule_precedence(tracer_provider, tracer, in_memory_exporter):
    rules = [
        TransformationRule(
            match_name="chat *",
            attribute_transformations={"rename": {"marker": "first.marker"}},
            name_transformations={"chat *": "first.chat"},
        ),
        TransformationRule(
            match_name="chat gpt-*",
            attribute_transformations={"rename": {"marker": "second.marker"}},
            name_transformations={"chat gpt-*": "second.chat"},
        ),
    ]
    processor = TraceloopSpanProcessor(
        rules=rules,
        load_env_rules=False,
        generator=TraceloopSpanGenerator(tracer=tracer),
    )
    tracer_provider.add_span_processor(processor)

    with tracer.start_as_current_span("chat gpt-4") as span:
        span.set_attribute("marker", True)

    spans = in_memory_exporter.get_finished_spans()
    transformed = _find_transformed_spans(spans)
    assert transformed, "Expected transformed span"
    # First rule wins
    assert transformed[0].name == "first.chat"
    assert transformed[0].attributes.get("first.marker") is True
    assert "second.marker" not in transformed[0].attributes


def test_env_rule_overrides(tracer_provider, tracer, in_memory_exporter, monkeypatch):
    env_spec = {
        "rules": [
            {
                "match": {"name": "chat *"},
                "attribute_transformations": {"rename": {"marker": "env.marker"}},
                "name_transformations": {"chat *": "env.chat"},
            }
        ]
    }
    monkeypatch.setenv("OTEL_GENAI_SPAN_TRANSFORM_RULES", json.dumps(env_spec))

    processor = TraceloopSpanProcessor(
        attribute_transformations={"rename": {"marker": "fallback.marker"}},
        name_transformations={"chat *": "fallback.chat"},
        generator=TraceloopSpanGenerator(tracer=tracer),
    )
    tracer_provider.add_span_processor(processor)

    with tracer.start_as_current_span("chat model") as span:
        span.set_attribute("marker", 123)

    spans = in_memory_exporter.get_finished_spans()
    transformed = _find_transformed_spans(spans)
    assert transformed
    span = transformed[0]
    assert span.name == "env.chat"  # env rule used
    assert span.attributes.get("env.marker") == 123
    # Fallback rename should not happen because env rule applied instead
    assert "fallback.marker" not in span.attributes


def test_recursion_guard(tracer_provider, tracer, in_memory_exporter):
    # Span already marked as processed should not be processed again
    processor = TraceloopSpanProcessor(
        attribute_transformations={"rename": {"foo": "service.name"}},
        generator=TraceloopSpanGenerator(tracer=tracer),
    )
    tracer_provider.add_span_processor(processor)

    # Manually create a span that already has sentinel
    with tracer.start_as_current_span("chat something") as span:
        span.set_attribute("_traceloop_processed", True)

    spans = in_memory_exporter.get_finished_spans()
    transformed = _find_transformed_spans(spans)
    # Only the original (pre-marked) should exist, no new duplicate
    assert len(transformed) == 1
    assert transformed[0].name == "chat something"


def test_non_matching_span_not_transformed(tracer_provider, tracer, in_memory_exporter):
    processor = TraceloopSpanProcessor(
        attribute_transformations={"rename": {"some.attr": "unused"}},
        generator=TraceloopSpanGenerator(tracer=tracer),
    )
    tracer_provider.add_span_processor(processor)

    with tracer.start_as_current_span("unrelated operation"):
        pass

    spans = in_memory_exporter.get_finished_spans()
    transformed = _find_transformed_spans(spans)
    assert not transformed
