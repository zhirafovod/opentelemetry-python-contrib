# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import importlib

import pytest

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import LLMInvocation, InputMessage, OutputMessage, Text


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Ensure flags start unset each test
    for key in [
        "OTEL_GENAI_ENABLE_TRACELOOP_TRANSLATOR",
        "OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY",
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield
    for key in [
        "OTEL_GENAI_ENABLE_TRACELOOP_TRANSLATOR",
        "OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY",
        "OTEL_INSTRUMENTATION_GENAI_EMITTERS",
    ]:
        monkeypatch.delenv(key, raising=False)


def _fresh_handler():
    # Force re-parse of env + pipeline rebuild by reloading config-dependent modules
    import opentelemetry.util.genai.emitters.configuration as cfg
    importlib.reload(cfg)
    import opentelemetry.util.genai.handler as handler_mod
    importlib.reload(handler_mod)
    return handler_mod.TelemetryHandler()


def test_translator_promotes_prefixed(monkeypatch):
    monkeypatch.setenv("OTEL_GENAI_ENABLE_TRACELOOP_TRANSLATOR", "1")
    # Ensure standard span + compat so we can observe merged attributes
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span,traceloop_compat")

    handler = _fresh_handler()
    inv = LLMInvocation(
        request_model="gpt-4",
        input_messages=[InputMessage(role="user", parts=[Text("hi")])],
        attributes={
            "traceloop.workflow.name": "main_flow",
            "traceloop.entity.name": "AgentX",
            "traceloop.entity.path": "root/branch/leaf",
            "traceloop.callback.name": "root_chain",
            "traceloop.callback.id": "cb-123",
        },
    )
    handler.start_llm(inv)
    # Translator runs on start; attributes should be promoted now
    assert inv.attributes.get("gen_ai.workflow.name") == "main_flow"
    assert inv.attributes.get("gen_ai.agent.name") == "AgentX"
    assert inv.attributes.get("gen_ai.workflow.path") == "root/branch/leaf"
    assert inv.attributes.get("gen_ai.callback.name") == "root_chain"
    assert inv.attributes.get("gen_ai.callback.id") == "cb-123"
    # Original keys retained by default
    assert "traceloop.entity.path" in inv.attributes


def test_translator_promotes_raw(monkeypatch):
    monkeypatch.setenv("OTEL_GENAI_ENABLE_TRACELOOP_TRANSLATOR", "1")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    handler = _fresh_handler()
    inv = LLMInvocation(
        request_model="gpt-4",
        input_messages=[],
        attributes={
            "workflow.name": "flow_raw",
            "entity.name": "AgentRaw",
            "entity.path": "a/b/c",
        },
    )
    handler.start_llm(inv)
    assert inv.attributes.get("gen_ai.workflow.name") == "flow_raw"
    assert inv.attributes.get("gen_ai.agent.name") == "AgentRaw"
    assert inv.attributes.get("gen_ai.workflow.path") == "a/b/c"


def test_translator_does_not_overwrite(monkeypatch):
    monkeypatch.setenv("OTEL_GENAI_ENABLE_TRACELOOP_TRANSLATOR", "1")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    handler = _fresh_handler()
    inv = LLMInvocation(
        request_model="gpt-4",
        input_messages=[],
        attributes={
            "traceloop.workflow.name": "legacy_name",
            "gen_ai.workflow.name": "canonical_name",
        },
    )
    handler.start_llm(inv)
    # Existing canonical value preserved
    assert inv.attributes.get("gen_ai.workflow.name") == "canonical_name"


def test_translator_strip_legacy(monkeypatch):
    monkeypatch.setenv("OTEL_GENAI_ENABLE_TRACELOOP_TRANSLATOR", "1")
    monkeypatch.setenv("OTEL_GENAI_TRACELOOP_TRANSLATOR_STRIP_LEGACY", "1")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    handler = _fresh_handler()
    inv = LLMInvocation(
        request_model="gpt-4",
        input_messages=[],
        attributes={
            "traceloop.entity.path": "strip/me",
        },
    )
    handler.start_llm(inv)
    assert inv.attributes.get("gen_ai.workflow.path") == "strip/me"
    # Legacy removed
    assert "traceloop.entity.path" not in inv.attributes


def test_callback_sets_operation_source(monkeypatch):
    monkeypatch.setenv("OTEL_GENAI_ENABLE_TRACELOOP_TRANSLATOR", "1")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_EMITTERS", "span")
    handler = _fresh_handler()
    inv = LLMInvocation(
        request_model="gpt-4",
        input_messages=[],
        attributes={
            "traceloop.callback.name": "chain_node",
        },
    )
    handler.start_llm(inv)
    assert inv.attributes.get("gen_ai.callback.name") == "chain_node"
    assert inv.attributes.get("gen_ai.operation.source") == "chain_node"
