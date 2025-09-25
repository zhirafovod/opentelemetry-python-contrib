"""
Unit tests for BaseGenerator abstract class in opentelemetry-util-genai.

Verifies interface, hook call order, and extensibility via subclassing.
"""

import pytest

from opentelemetry.util.genai.generators.base_generator import BaseGenerator
from opentelemetry.util.genai.types.generic import GenAI


class TestableGenerator(BaseGenerator):
    def __init__(self):
        self.calls = []
        super().__init__()

    def _on_before_start(self, data):
        self.calls.append("before_start")

    def _start(self, data):
        self.calls.append("start")

    def _on_after_start(self, data):
        self.calls.append("after_start")

    def _on_before_stop(self, data):
        self.calls.append("before_stop")

    def _stop(self, data):
        self.calls.append("stop")

    def _on_after_stop(self, data):
        self.calls.append("after_stop")

    def _on_before_fail(self, data, error):
        self.calls.append("before_fail")

    def _fail(self, data, error):
        self.calls.append("fail")

    def _on_after_fail(self, data, error):
        self.calls.append("after_fail")


def test_start_hook_order():
    g = TestableGenerator()
    g.start(GenAI(request_model="test-model"))
    assert g.calls == ["before_start", "start", "after_start"]


def test_stop_hook_order():
    g = TestableGenerator()
    g.stop(GenAI(request_model="test-model"))
    assert g.calls == ["before_stop", "stop", "after_stop"]


def test_fail_hook_order():
    g = TestableGenerator()
    err = Exception("fail")
    g.fail(GenAI(request_model="test-model"), err)
    assert g.calls == ["before_fail", "fail", "after_fail"]


def test_subclass_can_override_hooks():
    class CustomGen(TestableGenerator):
        def _on_before_start(self, data):
            super()._on_before_start(data)
            self.calls.append("custom_before_start")

    g = CustomGen()
    g.start(GenAI(request_model="test-model"))
    assert g.calls == [
        "before_start",
        "custom_before_start",
        "start",
        "after_start",
    ]


def test_abstract_methods_enforced():
    class IncompleteGen(BaseGenerator):
        def _start(self, data):
            pass

        def _stop(self, data):
            pass

        # _fail not implemented

    with pytest.raises(TypeError):
        IncompleteGen()
