from opentelemetry.util.genai.generators.semconv_generator import (
    SemConvGenerator,
)
from opentelemetry.util.genai.types.generic import GenAI


class HookOrderTracker(SemConvGenerator):
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


def make_genai():
    return GenAI(request_model="test-model")


def test_start_hook_order():
    tracker = HookOrderTracker()
    tracker.start(make_genai())
    assert tracker.calls == ["before_start", "start", "after_start"]


def test_stop_hook_order():
    tracker = HookOrderTracker()
    tracker.stop(make_genai())
    assert tracker.calls == ["before_stop", "stop", "after_stop"]


def test_fail_hook_order():
    tracker = HookOrderTracker()
    tracker.fail(make_genai(), RuntimeError("fail"))
    assert tracker.calls == ["before_fail", "fail", "after_fail"]


def test_subclass_override():
    class Custom(SemConvGenerator):
        def _on_before_start(self, data):
            self.custom_called = True

    c = Custom()
    c.custom_called = False
    c.start(make_genai())
    assert c.custom_called


def test_interface_methods_exist():
    g = SemConvGenerator()
    assert hasattr(g, "start")
    assert hasattr(g, "stop")
    assert hasattr(g, "fail")
    assert callable(g.start)
    assert callable(g.stop)
    assert callable(g.fail)
