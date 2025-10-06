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

from typing import Optional
from opentelemetry import trace
from opentelemetry.trace import Tracer
from ..types import LLMInvocation

class TraceloopSpanGenerator:
    def __init__(self, tracer: Optional[Tracer] = None, capture_content: bool = False):
        self._tracer = tracer or trace.get_tracer(__name__)
        self._capture_content = capture_content

    def start(self, invocation: LLMInvocation):
        override = getattr(invocation, "attributes", {}).get("_traceloop_new_name")
        if override:
            span_name = override
        else:
            name = getattr(invocation, "request_model", "llm")
            span_name = f"chat {name}" if not str(name).startswith("chat ") else str(name)
        span = self._tracer.start_span(span_name, kind=trace.SpanKind.CLIENT)
        invocation.span = span
        invocation.context_token = trace.use_span(span, end_on_exit=False)
        invocation.context_token.__enter__()
        # apply starting attributes
        for k, v in getattr(invocation, "attributes", {}).items():
            try:
                span.set_attribute(k, v)
            except Exception:
                pass

    def finish(self, invocation: LLMInvocation):
        span = getattr(invocation, "span", None)
        if not span:
            return
        # re-apply attributes (after transformations)
        for k, v in getattr(invocation, "attributes", {}).items():
            try:
                span.set_attribute(k, v)
            except Exception:
                pass
        token = getattr(invocation, "context_token", None)
        if token and hasattr(token, "__exit__"):
            try:
                token.__exit__(None, None, None)
            except Exception:
                pass
        span.end()

    def error(self, error, invocation: LLMInvocation):  # pragma: no cover - unused in tests now
        self.finish(invocation)
