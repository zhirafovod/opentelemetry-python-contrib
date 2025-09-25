# Composite generator (Phase 1)
# Thin wrapper that allows future multiple emitters while today delegating to a
# single existing generator implementation (span/span+metric/span+metric+event).
from __future__ import annotations

from typing import Iterable, List

from .interfaces import GeneratorProtocol
from .types import Error, LLMInvocation


class CompositeGenerator(GeneratorProtocol):
    """Delegates lifecycle calls to an ordered list of generator instances.

    Enhanced in Phase 2 to enforce ordering semantics:
    - start: span emitters first, then others
    - finish: non-span emitters first, span emitters last
    - error: non-span emitters first, span emitters last
    This ensures metrics/events are emitted while the span is still open.
    """

    def __init__(self, generators: Iterable[GeneratorProtocol]):
        self._generators: List[GeneratorProtocol] = list(generators)
        self._primary = self._generators[0] if self._generators else None

    def add(self, generator: GeneratorProtocol):  # pragma: no cover
        self._generators.append(generator)
        if not self._primary:
            self._primary = generator

    def set_capture_content(self, value: bool):  # pragma: no cover
        for g in self._generators:
            if hasattr(g, "_capture_content"):
                try:
                    setattr(g, "_capture_content", value)
                except Exception:
                    pass

    def __getattr__(self, item):  # pragma: no cover
        primary = getattr(self, "_primary", None)
        if primary is not None:
            try:
                return getattr(primary, item)
            except AttributeError:
                pass
        raise AttributeError(item)

    # Internal helpers -----------------------------------------------------
    def _partition(self):
        span_emitters = []
        other_emitters = []
        for g in self._generators:
            role = getattr(g, "role", None)
            if role == "span":
                span_emitters.append(g)
            else:
                other_emitters.append(g)
        # Guarantee deterministic ordering: only one span emitter expected, but keep list
        return span_emitters, other_emitters

    # Lifecycle ------------------------------------------------------------
    def start(self, invocation: LLMInvocation) -> None:  # type: ignore[override]
        span_emitters, other_emitters = self._partition()
        for g in span_emitters:
            g.start(invocation)
        for g in other_emitters:
            g.start(invocation)

    def finish(self, invocation: LLMInvocation) -> None:  # type: ignore[override]
        span_emitters, other_emitters = self._partition()
        for g in other_emitters:
            g.finish(invocation)
        for g in span_emitters:
            g.finish(invocation)

    def error(self, error: Error, invocation: LLMInvocation) -> None:  # type: ignore[override]
        span_emitters, other_emitters = self._partition()
        for g in other_emitters:
            # Allow metrics (duration) capture before span ends
            try:
                g.error(error, invocation)
            except Exception:  # pragma: no cover
                pass
        for g in span_emitters:
            g.error(error, invocation)
