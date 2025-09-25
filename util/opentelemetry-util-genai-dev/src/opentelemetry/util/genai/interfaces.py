# Phase 1 refactor: introduce lightweight protocol-style interfaces so future
# composite generator + plugin system can rely on a stable narrow contract.
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import Error, LLMInvocation


@runtime_checkable
class GeneratorProtocol(Protocol):
    """Protocol implemented by all telemetry generators / emitters.

    Current code base already has multiple concrete classes that expose
    start/finish/error(invocation) methods. We formalize that surface here
    so a CompositeGenerator can wrap *any* existing implementation without
    rewriting them or introducing deep inheritance.
    """

    def start(
        self, invocation: LLMInvocation
    ) -> None:  # pragma: no cover - structural
        ...

    def finish(
        self, invocation: LLMInvocation
    ) -> None:  # pragma: no cover - structural
        ...

    def error(
        self, error: Error, invocation: LLMInvocation
    ) -> None:  # pragma: no cover - structural
        ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol for evaluator objects (future phases may broaden).

    Current handler already uses a registry pattern; this protocol allows
    type narrowing without importing concrete evaluator implementations.
    """

    def evaluate(
        self, invocation: LLMInvocation
    ) -> Any:  # pragma: no cover - structural
        ...


class EmitterMeta:
    """Simple metadata mixin for emitters (role/name used by future plugin system).

    For Phase 1 this is optional; we include it now so existing generators can
    be wrapped without modification. If a concrete generator lacks these
    attributes defaults are applied by the composite builder.
    """

    role: str = "span"  # default / legacy generators are span focused
    name: str = "legacy"
    override: bool = False

    def handles(
        self, invocation: LLMInvocation
    ) -> bool:  # pragma: no cover (trivial)
        return True
