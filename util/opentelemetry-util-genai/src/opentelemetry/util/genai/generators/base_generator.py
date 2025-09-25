"""
Abstract base class for GenAI telemetry generators.

Defines the interface and extensibility hooks for telemetry generators that emit
telemetry (spans, metrics, events, etc.) for GenAI data types such as GenAIBaseInvocation.

Subclasses should implement the abstract methods and may override the protected hooks
for custom behavior before and after each main operation.
"""

from abc import ABC, abstractmethod

from opentelemetry.util.genai.types.generic import GenAI


class BaseGenerator(ABC):
    """
    Abstract base class for GenAI telemetry generators.

    Provides start, stop, and fail methods for emitting telemetry for GenAI data types.
    Subclasses can override protected hooks to customize behavior before and after each method.
    """

    def start(self, data: GenAI) -> None:
        """
        Start telemetry emission for a GenAI data type (e.g., GenAIBaseInvocation).
        Calls protected hooks before and after the main logic.
        """
        self._on_before_start(data)
        self._start(data)
        self._on_after_start(data)

    def stop(self, data: GenAI) -> None:
        """
        Stop telemetry emission for a GenAI data type.
        Calls protected hooks before and after the main logic.
        """
        self._on_before_stop(data)
        self._stop(data)
        self._on_after_stop(data)

    def fail(self, data: GenAI, error: Exception) -> None:
        """
        Emit telemetry for a failed GenAI operation.
        Calls protected hooks before and after the main logic.
        """
        self._on_before_fail(data, error)
        self._fail(data, error)
        self._on_after_fail(data, error)

    @abstractmethod
    def _start(self, data: GenAI) -> None:
        """
        Main logic for starting telemetry. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _stop(self, data: GenAI) -> None:
        """
        Main logic for stopping telemetry. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _fail(self, data: GenAI, error: Exception) -> None:
        """
        Main logic for handling telemetry on failure. Must be implemented by subclasses.
        """
        pass

    # Protected hooks for extensibility
    def _on_before_start(self, data: GenAI) -> None:
        """Hook called before start logic. Can be overridden by subclasses."""
        pass

    def _on_after_start(self, data: GenAI) -> None:
        """Hook called after start logic. Can be overridden by subclasses."""
        pass

    def _on_before_stop(self, data: GenAI) -> None:
        """Hook called before stop logic. Can be overridden by subclasses."""
        pass

    def _on_after_stop(self, data: GenAI) -> None:
        """Hook called after stop logic. Can be overridden by subclasses."""
        pass

    def _on_before_fail(self, data: GenAI, error: Exception) -> None:
        """Hook called before fail logic. Can be overridden by subclasses."""
        pass

    def _on_after_fail(self, data: GenAI, error: Exception) -> None:
        """Hook called after fail logic. Can be overridden by subclasses."""
        pass
