from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from ..adapters.base import PreparedDeployment


class ModelFamily(ABC):
    """Model-specific graph partitioning and compiled runtime orchestration."""

    name: str

    @classmethod
    @abstractmethod
    def matches(cls, model_config: Any) -> bool:
        """Return whether this family owns the supplied source-model config."""

    @abstractmethod
    def prepare(
        self,
        model: Any,
        model_config: Any,
        representative_inputs: Mapping[str, Any],
    ) -> PreparedDeployment:
        """Prepare the named graphs and end-to-end reference execution."""

    @abstractmethod
    def create_executor(
        self,
        engines: Mapping[str, Any],
        deployment: PreparedDeployment,
    ) -> Any:
        """Create the end-to-end executor from named compiled engines."""
