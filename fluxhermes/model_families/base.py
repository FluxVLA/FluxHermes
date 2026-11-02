from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Tuple

from ..adapters.base import PreparedDeployment


def config_mappings(value: Any) -> Tuple[Mapping[str, Any], ...]:
    """Return all mappings nested in a normalized training config."""
    nodes = []
    if isinstance(value, Mapping):
        nodes.append(value)
        for child in value.values():
            nodes.extend(config_mappings(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            nodes.extend(config_mappings(child))
    return tuple(nodes)


def config_nodes(value: Any, node_type: str) -> Tuple[Mapping[str, Any], ...]:
    """Return every nested config mapping with the requested type."""
    return tuple(
        node for node in config_mappings(value)
        if 'type' in node and node['type'] == node_type)


class ModelFamily(ABC):
    """Model-specific graph partitioning and compiled runtime orchestration."""

    name: str
    export_dtype: str
    validation_options: Mapping[str, float] = {
        'atol': 5e-2,
        'rtol': 5e-2,
        'min_cosine_similarity': 0.99,
    }

    @classmethod
    @abstractmethod
    def matches(cls, model_config: Any) -> bool:
        """Return whether this family owns the supplied source-model config."""

    @classmethod
    def configure_model(cls, model_config: Any) -> None:
        """Adjust construction-only settings before checkpoint loading."""

    @abstractmethod
    def make_representative_inputs(
        self,
        model: Any,
        training_config: Any,
        device: Any,
    ) -> Mapping[str, Any]:
        """Create deterministic policy inputs for export and parity checks."""

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
        metadata: Mapping[str, Any],
    ) -> Any:
        """Create the end-to-end executor from named compiled engines."""
