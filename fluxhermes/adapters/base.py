from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple


@dataclass
class PreparedModel:
    """An exportable module plus deterministic representative inputs."""

    module: Any
    inputs: Mapping[str, Any]
    output_names: Tuple[str, ...]
    dynamic_axes: Dict[str, Dict[int, str]] = field(default_factory=dict)
    calibration_source: Optional[Callable[[], Iterable[Mapping[str, Any]]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def run_reference(self) -> Dict[str, Any]:
        values = tuple(self.inputs[name] for name in self.inputs)
        result = self.module(*values)
        if isinstance(result, dict):
            return {name: result[name] for name in self.output_names}
        if len(self.output_names) == 1 and not isinstance(result, (list, tuple)):
            return {self.output_names[0]: result}
        if not isinstance(result, (list, tuple)):
            raise TypeError(
                'Multiple output_names require the module to return a list or tuple.')
        if len(result) != len(self.output_names):
            raise ValueError(
                f'Module returned {len(result)} values for '
                f'{len(self.output_names)} output names.')
        return dict(zip(self.output_names, result))


class ModelAdapter(ABC):
    """Boundary between FluxHermes and a source model repository."""

    @abstractmethod
    def prepare(self, device: str) -> PreparedModel:
        """Load the source checkpoint and return an export-ready graph."""
