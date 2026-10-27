from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from ..contracts import TensorSpec


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

    def as_graph(
        self,
        name: str = 'model',
        input_specs: Tuple[TensorSpec, ...] = (),
        export_options: Optional[Mapping[str, Any]] = None,
        build_options: Optional[Mapping[str, Any]] = None,
    ) -> 'PreparedGraph':
        """Wrap the legacy single-model contract as one named export graph."""
        return PreparedGraph(
            module=self.module,
            inputs=self.inputs,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            calibration_source=self.calibration_source,
            metadata=dict(self.metadata),
            name=name,
            input_specs=input_specs,
            export_options=dict(export_options or {}),
            build_options=dict(build_options or {}),
        )


@dataclass
class PreparedGraph(PreparedModel):
    """One named graph in a potentially partitioned deployment."""

    name: str = 'model'
    input_specs: Tuple[TensorSpec, ...] = field(default_factory=tuple)
    export_options: Dict[str, Any] = field(default_factory=dict)
    build_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.input_specs = tuple(self.input_specs)
        self.export_options = dict(self.export_options)
        self.build_options = dict(self.build_options)
        if not self.name:
            raise ValueError('PreparedGraph.name cannot be empty.')
        if self.input_specs:
            spec_names = tuple(spec.name for spec in self.input_specs)
            input_names = tuple(self.inputs)
            if spec_names != input_names:
                raise ValueError(
                    f'{self.name}: input_specs {spec_names} do not match '
                    f'prepared inputs {input_names}.')


@dataclass
class PreparedDeployment:
    """Named export graphs plus an end-to-end source-model reference."""

    family: str
    graphs: Mapping[str, PreparedGraph]
    reference_runner: Callable[[], Mapping[str, Any]]
    output_names: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError('PreparedDeployment.family cannot be empty.')
        if not self.output_names:
            raise ValueError('PreparedDeployment.output_names cannot be empty.')
        if len(self.output_names) != len(set(self.output_names)):
            raise ValueError('PreparedDeployment.output_names must be unique.')

        graphs = dict(self.graphs)
        if not graphs:
            raise ValueError('PreparedDeployment requires at least one graph.')
        for name, graph in graphs.items():
            if not name:
                raise ValueError('PreparedDeployment graph names cannot be empty.')
            if not isinstance(graph, PreparedGraph):
                raise TypeError(
                    f'PreparedDeployment graph {name!r} must be PreparedGraph.')
            if name != graph.name:
                raise ValueError(
                    f'Graph mapping key {name!r} does not match '
                    f'PreparedGraph.name {graph.name!r}.')
        self.graphs = graphs

    @classmethod
    def from_model(
        cls,
        prepared: PreparedModel,
        family: str = 'generic',
        graph_name: str = 'model',
        input_specs: Tuple[TensorSpec, ...] = (),
        export_options: Optional[Mapping[str, Any]] = None,
        build_options: Optional[Mapping[str, Any]] = None,
    ) -> 'PreparedDeployment':
        """Promote a legacy single graph without changing existing adapters."""
        graph = prepared.as_graph(
            graph_name,
            input_specs=input_specs,
            export_options=export_options,
            build_options=build_options,
        )
        return cls(
            family=family,
            graphs={graph_name: graph},
            reference_runner=prepared.run_reference,
            output_names=prepared.output_names,
            metadata=dict(prepared.metadata),
        )

    def run_reference(self) -> Dict[str, Any]:
        result = self.reference_runner()
        if not isinstance(result, Mapping):
            raise TypeError(
                'PreparedDeployment reference runner must return a mapping.')
        missing = [name for name in self.output_names if name not in result]
        unknown = [name for name in result if name not in self.output_names]
        if missing or unknown:
            raise ValueError(
                'PreparedDeployment reference outputs do not match output_names: '
                f'missing={missing}, unknown={unknown}.')
        return {name: result[name] for name in self.output_names}


class ModelAdapter(ABC):
    """Boundary between FluxHermes and a source model repository."""

    @abstractmethod
    def prepare(self, device: str) -> PreparedModel:
        """Load the source checkpoint and return an export-ready graph."""
