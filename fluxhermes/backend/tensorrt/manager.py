from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ...builders import build_tensorrt_engine
from ...contracts import DeploymentConfig
from ..base import BACKENDS, BackendBuildResult, BackendIR, BaseBackendManager
from .wrapper import TensorRTWrapper


def _artifact_name(value: str) -> str:
    name = str(value)
    if not name or Path(name).name != name or name in {'.', '..'}:
        raise ValueError(f'Unsafe backend artifact name: {value!r}')
    return name


def _normalize_ir_files(
    ir_files: Mapping[str, BackendIR] | Sequence[Path],
    config: DeploymentConfig,
) -> Dict[str, BackendIR]:
    if isinstance(ir_files, Mapping):
        normalized = {}
        for raw_name, value in ir_files.items():
            name = _artifact_name(raw_name)
            normalized[name] = (
                value if isinstance(value, BackendIR) else BackendIR(
                    path=Path(value), inputs=config.inputs))
        if not normalized:
            raise ValueError('TensorRT requires at least one ONNX graph.')
        return normalized

    paths = tuple(Path(path) for path in ir_files)
    if not paths:
        raise ValueError('TensorRT requires at least one ONNX graph.')
    names = ['model'] if len(paths) == 1 else [path.stem for path in paths]
    if len(names) != len(set(names)):
        raise ValueError('TensorRT could not infer unique ONNX graph names.')
    return {
        _artifact_name(name): BackendIR(path=path, inputs=config.inputs)
        for name, path in zip(names, paths)
    }


@BACKENDS.register_module('tensorrt')
class TensorRTBackendManager(BaseBackendManager):
    backend_name = 'tensorrt'

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec('tensorrt') is not None

    @classmethod
    def get_version(cls) -> str:
        import tensorrt

        return str(tensorrt.__version__)

    @classmethod
    def to_backend(
        cls,
        ir_files: Mapping[str, BackendIR] | Sequence[Path],
        work_dir: Path,
        config: DeploymentConfig,
    ) -> BackendBuildResult:
        graphs = _normalize_ir_files(ir_files, config)
        artifacts: Dict[str, Path] = {}
        graph_metadata: Dict[str, Dict[str, Any]] = {}
        for name, graph in graphs.items():
            if len(graphs) == 1 and name == 'model':
                output = work_dir / 'model.plan'
            else:
                output = work_dir / 'engines' / f'{name}.plan'
            options = dict(config.backend)
            options.update(graph.options)
            details = build_tensorrt_engine(
                graph.path,
                output,
                graph.inputs,
                options,
            )
            artifacts[name] = output
            graph_metadata[name] = details

        metadata: Dict[str, Any] = {'artifacts': graph_metadata}
        if len(graph_metadata) == 1:
            metadata.update(next(iter(graph_metadata.values())))
        return BackendBuildResult(artifacts=artifacts, metadata=metadata)

    @classmethod
    def build_wrapper(
        cls,
        backend_files: Mapping[str, Path] | Sequence[Path],
        device: str,
        **kwargs: Any,
    ) -> TensorRTWrapper:
        return TensorRTWrapper(
            backend_files,
            device=device,
            profile_index=int(kwargs.get('profile_index', 0)),
            target_metadata=kwargs.get('target_metadata'),
            require_target_match=bool(kwargs.get('require_target_match', False)),
        )
