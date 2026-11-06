from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

from ...contracts import DeploymentConfig
from ...errors import ConfigurationError
from ...exporters.artifacts import stage_graph
from ..base import BACKENDS, BackendBuildResult, BaseBackendManager
from .wrapper import ONNXRuntimeWrapper, require_cpu, require_cpu_graph, require_runtime


@BACKENDS.register_module('onnxruntime')
class ONNXRuntimeBackendManager(BaseBackendManager):
    """Explicit optional CPU validation backend, not a TensorRT substitute."""

    backend_name = 'onnxruntime'

    @classmethod
    def is_available(cls) -> bool:
        return all(
            importlib.util.find_spec(name) is not None
            for name in ('onnxruntime', 'onnx', 'torch', 'numpy'))

    @classmethod
    def get_version(cls) -> str:
        return str(require_runtime().__version__)

    @classmethod
    def to_backend(cls, ir_files: Sequence[Path], work_dir: Path,
                   config: DeploymentConfig) -> BackendBuildResult:
        if len(ir_files) != 1:
            raise ValueError('ONNX Runtime expects exactly one ONNX graph per build.')
        if config.backend.get('type') != cls.backend_name:
            raise ConfigurationError(
                'ONNX Runtime requires backend_config.type="onnxruntime".')
        device = config.backend.get(
            'device', config.codebase.get('device', config.source.get('device')))
        require_cpu(device)
        providers = config.backend.get('providers', ['CPUExecutionProvider'])
        if providers != ['CPUExecutionProvider']:
            raise ConfigurationError(
                'ONNX Runtime accepts only providers=["CPUExecutionProvider"].')
        if config.precision not in {'fp32', 'fp16'}:
            raise ConfigurationError(
                f'Unsupported CPU ONNX Runtime precision {config.precision!r}; '
                'no dtype conversion is performed.'
            )
        require_cpu_graph(Path(ir_files[0]))
        files, metadata = stage_graph(Path(ir_files[0]), Path(work_dir) / 'onnxruntime')
        wrapper = ONNXRuntimeWrapper(files, device='cpu')
        metadata.update({
            'backend': cls.backend_name,
            'onnxruntime_version': cls.get_version(),
            'device': 'cpu',
            'execution_provider': 'CPUExecutionProvider',
            'precision': config.precision,
            **wrapper.tensor_metadata,
        })
        return BackendBuildResult(files=files, metadata=metadata)

    @classmethod
    def build_wrapper(cls, backend_files: Sequence[Path], device: str,
                      **kwargs: Any) -> ONNXRuntimeWrapper:
        allowed = {'profile_index', 'target_metadata', 'require_target_match'}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ConfigurationError(
                f'Unsupported ONNX Runtime wrapper options: {sorted(unknown)}')
        if kwargs.get('profile_index', 0) != 0:
            raise ConfigurationError(
                'ONNX Runtime has no TensorRT optimization profiles; '
                'profile_index must be 0.'
            )
        return ONNXRuntimeWrapper(
            backend_files,
            device=device,
            target_metadata=kwargs.get('target_metadata'),
            require_target_match=kwargs.get('require_target_match', False))
