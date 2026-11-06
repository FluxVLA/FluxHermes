from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

from ...builders import build_tensorrt_engine
from ...contracts import DeploymentConfig
from ...errors import ConfigurationError
from ...runtime.tensorrt import _parse_cuda_device, _resolve_cuda_device
from ..base import BACKENDS, BackendBuildResult, BaseBackendManager
from .wrapper import TensorRTWrapper


def _build_device(config: DeploymentConfig) -> Any:
    source_value = config.codebase.get(
        'device', config.source.get('device', 'cuda:0'))
    source = _parse_cuda_device(source_value)
    backend = _parse_cuda_device(config.backend.get('device', source_value))
    if (source.index is not None and backend.index is not None
            and source.index != backend.index):
        raise ConfigurationError(
            f'TensorRT backend.device {backend} differs from source device '
            f'{source}; separate build/runtime devices are not supported.')
    source = _resolve_cuda_device(source)
    backend = _resolve_cuda_device(backend)
    if source != backend:
        raise ConfigurationError(
            f'TensorRT backend.device resolves to {backend}, but source '
            f'device resolves to {source}; '
            'use matching explicit CUDA indices.')
    return source


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
        ir_files: Sequence[Path],
        work_dir: Path,
        config: DeploymentConfig,
    ) -> BackendBuildResult:
        if len(ir_files) != 1:
            raise ValueError(
                'TensorRT currently expects exactly one ONNX graph.')
        device = _build_device(config)
        output = work_dir / 'model.plan'
        options = dict(config.backend, device=str(device))
        details = build_tensorrt_engine(
            ir_files[0],
            output,
            config.inputs,
            options,
        )
        return BackendBuildResult(files=(output,), metadata=details)

    @classmethod
    def build_wrapper(
        cls,
        backend_files: Sequence[Path],
        device: str,
        **kwargs: Any,
    ) -> TensorRTWrapper:
        if len(backend_files) != 1:
            raise ValueError(
                'TensorRT runtime expects exactly one engine file.')
        return TensorRTWrapper(
            backend_files[0],
            device=device,
            profile_index=kwargs.get('profile_index', 0),
            target_metadata=kwargs.get('target_metadata'),
            require_target_match=kwargs.get('require_target_match', False),
        )
