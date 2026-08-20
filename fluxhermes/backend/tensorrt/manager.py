from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Sequence

from ...builders import build_tensorrt_engine
from ...contracts import DeploymentConfig
from ..base import BACKENDS, BackendBuildResult, BaseBackendManager
from .wrapper import TensorRTWrapper


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
            raise ValueError('TensorRT currently expects exactly one ONNX graph.')
        output = work_dir / 'model.plan'
        details = build_tensorrt_engine(
            ir_files[0],
            output,
            config.inputs,
            config.backend,
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
            raise ValueError('TensorRT runtime expects exactly one engine file.')
        return TensorRTWrapper(
            backend_files[0],
            device=device,
            profile_index=int(kwargs.get('profile_index', 0)),
            target_metadata=kwargs.get('target_metadata'),
            require_target_match=bool(kwargs.get('require_target_match', False)),
        )
