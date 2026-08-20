from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ...runtime.tensorrt import TensorRTEngine
from ..base import BaseBackendWrapper


class TensorRTWrapper(BaseBackendWrapper):
    """MMDeploy-style wrapper around the low-level TensorRT engine."""

    def __init__(
        self,
        engine_path: str | Path,
        device: str = 'cuda:0',
        profile_index: int = 0,
        target_metadata: Optional[Mapping[str, Any]] = None,
        require_target_match: bool = False,
    ) -> None:
        self.engine = TensorRTEngine(
            engine_path,
            profile_index=profile_index,
            device=device,
        )
        if require_target_match and target_metadata is not None:
            self.engine._check_target(target_metadata)

    @property
    def input_names(self):
        return self.engine.input_names

    @property
    def output_names(self):
        return self.engine.output_names

    def infer(
        self,
        inputs: Mapping[str, Any],
        synchronize: bool = False,
    ) -> Dict[str, Any]:
        return self.engine.infer(inputs, synchronize=synchronize)
