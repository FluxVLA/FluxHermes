from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ...errors import ArtifactError
from ...runtime.tensorrt import TensorRTEngine
from ..base import BaseBackendWrapper


def _normalize_engine_paths(
    engine_paths: str | Path | Mapping[str, Path] | Sequence[Path],
) -> Dict[str, Path]:
    if isinstance(engine_paths, (str, Path)):
        return {'model': Path(engine_paths)}
    if isinstance(engine_paths, Mapping):
        paths = {str(name): Path(path) for name, path in engine_paths.items()}
    else:
        values = tuple(Path(path) for path in engine_paths)
        names = ['model'] if len(values) == 1 else [path.stem for path in values]
        if len(names) != len(set(names)):
            raise ValueError('TensorRT engine names must be unique.')
        paths = dict(zip(names, values))
    if not paths or any(not name for name in paths):
        raise ValueError('TensorRT requires named engine paths.')
    return paths


class TensorRTWrapper(BaseBackendWrapper):
    """MMDeploy-style wrapper around the low-level TensorRT engine."""

    def __init__(
        self,
        engine_path: str | Path | Mapping[str, Path] | Sequence[Path],
        device: str = 'cuda:0',
        profile_index: int = 0,
        target_metadata: Optional[Mapping[str, Any]] = None,
        require_target_match: bool = False,
    ) -> None:
        paths = _normalize_engine_paths(engine_path)
        artifact_metadata = (
            target_metadata.get('artifacts', {}) if target_metadata is not None else {})
        self.engines = {}
        for name, path in paths.items():
            engine = TensorRTEngine(
                path,
                profile_index=profile_index,
                device=device,
            )
            if require_target_match and target_metadata is not None:
                expected = artifact_metadata.get(name, target_metadata)
                engine._check_target(expected)
            self.engines[name] = engine
        self.engine = (
            next(iter(self.engines.values())) if len(self.engines) == 1 else None)

    @property
    def engine_names(self):
        return tuple(self.engines)

    def get_engine(self, name: str) -> TensorRTEngine:
        try:
            return self.engines[name]
        except KeyError as error:
            available = ', '.join(self.engines)
            raise ArtifactError(
                f'Unknown TensorRT engine {name!r}; available: {available}.') from error

    @property
    def input_names(self):
        return self._single_engine().input_names

    @property
    def output_names(self):
        return self._single_engine().output_names

    def _single_engine(self) -> TensorRTEngine:
        if self.engine is None:
            raise ArtifactError(
                'This TensorRT bundle contains multiple engines; select one '
                'with infer_graph().')
        return self.engine

    def infer_graph(
        self,
        name: str,
        inputs: Mapping[str, Any],
        synchronize: bool = False,
    ) -> Dict[str, Any]:
        return self.get_engine(name).infer(inputs, synchronize=synchronize)

    def infer(
        self,
        inputs: Mapping[str, Any],
        synchronize: bool = False,
    ) -> Dict[str, Any]:
        return self._single_engine().infer(inputs, synchronize=synchronize)
