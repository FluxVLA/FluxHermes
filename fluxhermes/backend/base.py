from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Type

from ..contracts import DeploymentConfig, TensorSpec
from ..core import Registry


@dataclass(frozen=True)
class BackendIR:
    """One named backend input graph and its build-time input contract."""

    path: Path
    inputs: Tuple[TensorSpec, ...]
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'path', Path(self.path).expanduser().resolve())
        object.__setattr__(self, 'inputs', tuple(self.inputs))
        object.__setattr__(self, 'options', dict(self.options))
        if not self.inputs:
            raise ValueError('BackendIR.inputs cannot be empty.')


@dataclass(frozen=True)
class BackendBuildResult:
    files: Tuple[Path, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        files = tuple(Path(path).expanduser().resolve() for path in self.files)
        artifacts = {
            str(name): Path(path).expanduser().resolve()
            for name, path in self.artifacts.items()
        }
        if files and artifacts and tuple(artifacts.values()) != files:
            raise ValueError(
                'BackendBuildResult files and named artifacts do not match.')
        if artifacts:
            files = tuple(artifacts.values())
        elif files:
            names = ['model'] if len(files) == 1 else [path.stem for path in files]
            if len(names) != len(set(names)):
                raise ValueError(
                    'BackendBuildResult cannot infer unique artifact names.')
            artifacts = dict(zip(names, files))
        else:
            raise ValueError('BackendBuildResult requires at least one artifact.')
        if any(not name for name in artifacts):
            raise ValueError('BackendBuildResult artifact names cannot be empty.')
        if any(Path(name).name != name or name in {'.', '..'} for name in artifacts):
            raise ValueError('BackendBuildResult artifact names must be safe names.')
        object.__setattr__(self, 'files', files)
        object.__setattr__(self, 'metadata', dict(self.metadata))
        object.__setattr__(self, 'artifacts', artifacts)


class BaseBackendWrapper(ABC):
    """Uniform named-tensor runtime wrapper for a deployment backend."""

    @abstractmethod
    def infer(
        self,
        inputs: Mapping[str, Any],
        synchronize: bool = False,
    ) -> Dict[str, Any]:
        pass

    def __call__(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        return self.infer(inputs)


class BaseBackendManager(ABC):
    backend_name: str

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        pass

    @classmethod
    @abstractmethod
    def get_version(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def to_backend(
        cls,
        ir_files: Mapping[str, BackendIR] | Sequence[Path],
        work_dir: Path,
        config: DeploymentConfig,
    ) -> BackendBuildResult:
        pass

    @classmethod
    @abstractmethod
    def build_wrapper(
        cls,
        backend_files: Mapping[str, Path] | Sequence[Path],
        device: str,
        **kwargs: Any,
    ) -> BaseBackendWrapper:
        pass

    @classmethod
    def check_env(cls) -> Dict[str, Any]:
        available = cls.is_available()
        return {
            'backend': cls.backend_name,
            'available': available,
            'version': cls.get_version() if available else None,
        }


BACKENDS: Registry[Type[BaseBackendManager]] = Registry('backends')


def get_backend_manager(name: str) -> Type[BaseBackendManager]:
    from . import register_builtin_backends

    register_builtin_backends()
    return BACKENDS.get(name)
