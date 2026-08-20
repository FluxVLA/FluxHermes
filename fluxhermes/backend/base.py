from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Type

from ..contracts import DeploymentConfig
from ..core import Registry


@dataclass(frozen=True)
class BackendBuildResult:
    files: Tuple[Path, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        ir_files: Sequence[Path],
        work_dir: Path,
        config: DeploymentConfig,
    ) -> BackendBuildResult:
        pass

    @classmethod
    @abstractmethod
    def build_wrapper(
        cls,
        backend_files: Sequence[Path],
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
