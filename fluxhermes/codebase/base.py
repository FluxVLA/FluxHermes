from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type

from ..adapters.base import PreparedModel
from ..contracts import DeploymentConfig
from ..core import Registry
from ..errors import ConfigurationError


class BaseTaskProcessor(ABC):
    """Model/task semantic boundary, analogous to an MMDeploy task processor."""

    def __init__(self, config: DeploymentConfig, device: str) -> None:
        self.config = config
        self.device = device

    @abstractmethod
    def prepare_model(self) -> PreparedModel:
        """Build source model, export wrapper, samples, and calibration source."""

    def runtime_metadata(self) -> Dict[str, Any]:
        return {}


class BaseCodebase(ABC):
    task_registry: Registry[Type[BaseTaskProcessor]]

    @classmethod
    def build_task_processor(
        cls,
        config: DeploymentConfig,
        device: str,
    ) -> BaseTaskProcessor:
        task_name = str(config.codebase.get('task', ''))
        if not task_name:
            raise ConfigurationError('codebase_config.task cannot be empty.')
        task_type = cls.task_registry.get(task_name)
        return task_type(config=config, device=device)

    @classmethod
    def register_deploy_modules(cls) -> None:
        return None

    @classmethod
    def register_all_modules(cls) -> None:
        cls.register_deploy_modules()


CODEBASES: Registry[Type[BaseCodebase]] = Registry('codebases')


def get_codebase(name: str) -> Type[BaseCodebase]:
    from . import register_builtin_codebases

    register_builtin_codebases()
    return CODEBASES.get(name)
