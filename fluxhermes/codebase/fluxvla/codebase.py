from __future__ import annotations

from ...core import Registry
from ..base import CODEBASES, BaseCodebase, BaseTaskProcessor

FLUXVLA_TASKS: Registry[type[BaseTaskProcessor]] = Registry('fluxvla_tasks')


@CODEBASES.register_module('fluxvla')
class FluxVLACodebase(BaseCodebase):
    task_registry = FLUXVLA_TASKS

    @classmethod
    def register_deploy_modules(cls) -> None:
        from . import rewriters  # noqa: F401
