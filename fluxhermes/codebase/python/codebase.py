from __future__ import annotations

from ...core import Registry
from ..base import CODEBASES, BaseCodebase, BaseTaskProcessor

PYTHON_TASKS: Registry[type[BaseTaskProcessor]] = Registry('python_tasks')


@CODEBASES.register_module('python')
class PythonCodebase(BaseCodebase):
    task_registry = PYTHON_TASKS
