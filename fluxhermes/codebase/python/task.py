from __future__ import annotations

from ...adapters import PythonFactoryAdapter
from ...adapters.base import PreparedModel
from ..base import BaseTaskProcessor
from .codebase import PYTHON_TASKS


@PYTHON_TASKS.register_module('custom')
class CustomTaskProcessor(BaseTaskProcessor):

    def prepare_model(self) -> PreparedModel:
        adapter = PythonFactoryAdapter(self.config.source, self.config)
        return adapter.prepare(self.device)

    def runtime_metadata(self):
        return {
            'codebase': 'python',
            'task': 'custom',
        }
