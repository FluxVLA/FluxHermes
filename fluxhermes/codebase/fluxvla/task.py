from __future__ import annotations

from ...adapters import FluxVLAAdapter
from ...adapters.base import PreparedModel
from ..base import BaseTaskProcessor
from .codebase import FLUXVLA_TASKS


@FLUXVLA_TASKS.register_module('predict_action')
class PredictActionTaskProcessor(BaseTaskProcessor):
    """Prepare FluxVLA ``predict_action`` as a named deployment graph."""

    def prepare_model(self) -> PreparedModel:
        adapter = FluxVLAAdapter(self.config.source, self.config)
        return adapter.prepare(self.device)

    def runtime_metadata(self):
        return {
            'codebase': 'fluxvla',
            'task': 'predict_action',
        }
