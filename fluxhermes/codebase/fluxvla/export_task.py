"""Explicit task for the independent installed FluxVLA export SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...adapters.fluxvla_export import FluxVLAExportAdapter
from ..base import BaseTaskProcessor
from .codebase import FLUXVLA_TASKS

if TYPE_CHECKING:
    from fluxvla_export import PreparedDeployment


@FLUXVLA_TASKS.register_module('export_package')
class ExportPackageTaskProcessor(BaseTaskProcessor):
    """Prepare an SDK-owned policy and preserve its named component graphs."""

    def prepare_model(self) -> PreparedDeployment:
        adapter = FluxVLAExportAdapter(self.config.source, self.config)
        return adapter.prepare(self.device)

    def runtime_metadata(self):
        # The pipeline takes the executor block directly from the prepared
        # deployment's metadata['runtime']; no architecture is copied here.
        return {
            'codebase': 'fluxvla',
            'task': 'export_package',
            'recipe': self.config.source['recipe'],
        }
