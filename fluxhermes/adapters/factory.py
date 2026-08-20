from __future__ import annotations

from typing import Any, Dict

from ..errors import ConfigurationError
from .base import ModelAdapter
from .fluxvla import FluxVLAAdapter
from .python import PythonFactoryAdapter


def build_adapter(source: Dict[str, Any], deployment: Any) -> ModelAdapter:
    source_type = source.get('type')
    if source_type == 'fluxvla':
        return FluxVLAAdapter(source, deployment)
    if source_type == 'python':
        return PythonFactoryAdapter(source, deployment)
    raise ConfigurationError(f'Unknown source adapter type: {source_type}')
