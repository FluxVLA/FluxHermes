from __future__ import annotations

from typing import Any, Dict

from ..errors import ConfigurationError
from ..plugins import add_python_paths, load_symbol
from .base import ModelAdapter, PreparedModel


class PythonFactoryAdapter(ModelAdapter):
    """Adapter for project-specific recipes supplied as a Python plugin."""

    def __init__(self, source: Dict[str, Any], deployment: Any) -> None:
        self.source = source
        self.deployment = deployment

    def prepare(self, device: str) -> PreparedModel:
        add_python_paths(self.source.get('python_paths', ()))
        reference = self.source.get('factory')
        if not reference:
            raise ConfigurationError('python source adapter requires factory.')
        factory = load_symbol(reference)
        result = factory(
            deployment=self.deployment,
            device=device,
            **dict(self.source.get('kwargs', {})),
        )
        if isinstance(result, PreparedModel):
            return result
        if hasattr(result, 'prepare'):
            prepared = result.prepare(device)
            if isinstance(prepared, PreparedModel):
                return prepared
        raise ConfigurationError(
            f'Python factory {reference} must return PreparedModel or ModelAdapter.')
