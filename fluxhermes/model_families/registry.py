from __future__ import annotations

from typing import Any, Type

from ..core import Registry
from ..errors import ConfigurationError
from .base import ModelFamily

MODEL_FAMILIES: Registry[Type[ModelFamily]] = Registry('model_families')


def get_model_family(name: str) -> Type[ModelFamily]:
    """Return one explicitly selected model-family implementation."""
    from . import register_builtin_model_families

    register_builtin_model_families()
    return MODEL_FAMILIES.get(name)


def detect_model_family(model_config: Any) -> Type[ModelFamily]:
    """Resolve exactly one registered family from a source-model config."""
    from . import register_builtin_model_families

    register_builtin_model_families()
    matches = [
        (name, family)
        for name, family in MODEL_FAMILIES.module_dict.items()
        if family.matches(model_config)
    ]
    if not matches:
        available = ', '.join(sorted(MODEL_FAMILIES.module_dict)) or '<empty>'
        raise ConfigurationError(
            'No registered model family matches the source-model config. '
            f'Available: {available}.')
    if len(matches) > 1:
        names = ', '.join(name for name, _ in matches)
        raise ConfigurationError(
            f'Multiple model families match the source-model config: {names}.')
    return matches[0][1]
