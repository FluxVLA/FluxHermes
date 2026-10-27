from .base import ModelFamily
from .registry import MODEL_FAMILIES, detect_model_family, get_model_family


def register_builtin_model_families() -> None:
    from .pi05 import PI05ModelFamily  # noqa: F401


__all__ = [
    'MODEL_FAMILIES',
    'ModelFamily',
    'detect_model_family',
    'get_model_family',
    'register_builtin_model_families',
]
