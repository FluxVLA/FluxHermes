from .base import (BACKENDS, BackendBuildResult, BaseBackendManager, BaseBackendWrapper,
                   get_backend_manager)


def register_builtin_backends() -> None:
    from .tensorrt import TensorRTBackendManager  # noqa: F401


__all__ = [
    'BACKENDS',
    'BackendBuildResult',
    'BaseBackendManager',
    'BaseBackendWrapper',
    'get_backend_manager',
    'register_builtin_backends',
]
