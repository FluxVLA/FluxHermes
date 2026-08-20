from __future__ import annotations

import os
import runpy
from pathlib import Path
from typing import Any, Dict

from .contracts import DeploymentConfig
from .errors import ConfigurationError


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> DeploymentConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f'Deployment config not found: {config_path}')
    if config_path.suffix != '.py':
        raise ConfigurationError('Deployment configs must be Python files.')
    namespace = runpy.run_path(str(config_path))
    deployment = namespace.get('deployment')
    if not isinstance(deployment, dict):
        raise ConfigurationError(
            f'{config_path} must define a top-level deployment dictionary.')
    expanded: Dict[str, Any] = _expand(deployment)
    return DeploymentConfig.from_dict(expanded, config_path=config_path)
