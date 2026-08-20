from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Iterable

from .errors import ConfigurationError


def add_python_paths(paths: Iterable[str]) -> None:
    for raw_path in reversed(tuple(paths)):
        path = str(Path(raw_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def load_symbol(reference: str) -> Any:
    if ':' not in reference:
        raise ConfigurationError(
            f"Plugin '{reference}' must use the 'module:attribute' form.")
    module_name, attribute = reference.split(':', 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise ConfigurationError(
            f"Could not import plugin module '{module_name}': {error}") from error
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise ConfigurationError(
            f"Plugin module '{module_name}' has no attribute '{attribute}'.") from error
