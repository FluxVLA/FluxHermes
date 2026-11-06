"""One lazy, process-owned TensorRT logger with diagnostics on stderr."""
from __future__ import annotations

import sys
from threading import Lock
from typing import Any

from ..errors import DependencyError

_LOGGER: Any = None
_LOGGER_LOCK = Lock()


def get_tensorrt_logger(severity: Any = None) -> Any:
    """Return the same ILogger for every builder, parser, plugin and runtime.

    TensorRT retains a process-global logger pointer. Keep a strong reference
    here for its lifetime instead of creating a logger per engine. Its severity
    threshold is process-wide too: each explicit request sets the threshold;
    the default remains WARNING. Calls never redirect or discard stdout.
    """
    global _LOGGER
    try:
        import tensorrt as trt
    except ImportError as error:
        raise DependencyError('TensorRT logging requires tensorrt.') from error
    if severity is None:
        severity = trt.Logger.WARNING
    with _LOGGER_LOCK:
        if _LOGGER is None:

            class StderrLogger(trt.ILogger):

                def __init__(self) -> None:
                    trt.ILogger.__init__(self)
                    self._write_lock = Lock()
                    self.min_severity = severity

                def log(self, level: Any, message: str) -> None:
                    with self._write_lock:
                        if int(level) <= int(self.min_severity):
                            print(f'[TensorRT] [{level.name}] {message}',
                                  file=sys.stderr, flush=True)

            _LOGGER = StderrLogger()
        with _LOGGER._write_lock:
            _LOGGER.min_severity = severity
        return _LOGGER
