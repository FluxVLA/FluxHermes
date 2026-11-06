"""Compatibility imports for backend-independent ONNX artifact utilities.

CPU dtype/provider restrictions belong to the ONNX Runtime wrapper/manager.
The shared staging implementation intentionally preserves BF16 artifacts.
"""
from ...exporters.artifacts import checked_model, stage_graph, validate_backend_files

__all__ = ['checked_model', 'stage_graph', 'validate_backend_files']
