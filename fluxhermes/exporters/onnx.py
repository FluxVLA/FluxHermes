from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict

from ..adapters.base import PreparedModel
from ..errors import DependencyError


def export_onnx(
    prepared: PreparedModel,
    output_path: str | Path,
    options: Dict[str, Any],
) -> Path:
    """Export a prepared source model with stable tensor names."""
    try:
        import torch
    except Exception as error:
        raise DependencyError('ONNX export requires PyTorch.') from error
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    input_names = list(prepared.inputs)
    arguments = tuple(prepared.inputs[name] for name in input_names)
    kwargs: Dict[str, Any] = {
        'input_names': input_names,
        'output_names': list(prepared.output_names),
        'opset_version': int(options.get('opset', 18)),
        'do_constant_folding': bool(options.get('do_constant_folding', True)),
        'dynamic_axes': prepared.dynamic_axes or None,
        'verbose': bool(options.get('verbose', False)),
    }
    signature = inspect.signature(torch.onnx.export)
    if 'dynamo' in signature.parameters:
        kwargs['dynamo'] = bool(options.get('dynamo', False))
    if 'external_data' in signature.parameters:
        kwargs['external_data'] = bool(options.get('external_data', True))
    with torch.inference_mode():
        torch.onnx.export(prepared.module, arguments, str(output), **kwargs)
    if not output.is_file():
        raise RuntimeError(f'PyTorch did not create the ONNX file: {output}')
    if bool(options.get('check', True)):
        try:
            import onnx
        except Exception as error:
            raise DependencyError('ONNX checking requires the onnx package.') from error
        onnx.checker.check_model(str(output))
    return output
