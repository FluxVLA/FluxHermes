from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..errors import DependencyError


def _save_external_model(model: Any, output: Path, threshold: int) -> None:
    import onnx

    data_path = output.with_name(output.name + '.data')
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=threshold,
    )


def optimize_onnx(
    input_path: str | Path,
    output_path: str | Path,
    options: Dict[str, Any],
) -> Path:
    """Run conservative ONNX graph simplification when requested."""
    source = Path(input_path).expanduser().resolve()
    if not bool(options.get('simplify', False)):
        return source
    try:
        import onnx
        from onnxsim import simplify
    except Exception as error:
        raise DependencyError(
            'ONNX simplification requires onnx and onnxsim.') from error
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source), load_external_data=True)
    shapes = options.get('input_shapes')
    simplified, checked = simplify(
        model,
        input_shapes=shapes,
        dynamic_input_shape=bool(options.get('dynamic_input_shape', False)),
        skipped_optimizers=list(options.get('skipped_optimizers', ())),
    )
    if not checked:
        raise RuntimeError('onnxsim equivalence check failed.')
    if bool(options.get('external_data', True)):
        _save_external_model(
            simplified,
            output,
            int(options.get('external_data_threshold', 1024)),
        )
    else:
        onnx.save_model(simplified, str(output))
    return output


def lower_onnx_precision(
    input_path: str | Path,
    output_path: str | Path,
    precision: str,
    options: Dict[str, Any],
) -> Path:
    """Make reduced precision explicit for TensorRT strongly typed builds."""
    source = Path(input_path).expanduser().resolve()
    if precision in {'fp32', 'int8'}:
        return source
    if precision != 'fp16':
        raise ValueError(f'Unsupported explicit graph precision: {precision}')
    if not bool(options.get('fp16_conversion', True)):
        return source
    try:
        import onnx
        from onnxconverter_common import float16
    except Exception as error:
        raise DependencyError(
            'FP16 ONNX lowering requires onnx and onnxconverter-common.') from error
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(source), load_external_data=True)
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=bool(options.get('keep_io_types', True)),
        disable_shape_infer=bool(options.get('disable_shape_infer', False)),
        op_block_list=list(options.get('fp16_op_block_list', ())) or None,
        node_block_list=list(options.get('fp16_node_block_list', ())) or None,
    )
    _save_external_model(
        converted,
        output,
        int(options.get('external_data_threshold', 1024)),
    )
    return output
