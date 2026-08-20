from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

from ..errors import ConfigurationError, DependencyError


class _CalibrationReader:

    def __init__(
        self,
        source: Callable[[], Iterable[Mapping[str, Any]]],
    ) -> None:
        self._source = source
        self._iterator = iter(source())

    @staticmethod
    def _numpy(value: Any) -> Any:
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        return value

    def get_next(self):
        try:
            sample = next(self._iterator)
        except StopIteration:
            return None
        return {name: self._numpy(value) for name, value in sample.items()}

    def rewind(self) -> None:
        self._iterator = iter(self._source())


def quantize_onnx_qdq(
    input_path: str | Path,
    output_path: str | Path,
    calibration_source: Callable[[], Iterable[Mapping[str, Any]]] | None,
    options: Dict[str, Any],
) -> Path:
    """Create an INT8 Q/DQ ONNX graph using representative robot data."""
    if calibration_source is None:
        raise ConfigurationError(
            'INT8 quantization requires source.calibration_factory with '
            'representative, preprocessed robot samples.')
    try:
        from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                              quantize_static)
    except Exception as error:
        raise DependencyError(
            'INT8 Q/DQ quantization requires onnxruntime-gpu.') from error
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    method_name = str(options.get('calibration_method', 'minmax')).lower()
    methods = {
        'minmax': CalibrationMethod.MinMax,
        'entropy': CalibrationMethod.Entropy,
        'percentile': CalibrationMethod.Percentile,
    }
    if method_name not in methods:
        raise ConfigurationError(
            f'Unsupported calibration method: {method_name}')
    extra_options = {
        'ActivationSymmetric': bool(options.get('activation_symmetric', True)),
        'WeightSymmetric': bool(options.get('weight_symmetric', True)),
    }
    extra_options.update(dict(options.get('extra_options', {})))
    quantize_static(
        model_input=str(source),
        model_output=str(output),
        calibration_data_reader=_CalibrationReader(calibration_source),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=bool(options.get('per_channel', True)),
        reduce_range=bool(options.get('reduce_range', False)),
        use_external_data_format=bool(
            options.get('use_external_data_format', True)),
        calibrate_method=methods[method_name],
        nodes_to_exclude=list(options.get('nodes_to_exclude', ())),
        op_types_to_quantize=(
            list(options['op_types_to_quantize'])
            if options.get('op_types_to_quantize') else None),
        extra_options=extra_options,
    )
    return output
