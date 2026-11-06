from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ...errors import ArtifactError, ConfigurationError, DependencyError
from ...exporters.artifacts import (checked_model, iter_model_messages, require_onnx,
                                    validate_backend_files)
from ..base import BaseBackendWrapper


def require_runtime():
    try:
        import onnxruntime
    except ImportError as error:
        raise DependencyError(
            'The onnxruntime backend requires optional dependency onnxruntime.'
        ) from error
    return onnxruntime


def require_cpu(device: str) -> None:
    if device != 'cpu':
        raise ConfigurationError(
            'ONNX Runtime local validation requires explicit device="cpu", '
            f'got {device!r}; GPU fallback is not supported.'
        )


def require_cpu_graph(graph_path: Path) -> None:
    """Reject BF16 graph computation before ORT import, staging or execution."""
    onnx = require_onnx()
    model, _ = checked_model(graph_path)
    for message in iter_model_messages(model):
        description = None
        if isinstance(
                message,
                onnx.TensorProto) and message.data_type == onnx.TensorProto.BFLOAT16:
            description = message.name or 'tensor'
        elif isinstance(message, onnx.TypeProto):
            for field in ('tensor_type', 'sparse_tensor_type'):
                if message.HasField(field) and getattr(
                        message, field).elem_type == onnx.TensorProto.BFLOAT16:
                    description = field
        elif (isinstance(message, onnx.NodeProto) and message.op_type == 'Cast'
              and message.domain in ('', 'ai.onnx') and any(
                  attribute.name == 'to' and attribute.i == onnx.TensorProto.BFLOAT16
                  for attribute in message.attribute)):
            description = message.name or 'Cast'
        if description is not None:
            raise ConfigurationError(
                f'{description}: BF16 is unsupported by the CPU onnxruntime backend; '
                'no implicit conversion is performed.')


class ONNXRuntimeWrapper(BaseBackendWrapper):
    """One real ONNX graph, CPUExecutionProvider only, named Torch CPU tensors."""

    def __init__(
        self,
        backend_files: Sequence[Path],
        device: str,
        target_metadata: Mapping[str, Any] | None = None,
        require_target_match: bool = False,
    ) -> None:
        require_cpu(device)
        if require_target_match and target_metadata is None:
            raise ConfigurationError(
                'ONNX Runtime artifact metadata is required '
                'when require_target_match=True.'
            )
        if target_metadata is not None and (
                target_metadata.get('backend') != 'onnxruntime'
                or target_metadata.get('device') != 'cpu'
                or target_metadata.get('execution_provider') != 'CPUExecutionProvider'):
            raise ArtifactError('ONNX Runtime metadata must identify the CPU backend.')
        graph_path = validate_backend_files(backend_files, target_metadata)
        require_cpu_graph(graph_path)
        ort = require_runtime()
        if 'CPUExecutionProvider' not in ort.get_available_providers():
            raise DependencyError(
                'ONNX Runtime CPUExecutionProvider is unavailable; '
                'no alternate provider will be used.'
            )
        try:
            import numpy as np
            import torch
        except ImportError as error:
            raise DependencyError(
                'ONNX Runtime named-tensor execution requires optional '
                'dependencies torch and numpy.'
            ) from error
        self._torch = torch
        self._np = np
        self._dtype_map = {
            'tensor(bool)': torch.bool,
            'tensor(uint8)': torch.uint8,
            'tensor(int8)': torch.int8,
            'tensor(int16)': torch.int16,
            'tensor(int32)': torch.int32,
            'tensor(int64)': torch.int64,
            'tensor(float16)': torch.float16,
            'tensor(float)': torch.float32,
            'tensor(double)': torch.float64,
        }
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(graph_path), sess_options=options, providers=['CPUExecutionProvider'])
        self.session.disable_fallback()
        if self.session.get_providers() != ['CPUExecutionProvider']:
            raise ConfigurationError(
                f'Unexpected ONNX Runtime providers: {self.session.get_providers()}')
        self._inputs = tuple(self.session.get_inputs())
        self._outputs = tuple(self.session.get_outputs())
        if not self._inputs or not self._outputs:
            raise ConfigurationError(
                'ONNX Runtime graph must have at least one tensor input and output.')
        for value in self._inputs + self._outputs:
            if value.type not in self._dtype_map:
                raise ConfigurationError(
                    f'{value.name}: unsupported ONNX Runtime tensor type '
                    f'{value.type}; BF16 and implicit dtype conversions '
                    'are unsupported.'
                )
            if not isinstance(value.shape, list) or any(
                    dim is not None and not isinstance(dim, (str, int))
                    for dim in value.shape):
                raise ConfigurationError(
                    f'{value.name}: invalid ONNX Runtime tensor shape {value.shape!r}.')
        if len(set(self.input_names)) != len(self._inputs) or len(
                set(self.output_names)) != len(self._outputs):
            raise ConfigurationError(
                'ONNX Runtime graph contains duplicate input or output names.')

    @property
    def input_names(self):
        return [value.name for value in self._inputs]

    @property
    def output_names(self):
        return [value.name for value in self._outputs]

    @property
    def tensor_metadata(self) -> Dict[str, Any]:

        def records(values):
            return [{
                'name': value.name,
                'dtype': str(self._dtype_map[value.type]).removeprefix('torch.'),
                'shape': list(value.shape)
            } for value in values]

        return {'inputs': records(self._inputs), 'outputs': records(self._outputs)}

    @staticmethod
    def _check_shape(name: str, shape: Sequence[int], expected: Sequence[Any],
                     symbols: Dict[str, int]) -> None:
        if len(shape) != len(expected):
            raise ConfigurationError(
                f'{name}: expected rank {len(expected)}, got shape {tuple(shape)}.')
        for axis, (size, dimension) in enumerate(zip(shape, expected)):
            if isinstance(dimension, int) and dimension >= 0 and size != dimension:
                raise ConfigurationError(
                    f'{name}: axis {axis} must be {dimension}, got {size}.')
            if isinstance(dimension, str) and dimension:
                if dimension in symbols and symbols[dimension] != size:
                    raise ConfigurationError(
                        f'{name}: symbolic dimension {dimension!r} '
                        f'must be {symbols[dimension]}, got {size}.'
                    )
                symbols[dimension] = size

    def infer(self,
              inputs: Mapping[str, Any],
              synchronize: bool = False) -> Dict[str, Any]:
        # CPU session.run is synchronous; synchronize never selects a GPU path.
        if not isinstance(inputs, Mapping) or any(not isinstance(name, str)
                                                  for name in inputs):
            raise TypeError(
                'ONNX Runtime inputs must be a mapping of exact string names '
                'to Torch CPU tensors.'
            )
        if set(inputs) != set(self.input_names):
            raise ConfigurationError(
                'ONNX Runtime input names differ: '
                f'missing={sorted(set(self.input_names) - set(inputs))}, '
                f'extra={sorted(set(inputs) - set(self.input_names))}.'
            )
        torch = self._torch
        symbols: Dict[str, int] = {}
        feed = {}
        for spec in self._inputs:
            tensor = inputs[spec.name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f'{spec.name}: expected a Torch CPU Tensor, '
                    f'got {type(tensor).__name__}.'
                )
            if tensor.device.type != 'cpu':
                raise ConfigurationError(
                    f'{spec.name}: expected a CPU tensor, got device {tensor.device}.')
            if tensor.dtype == torch.bfloat16:
                raise ConfigurationError(
                    f'{spec.name}: BF16 is unsupported; '
                    'no implicit conversion is performed.'
                )
            if tensor.dtype != self._dtype_map[spec.type]:
                raise ConfigurationError(
                    f'{spec.name}: expected dtype {self._dtype_map[spec.type]}, '
                    f'got {tensor.dtype}.'
                )
            if tensor.layout != torch.strided:
                raise ConfigurationError(
                    f'{spec.name}: only strided Torch tensors are supported, '
                    f'got {tensor.layout}.'
                )
            self._check_shape(spec.name, tensor.shape, spec.shape, symbols)
            feed[spec.name] = tensor.detach().contiguous().numpy()
        values = self.session.run(self.output_names, feed)
        if len(values) != len(self._outputs):
            raise ConfigurationError(
                'ONNX Runtime returned an unexpected number of outputs.')
        results = {}
        for spec, value in zip(self._outputs, values):
            if not isinstance(value, self._np.ndarray):
                raise ConfigurationError(
                    f'{spec.name}: ONNX Runtime returned a non-tensor output.')
            tensor = torch.from_numpy(value)
            if tensor.dtype != self._dtype_map[
                    spec.type] or tensor.device.type != 'cpu':
                raise ConfigurationError(
                    f'{spec.name}: ONNX Runtime output dtype/device violated '
                    'the graph contract.'
                )
            self._check_shape(spec.name, tensor.shape, spec.shape, symbols)
            results[spec.name] = tensor
        return results
