from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping

from ..bundle import verify_bundle
from ..errors import ArtifactError, ConfigurationError, DependencyError
from .tensorrt_logger import get_tensorrt_logger


def _parse_cuda_device(value: Any) -> Any:
    """Validate a CUDA selector using real torch, without initializing CUDA."""
    try:
        import torch
    except ImportError as error:
        raise DependencyError(
            'TensorRT device selection requires torch.') from error
    if not isinstance(value, (str, torch.device)):
        raise ConfigurationError(
            f'TensorRT device must be a CUDA device string, got {value!r}.')
    try:
        device = torch.device(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ConfigurationError(
            f'Invalid TensorRT device {value!r}.') from error
    if device.type != 'cuda':
        raise ConfigurationError(
            f'TensorRT device must be CUDA, got {device}.')
    return device


def _resolve_cuda_device(value: Any) -> Any:
    """Bind an unindexed CUDA selector without changing the current device."""
    device = _parse_cuda_device(value)
    import torch

    if not torch.cuda.is_available():
        raise DependencyError('TensorRT requires an available CUDA GPU.')
    try:
        count = torch.cuda.device_count()
        index = (torch.cuda.current_device()
                 if device.index is None else device.index)
    except Exception as error:
        raise DependencyError(
            f'Cannot resolve TensorRT CUDA device {device}: {error}'
        ) from error
    if not 0 <= index < count:
        raise ConfigurationError(
            f'TensorRT CUDA device index {index} is outside [0, {count}).')
    return torch.device('cuda', index)


def _validate_profile_index(value: Any, count: int | None = None) -> int:
    if type(value) is not int or value < 0:
        raise ConfigurationError(
            'TensorRT profile_index must be a non-negative integer: '
            f'{value!r}.')
    if count is not None and value >= max(count, 1):
        raise ConfigurationError(
            f'TensorRT profile_index {value} is outside the engine profile '
            f'range [0, {max(count, 1)}).')
    return value


def _validate_target_metadata(expected: Any) -> None:
    """Reject incomplete strict-match facts before loading an engine."""
    if not isinstance(expected, Mapping):
        raise ArtifactError(
            'Strict TensorRT target matching requires metadata.')
    for field in ('tensorrt_version', 'device', 'cuda_version'):
        value = expected.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ArtifactError(
                f'Strict TensorRT target metadata requires {field}.')
    capability = expected.get('compute_capability')
    if (not isinstance(capability, (list, tuple)) or len(capability) != 2
            or any(type(value) is not int for value in capability)
            or capability[0] <= 0 or capability[1] < 0):
        raise ArtifactError(
            'Strict TensorRT target metadata requires compute_capability '
            'as [positive major, non-negative minor].')


def _cuda_target_metadata(device: Any, *, tensorrt_version: str) -> dict:
    import torch

    try:
        with torch.cuda.device(device):
            details = {
                'tensorrt_version': tensorrt_version,
                'device': torch.cuda.get_device_name(device.index),
                'device_index': device.index,
                'compute_capability': list(
                    torch.cuda.get_device_capability(device.index)),
                'cuda_version': torch.version.cuda,
            }
        _validate_target_metadata(details)
    except Exception as error:
        raise DependencyError(
            f'Cannot collect TensorRT target facts for {device}: {error}'
        ) from error
    return details


def _typed_input(name: str, value: Any, expected: Any) -> Any:
    """Accept real Torch/array inputs, but never convert their dtype."""
    import torch

    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f'TensorRT input {name!r} must be tensor-compatible: {error}'
            ) from error
    if value.dtype != expected:
        raise TypeError(
            f'TensorRT input {name!r} dtype mismatch: expected {expected}, '
            f'actual {value.dtype}; implicit dtype conversion is forbidden.')
    return value


class TensorRTEngine:
    """TensorRT 10/11 with a fixed device and PyTorch-owned CUDA buffers.

    Calls on one execution context are serialized. A later call finishes the
    previous enqueue before changing bindings/shapes; synchronize=False still
    returns the current call asynchronously. Input dtypes are preserved.
    """

    def __init__(
        self,
        engine_path: str | Path,
        profile_index: int = 0,
        device: str = 'cuda:0',
        *,
        target_metadata: Mapping[str, Any] | None = None,
        require_target_match: bool = False,
    ) -> None:
        if type(require_target_match) is not bool:
            raise ConfigurationError('require_target_match must be a boolean.')
        if require_target_match:
            _validate_target_metadata(target_metadata)
        self.profile_index = _validate_profile_index(profile_index)
        self.device = _resolve_cuda_device(device)
        try:
            import tensorrt as trt
            import torch
        except ImportError as error:
            raise DependencyError(
                'TensorRT runtime requires both tensorrt and torch.'
            ) from error
        self.trt = trt
        self.torch = torch
        self.path = Path(engine_path).expanduser().resolve()
        if not self.path.is_file():
            raise ArtifactError(f'TensorRT engine not found: {self.path}')
        self._lock = Lock()
        self._pending_event = None
        self._pending_buffers = None
        with torch.cuda.device(self.device):
            if require_target_match:
                self._check_target(target_metadata)
            self.logger = get_tensorrt_logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self.logger, '')
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(
                self.path.read_bytes())
            if self.engine is None:
                raise ArtifactError(
                    f'Could not deserialize TensorRT engine: {self.path}')
            count = self.engine.num_optimization_profiles
            _validate_profile_index(self.profile_index, count)
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise ArtifactError('Could not create TensorRT context.')
            if count > 0:
                stream = torch.cuda.current_stream(self.device)
                accepted = self.context.set_optimization_profile_async(
                    self.profile_index, stream.cuda_stream)
                if not accepted:
                    raise ConfigurationError(
                        f'TensorRT rejected profile {self.profile_index}.')
                self._pending_event = torch.cuda.Event()
                self._pending_event.record(stream)
            self.input_names = []
            self.output_names = []
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                mode = self.engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        profile_index: int = 0,
        device: str = 'cuda:0',
        require_target_match: bool = True,
    ) -> 'TensorRTEngine':
        if type(require_target_match) is not bool:
            raise ConfigurationError('require_target_match must be a boolean.')
        root = Path(bundle_dir).expanduser().resolve()
        manifest = verify_bundle(root)
        if len(manifest.backend_files) != 1:
            raise ArtifactError(
                'TensorRT low-level runtime expects exactly one backend file.')
        if require_target_match:
            _validate_target_metadata(manifest.backend_metadata)
        relative = manifest.backend_files[0]
        return cls(
            root / relative, profile_index=profile_index, device=device,
            target_metadata=manifest.backend_metadata,
            require_target_match=require_target_match)

    def _check_target(self, expected: Mapping[str, Any]) -> None:
        # Diagnose an explicit version mismatch before querying any CUDA facts.
        expected_trt = expected.get('tensorrt_version') if isinstance(
            expected, Mapping) else None
        if expected_trt and str(expected_trt) != str(self.trt.__version__):
            raise ArtifactError(
                'TensorRT version mismatch: bundle was built with '
                f'{expected_trt}, runtime has {self.trt.__version__}.')
        _validate_target_metadata(expected)
        actual = _cuda_target_metadata(
            self.device, tensorrt_version=str(self.trt.__version__))
        for field, label in (
            ('compute_capability', 'GPU compute capability'),
            ('device', 'GPU model'),
            ('cuda_version', 'CUDA version'),
        ):
            supplied = expected[field]
            if field == 'compute_capability':
                supplied = list(supplied)
            if supplied != actual[field]:
                raise ArtifactError(
                    f'{label} mismatch: bundle was built for '
                    f'{supplied!r}, runtime has {actual[field]!r}.')

    def _torch_dtype(self, dtype: Any) -> Any:
        candidates = {
            'float32': 'float32',
            'float16': 'float16',
            'bfloat16': 'bfloat16',
            'int8': 'int8',
            'int32': 'int32',
            'int64': 'int64',
            'uint8': 'uint8',
            'bool': 'bool',
            'fp8': 'float8_e4m3fn',
        }
        for trt_name, torch_name in candidates.items():
            trt_dtype = getattr(self.trt, trt_name, None)
            torch_dtype = getattr(self.torch, torch_name, None)
            if (trt_dtype is not None and torch_dtype is not None
                    and dtype == trt_dtype):
                return torch_dtype
        raise TypeError(f'Unsupported TensorRT tensor dtype: {dtype}')

    def _input_tensor(self, name: str, value: Any) -> Any:
        expected = self._torch_dtype(self.engine.get_tensor_dtype(name))
        value = _typed_input(name, value, expected)
        return value.to(device=self.device, non_blocking=True).contiguous()

    def infer(
        self,
        inputs: Mapping[str, Any],
        synchronize: bool = False,
    ) -> Dict[str, Any]:
        if type(synchronize) is not bool:
            raise ConfigurationError('synchronize must be a boolean.')
        missing = [name for name in self.input_names if name not in inputs]
        unknown = [name for name in inputs if name not in self.input_names]
        if missing or unknown:
            raise ValueError(
                f'Engine inputs mismatch: missing={missing}, '
                f'unknown={unknown}')
        with self._lock, self.torch.cuda.device(self.device):
            if self._pending_event is not None:
                self._pending_event.synchronize()
                self._pending_event = None
                self._pending_buffers = None
            stream = self.torch.cuda.current_stream(self.device)
            bound_inputs: Dict[str, Any] = {}
            for name in self.input_names:
                tensor = self._input_tensor(name, inputs[name])
                shape = tuple(tensor.shape)
                declared = tuple(self.engine.get_tensor_shape(name))
                if len(shape) != len(declared) or any(
                        expected >= 0 and actual != expected
                        for actual, expected in zip(shape, declared)):
                    raise ValueError(
                        f'TensorRT input {name!r} shape {shape} does not '
                        f'match engine shape {declared}.')
                if -1 in declared:
                    lower, _, upper = self.engine.get_tensor_profile_shape(
                        name, self.profile_index)
                    if any(not minimum <= actual <= maximum
                           for actual, minimum, maximum
                           in zip(shape, lower, upper)):
                        raise ValueError(
                            f'TensorRT input {name!r} shape {shape} is '
                            f'outside profile {self.profile_index}: '
                            f'{lower}..{upper}.')
                if not self.context.set_input_shape(name, shape):
                    raise ValueError(
                        f'TensorRT rejected input shape {shape} for {name}.')
                bound_inputs[name] = tensor
                if not self.context.set_tensor_address(
                        name, tensor.data_ptr()):
                    raise RuntimeError(
                        f'Could not bind TensorRT input {name}.')
            unresolved = self.context.infer_shapes()
            if unresolved:
                raise ValueError(
                    f'TensorRT could not infer shapes for: {unresolved}')
            outputs: Dict[str, Any] = {}
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    raise ValueError(
                        f'Output shape is unresolved for {name}: {shape}')
                outputs[name] = self.torch.empty(
                    shape, device=self.device,
                    dtype=self._torch_dtype(
                        self.engine.get_tensor_dtype(name)))
                if not self.context.set_tensor_address(
                        name, outputs[name].data_ptr()):
                    raise RuntimeError(
                        f'Could not bind TensorRT output {name}.')
            for tensor in (*bound_inputs.values(), *outputs.values()):
                tensor.record_stream(stream)
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError('TensorRT execute_async_v3 failed.')
            self._pending_event = self.torch.cuda.Event()
            self._pending_event.record(stream)
            self._pending_buffers = (bound_inputs, outputs)
            if synchronize:
                self._pending_event.synchronize()
                self._pending_event = None
                self._pending_buffers = None
            return outputs
