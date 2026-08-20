from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from ..bundle import verify_bundle
from ..errors import ArtifactError, DependencyError


class TensorRTEngine:
    """TensorRT 10/11 runtime using PyTorch-owned CUDA buffers and streams."""

    def __init__(
        self,
        engine_path: str | Path,
        profile_index: int = 0,
        device: str = 'cuda:0',
    ) -> None:
        try:
            import tensorrt as trt
            import torch
        except Exception as error:
            raise DependencyError(
                'TensorRT runtime requires both tensorrt and torch.') from error
        if not torch.cuda.is_available():
            raise DependencyError('TensorRT runtime requires an available CUDA GPU.')
        self.trt = trt
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError(f'TensorRT device must be CUDA, got {self.device}.')
        torch.cuda.set_device(self.device)
        self.path = Path(engine_path).expanduser().resolve()
        if not self.path.is_file():
            raise ArtifactError(f'TensorRT engine not found: {self.path}')
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise ArtifactError(f'Could not deserialize TensorRT engine: {self.path}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise ArtifactError('Could not create TensorRT execution context.')
        self.profile_index = int(profile_index)
        if self.engine.num_optimization_profiles > 0:
            self.context.active_optimization_profile = self.profile_index
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
        root = Path(bundle_dir).expanduser().resolve()
        manifest = verify_bundle(root)
        if len(manifest.backend_files) != 1:
            raise ArtifactError(
                'TensorRT low-level runtime expects exactly one backend file.')
        relative = manifest.backend_files[0]
        instance = cls(
            root / relative,
            profile_index=profile_index,
            device=device,
        )
        if require_target_match:
            instance._check_target(manifest.backend_metadata)
        return instance

    def _check_target(self, expected: Mapping[str, Any]) -> None:
        expected_trt = expected.get('tensorrt_version')
        if expected_trt and str(expected_trt) != str(self.trt.__version__):
            raise ArtifactError(
                'TensorRT version mismatch: bundle was built with '
                f'{expected_trt}, runtime has {self.trt.__version__}.')
        index = self.device.index
        if index is None:
            index = self.torch.cuda.current_device()
        expected_capability = expected.get('compute_capability')
        actual_capability = list(self.torch.cuda.get_device_capability(index))
        if expected_capability and list(expected_capability) != actual_capability:
            raise ArtifactError(
                'GPU compute capability mismatch: bundle was built for '
                f'{expected_capability}, runtime has {actual_capability}.')
        expected_device = expected.get('device')
        actual_device = self.torch.cuda.get_device_name(index)
        if expected_device and str(expected_device) != actual_device:
            raise ArtifactError(
                'GPU model mismatch: bundle was built for '
                f'{expected_device!r}, runtime has {actual_device!r}.')

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
            if trt_dtype is not None and torch_dtype is not None and dtype == trt_dtype:
                return torch_dtype
        raise TypeError(f'Unsupported TensorRT tensor dtype: {dtype}')

    def _input_tensor(self, name: str, value: Any) -> Any:
        expected = self._torch_dtype(self.engine.get_tensor_dtype(name))
        if not isinstance(value, self.torch.Tensor):
            value = self.torch.as_tensor(value)
        value = value.to(device=self.device, dtype=expected, non_blocking=True)
        return value.contiguous()

    def infer(
        self,
        inputs: Mapping[str, Any],
        synchronize: bool = False,
    ) -> Dict[str, Any]:
        missing = [name for name in self.input_names if name not in inputs]
        unknown = [name for name in inputs if name not in self.input_names]
        if missing or unknown:
            raise ValueError(
                f'Engine inputs mismatch: missing={missing}, unknown={unknown}')
        bound_inputs: Dict[str, Any] = {}
        for name in self.input_names:
            tensor = self._input_tensor(name, inputs[name])
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(
                    f'TensorRT rejected input shape {tuple(tensor.shape)} for {name}.')
            bound_inputs[name] = tensor
        unresolved = self.context.infer_shapes()
        if unresolved:
            raise ValueError(
                f'TensorRT could not infer shapes for tensors: {unresolved}')
        outputs: Dict[str, Any] = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise ValueError(f'Output shape is unresolved for {name}: {shape}')
            outputs[name] = self.torch.empty(
                shape,
                device=self.device,
                dtype=self._torch_dtype(self.engine.get_tensor_dtype(name)),
            )
        for name, tensor in {**bound_inputs, **outputs}.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f'Could not bind TensorRT tensor: {name}')
        stream = self.torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execute_async_v3 failed.')
        if synchronize:
            stream.synchronize()
        return outputs
