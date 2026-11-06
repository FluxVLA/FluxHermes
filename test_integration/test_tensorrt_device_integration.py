"""Real Torch/NPZ CPU checks; no fake CUDA engines and no skipped tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fluxhermes.backend.tensorrt.manager import TensorRTBackendManager, _build_device
from fluxhermes.backend.tensorrt.wrapper import TensorRTWrapper
from fluxhermes.builders.tensorrt import build_tensorrt_engine
from fluxhermes.contracts import DeploymentConfig
from fluxhermes.errors import ConfigurationError, DependencyError
from fluxhermes.runtime.tensorrt import (_parse_cuda_device, _resolve_cuda_device,
                                         _typed_input)


def _deployment(source_device='cuda:0', backend_device='cuda:0'):
    return DeploymentConfig.from_dict({
        'name': 'device-preflight', 'work_dir': '/tmp/device-preflight',
        'codebase_config': {
            'type': 'python', 'task': 'custom', 'device': source_device,
            'source': {'type': 'python', 'factory': 'unused:factory'},
        },
        'backend_config': {
            'type': 'tensorrt', 'precision': 'fp32', 'device': backend_device,
        },
        'inputs': [{'name': 'x', 'dtype': 'float32', 'shape': (1, 4)}],
        'outputs': [{'name': 'y', 'dtype': 'float32', 'shape': (1, 4)}],
    })


class TensorRTDeviceIntegrationTest(unittest.TestCase):

    def test_parse_accepts_only_explicit_cuda_selectors(self):
        self.assertEqual(_parse_cuda_device('cuda'), torch.device('cuda'))
        self.assertEqual(_parse_cuda_device('cuda:3'), torch.device('cuda:3'))
        self.assertEqual(_parse_cuda_device(torch.device('cuda:2')),
                         torch.device('cuda:2'))
        for value in ('cpu', 'mps', 'cuda:-1', 'cuda:abc', '', 0, True, None):
            with self.subTest(value=value):
                with self.assertRaises(ConfigurationError):
                    _parse_cuda_device(value)

    def test_real_cuda_availability_or_index_failure_is_explicit(self):
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            with self.assertRaisesRegex(ConfigurationError, 'outside'):
                _resolve_cuda_device(f'cuda:{count}')
        else:
            with self.assertRaisesRegex(DependencyError, 'available CUDA GPU'):
                _resolve_cuda_device('cuda')

    def test_non_cuda_builder_request_fails_before_loading_tensorrt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'model.plan'
            with self.assertRaisesRegex(ConfigurationError, 'must be CUDA'):
                build_tensorrt_engine(
                    Path(directory) / 'unused.onnx', output, (),
                    {'device': 'cpu', 'precision': 'fp32'})
            self.assertFalse(output.exists())

    def test_backend_source_device_mismatch_is_rejected_before_cuda_init(self):
        config = _deployment('cuda:0', 'cuda:1')
        with self.assertRaisesRegex(ConfigurationError, 'differs from source'):
            _build_device(config)
        with self.assertRaisesRegex(ConfigurationError, 'differs from source'):
            TensorRTBackendManager.to_backend(
                (Path('unused.onnx'), ), Path('/tmp/unused'), config)

    def test_source_and_backend_must_both_be_cuda(self):
        for source, backend in (('cpu', 'cuda:0'), ('cuda:0', 'cpu')):
            with self.subTest(source=source, backend=backend):
                with self.assertRaisesRegex(
                        ConfigurationError, 'must be CUDA'):
                    _build_device(_deployment(source, backend))

    def test_allow_target_mismatch_does_not_require_metadata(self):
        # An actual invalid CPU selector, not a fake engine, proves metadata
        # validation was bypassed only by the explicit False flag.
        with self.assertRaisesRegex(ConfigurationError, 'must be CUDA'):
            TensorRTWrapper('unused.plan', device='cpu', target_metadata=None,
                            require_target_match=False)

    def test_real_torch_inputs_keep_native_dtype_and_storage(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16,
                      torch.int64, torch.int32, torch.bool):
            with self.subTest(dtype=dtype):
                value = torch.ones((2, 4), dtype=dtype)
                self.assertIs(_typed_input('x', value, dtype), value)
                self.assertEqual(value.dtype, dtype)

    def test_real_npz_inputs_preserve_array_dtypes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'inputs.npz'
            np.savez(path, float_value=np.ones((2, 4), dtype=np.float32),
                     int_value=np.ones((2, 4), dtype=np.int64),
                     mask=np.ones((2, 4), dtype=np.bool_))
            with np.load(path, allow_pickle=False) as archive:
                for name, dtype in (('float_value', torch.float32),
                                    ('int_value', torch.int64),
                                    ('mask', torch.bool)):
                    value = _typed_input(name, archive[name], dtype)
                    self.assertEqual(value.dtype, dtype)
                    torch.testing.assert_close(value, torch.ones_like(value))

    def test_dtype_mismatch_is_not_silently_cast(self):
        for actual, expected in (
            (torch.float32, torch.float16),
            (torch.bfloat16, torch.float32),
            (torch.int64, torch.int32),
            (torch.bool, torch.int64),
        ):
            with self.subTest(actual=actual, expected=expected):
                value = torch.ones(2, dtype=actual)
                with self.assertRaisesRegex(
                        TypeError, "'tokens'.*expected.*actual.*forbidden"):
                    _typed_input('tokens', value, expected)
                self.assertEqual(value.dtype, actual)


if __name__ == '__main__':
    unittest.main()
