"""Real CUDA/TensorRT target smoke, never mocked and never skipped.

Run on the target host, preferably selecting a non-current GPU:
    FLUXHERMES_CUDA_DEVICE=cuda:1 python test_gpu/test_tensorrt_smoke.py -v
Requires installed FluxHermes, torch, onnx, TensorRT and an actual CUDA GPU.
This test is separate from dependency-free and CPU integration discovery.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import onnx
import torch
from onnx import TensorProto, helper

from fluxhermes.builders.tensorrt import build_tensorrt_engine
from fluxhermes.contracts import TensorSpec
from fluxhermes.errors import ConfigurationError
from fluxhermes.runtime.tensorrt import (TensorRTEngine, _resolve_cuda_device,
                                         _validate_target_metadata)


class TensorRTTargetSmokeTest(unittest.TestCase):

    def test_real_build_runtime_device_stream_and_profile_safety(self):
        # Explicit invocation must FAIL if target dependencies are unavailable.
        target = _resolve_cuda_device(
            os.environ.get('FLUXHERMES_CUDA_DEVICE', 'cuda:0'))
        other = ((target.index + 1) % torch.cuda.device_count())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = helper.make_graph(
                [helper.make_node('Add', ['x', 'x'], ['y'])], 'double',
                [helper.make_tensor_value_info(
                    'x', TensorProto.FLOAT, ['batch', 4])],
                [helper.make_tensor_value_info(
                    'y', TensorProto.FLOAT, ['batch', 4])])
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid('', 18)],
                ir_version=10)
            onnx.checker.check_model(model)
            onnx_path = root / 'double.onnx'
            onnx.save(model, onnx_path)
            spec = TensorSpec(
                'x', 'float32', (-1, 4), min_shape=(1, 4),
                opt_shape=(2, 4), max_shape=(4, 4))
            original = torch.cuda.current_device()
            with torch.cuda.device(other):
                details = build_tensorrt_engine(
                    onnx_path, root / 'model.plan', (spec, ), {
                        'device': str(target), 'precision': 'fp32',
                        'tf32': False, 'workspace_size_gib': 1,
                    })
                self.assertEqual(torch.cuda.current_device(), other)
                self.assertEqual(details['device_index'], target.index)
                self.assertEqual(details['device'],
                                 torch.cuda.get_device_name(target.index))
                self.assertFalse(details['tf32'])
                _validate_target_metadata(details)
                engine = TensorRTEngine(
                    root / 'model.plan', device=str(target),
                    target_metadata=details, require_target_match=True)
                self.assertEqual(torch.cuda.current_device(), other)
                self.assertEqual(engine.engine.num_optimization_profiles, 1)
                bounds = engine.engine.get_tensor_profile_shape('x', 0)
                self.assertEqual(
                    tuple(tuple(int(dim) for dim in shape)
                          for shape in bounds),
                    ((1, 4), (2, 4), (4, 4)))
                self.assertTrue(Path(details['timing_cache']).is_file())
                engine._check_target(details)
                self.assertEqual(torch.cuda.current_device(), other)
                with self.assertRaisesRegex(ConfigurationError, 'profile'):
                    TensorRTEngine(root / 'model.plan', device=str(target),
                                   profile_index=1)
                self.assertEqual(torch.cuda.current_device(), other)
            self.assertEqual(torch.cuda.current_device(), original)
            with torch.cuda.device(target):
                unindexed = TensorRTEngine(root / 'model.plan', device='cuda')
            self.assertEqual(unindexed.device, target)
            first = torch.cuda.Stream(device=target)
            second = torch.cuda.Stream(device=target)
            results = []
            for stream, batch in ((first, 2), (second, 4)):
                with torch.cuda.stream(stream), torch.cuda.device(other):
                    result = unindexed.infer(
                        {'x': torch.ones(batch, 4)}, synchronize=False)
                    self.assertEqual(torch.cuda.current_device(), other)
                    self.assertEqual(result['y'].device, target)
                    results.append((result['y'], batch))
            # The second enqueue must not overwrite or race the first context.
            second.synchronize()
            for output, batch in results:
                torch.testing.assert_close(output.cpu(),
                                           torch.full((batch, 4), 2.0))
            with self.assertRaisesRegex(TypeError, 'dtype mismatch'):
                unindexed.infer({'x': torch.ones(2, 4, dtype=torch.float16)})
            with self.assertRaisesRegex(ValueError, 'outside profile'):
                unindexed.infer({'x': torch.ones(5, 4)})
            self.assertEqual(torch.cuda.current_device(), original)


if __name__ == '__main__':
    unittest.main()
