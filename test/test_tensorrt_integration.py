"""Opt-in tests against an actual CUDA device and TensorRT, without mocks."""
import os
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    os.environ.get('FLUXHERMES_RUN_TRT_TESTS') == '1',
    'Set FLUXHERMES_RUN_TRT_TESTS=1 for real GPU integration')
class TensorRTIntegrationTests(unittest.TestCase):

    def test_strong_bf16_engine_profile_and_inference(self):
        import onnx
        import torch
        from onnx import TensorProto, helper

        from fluxhermes.builders.tensorrt import build_tensorrt_engine
        from fluxhermes.contracts import TensorSpec
        from fluxhermes.runtime.tensorrt import TensorRTEngine

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = helper.make_graph(
                [
                    helper.make_node('Cast', ['x'], ['b'], to=TensorProto.BFLOAT16),
                    helper.make_node('Cast', ['b'], ['y'], to=TensorProto.FLOAT),
                ], 'bf16-roundtrip',
                [helper.make_tensor_value_info('x', TensorProto.FLOAT, ['batch', 2])],
                [helper.make_tensor_value_info('y', TensorProto.FLOAT, ['batch', 2])])
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid('', 22)])
            model.ir_version = 10
            onnx.save(model, root / 'model.onnx')
            details = build_tensorrt_engine(
                root / 'model.onnx', root / 'model.engine',
                [TensorSpec('x', 'float32', (-1, 2), (1, 2), (2, 2), (4, 2))],
                dict(
                    precision='bf16',
                    strongly_typed=True,
                    allow_tf32=False,
                    workspace_size_gib=0.25,
                    tactic_dram_size_gib=1,
                    builder_optimization_level=0))
            self.assertEqual(details['tactic_dram_size_gib'], 1)
            self.assertEqual(details['builder_optimization_level'], 0)
            self.assertTrue(details['strongly_typed'])
            self.assertFalse(details['allow_tf32'])
            runtime = TensorRTEngine(root / 'model.engine')
            sample = torch.tensor([[1.25, 2.5]], device='cuda:0')
            for batch in (1, 2, 4):
                inputs = sample.repeat(batch, 1)
                result = runtime.infer({'x': inputs}, synchronize=True)['y']
                torch.testing.assert_close(result, inputs, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'Invalid TensorRT profile'):
                TensorRTEngine(root / 'model.engine', profile_index=1)


if __name__ == '__main__':
    unittest.main()
