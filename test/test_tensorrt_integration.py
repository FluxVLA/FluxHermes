"""Opt-in tests against an actual CUDA device and TensorRT, without mocks."""
import json
import os
import tempfile
import unittest
from pathlib import Path


def prepare_add_reference(*, deployment, device, output_dtype='bfloat16'):
    import torch

    from fluxhermes.adapters.base import PreparedModel

    class Model(torch.nn.Module):

        def forward(self, x):
            return (x + 1).to(getattr(torch, output_dtype))

    return PreparedModel(Model(), {'x': torch.ones(1, 2, device=device)}, ('y', ))


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

    def test_bf16_bundle_cli_external_weights_and_relative_manifest(self):
        import numpy as np
        import onnx
        import torch
        from onnx import TensorProto, helper, numpy_helper

        from fluxhermes.bundle import verify_bundle
        from fluxhermes.cli.run_bundle import main as run_bundle
        from fluxhermes.config import load_config
        from fluxhermes.pipeline import CompilePipeline
        from fluxhermes.runtime.tensorrt import TensorRTEngine

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config_dir = root / 'configs'
            config_dir.mkdir()
            graph = helper.make_graph(
                [
                    helper.make_node('Add', ['x', 'weight'], ['sum']),
                    helper.make_node('Cast', ['sum'], ['y'], to=TensorProto.BFLOAT16)
                ], 'bf16-output',
                [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])],
                [helper.make_tensor_value_info('y', TensorProto.BFLOAT16, [1, 2])],
                [numpy_helper.from_array(np.ones((1, 2), dtype=np.float32), 'weight')])
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid('', 22)])
            model.ir_version = 10
            onnx.save_model(
                model,
                config_dir / 'model.onnx',
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location='weights.bin',
                size_threshold=0)
            values = np.array([[1.0078125, -3.140625]], dtype=np.float32)
            np.savez(config_dir / 'inputs.npz', x=values)
            (config_dir / 'samples.json').write_text(
                json.dumps(
                    {'samples': [{
                        'id': 'config-relative',
                        'inputs': 'inputs.npz'
                    }]}),
                encoding='utf-8')
            # A same-named launch-directory manifest must not be used.
            (root / 'samples.json').write_text('{}', encoding='utf-8')
            config = dict(
                name='bf16-io-regression',
                work_dir='work',
                source=dict(
                    type='python',
                    factory=('test_tensorrt_integration:prepare_add_reference'),
                    python_paths=[str(Path(__file__).resolve().parent)]),
                inputs=[dict(name='x', dtype='float32', shape=(1, 2))],
                outputs=[dict(name='y', dtype='bfloat16', shape=(1, 2))],
                pipeline=[
                    dict(type='LoadONNX', path='model.onnx'),
                    dict(type='PrepareModel'),
                    dict(type='ConvertBackend'),
                    dict(type='ValidateBackend'),
                    dict(type='PackageBundle')
                ],
                backend_config=dict(
                    type='tensorrt',
                    precision='bf16',
                    strongly_typed=True,
                    workspace_size_gib=0.25,
                    builder_optimization_level=0,
                    allow_tf32=False),
                validation=dict(
                    samples_manifest='samples.json',
                    atol=0,
                    rtol=0,
                    min_cosine_similarity=0.999999),
                package=dict(output_dir=str(root / 'bundle'), include_onnx=True))
            config_path = config_dir / 'deploy.py'
            config_path.write_text('deployment = ' + repr(config), encoding='utf-8')
            previous = Path.cwd()
            try:
                os.chdir(root)
                bundle = CompilePipeline(load_config(config_path)).execute()
                manifest = verify_bundle(bundle)
                self.assertTrue(manifest.validation['passed'])
                self.assertEqual(manifest.validation['samples'][0]['id'],
                                 'config-relative')
                self.assertIn('model/weights.bin', manifest.files)
                onnx.load(str(bundle / 'model/model.onnx'), load_external_data=True)
                engine = TensorRTEngine(bundle / manifest.backend_files[0])
                actual = engine.infer({'x': values}, synchronize=True)['y']
                self.assertEqual(actual.dtype, torch.bfloat16)
                expected = (torch.from_numpy(values) + 1).to(
                    torch.bfloat16).float().numpy()
                output_path = root / 'outputs.npz'
                self.assertEqual(
                    run_bundle([
                        str(bundle), '--inputs',
                        str(config_dir / 'inputs.npz'), '--output',
                        str(output_path)
                    ]), 0)
                with np.load(output_path, allow_pickle=False) as archive:
                    self.assertEqual(archive['y'].dtype, np.float32)
                    np.testing.assert_array_equal(archive['y'], expected)
            finally:
                os.chdir(previous)

    def test_legacy_cli_executes_implicit_pipeline(self):
        import numpy as np

        from fluxhermes.bundle import verify_bundle
        from fluxhermes.cli.main import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = dict(
                name='legacy-implicit',
                work_dir=str(root / 'work'),
                source=dict(
                    type='python',
                    factory='test_tensorrt_integration:prepare_add_reference',
                    python_paths=[str(Path(__file__).resolve().parent)],
                    kwargs=dict(output_dtype='float32')),
                inputs=[dict(name='x', dtype='float32', shape=(1, 2))],
                outputs=[dict(name='y', dtype='float32', shape=(1, 2))],
                export=dict(opset=18, dynamo=False),
                tensorrt=dict(
                    precision='fp32',
                    workspace_size_gib=0.25,
                    builder_optimization_level=0,
                    allow_tf32=False),
                quantization=dict(mode='none'),
                validation=dict(enabled=True, atol=0, rtol=0),
                package=dict(output_dir=str(root / 'bundle'), include_onnx=True))
            config_path = root / 'legacy.py'
            config_path.write_text('deployment = ' + repr(config), encoding='utf-8')
            self.assertEqual(main(['compile', str(config_path), '--execute']), 0)
            manifest = verify_bundle(root / 'bundle')
            self.assertTrue(manifest.validation['passed'])
            np.savez(root / 'input.npz', x=np.array([[2., 3.]], dtype=np.float32))
            self.assertEqual(
                main([
                    'run',
                    str(root / 'bundle'), '--inputs',
                    str(root / 'input.npz'), '--output',
                    str(root / 'output.npz')
                ]), 0)
            self.assertEqual(main(['inspect', str(root / 'bundle')]), 0)
            with np.load(root / 'output.npz', allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['y'], [[3., 4.]])


if __name__ == '__main__':
    unittest.main()
