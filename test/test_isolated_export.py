"""Real opt-in BF16 representative-input round trip through the export worker."""
import json
import os
import tempfile
import unittest
from pathlib import Path


def bf16_identity_factory(deployment, device):
    import torch

    from fluxhermes.adapters.base import PreparedModel

    # Scaled 5-12-13 vector: exact norm 3.25 avoids cosine roundoff at the
    # strict 1.0 test threshold. Full-model acceptance thresholds stay unchanged.
    return PreparedModel(
        module=torch.nn.Identity(),
        inputs={'x': torch.tensor([[1.25, -3.0]], dtype=torch.bfloat16, device=device)},
        output_names=('y', ))


@unittest.skipUnless(
    os.environ.get('FLUXHERMES_RUN_TRT_TESTS') == '1',
    'Opt-in real CUDA and TensorRT integration')
class IsolatedExportTest(unittest.TestCase):

    def test_bf16_input_snapshot_and_positive_parity(self):
        import numpy as np

        from fluxhermes.config import load_config
        from fluxhermes.pipeline import CompilePipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'deploy.py'
            config.write_text(
                'deployment = ' + repr(
                    dict(
                        name='bf16-worker-roundtrip',
                        work_dir=str(root / 'work'),
                        codebase_config=dict(
                            type='python',
                            task='custom',
                            device='cuda:0',
                            isolate_export=True,
                            python_paths=[str(Path(__file__).parent)],
                            source=dict(
                                type='python',
                                factory=f'{__name__}:bf16_identity_factory')),
                        inputs=[dict(name='x', dtype='bfloat16', shape=(1, 2))],
                        outputs=[dict(name='y', dtype='bfloat16', shape=(1, 2))],
                        ir_config=dict(type='onnx', opset=18, dynamo=False),
                        backend_config=dict(
                            type='tensorrt',
                            precision='bf16',
                            strongly_typed=True,
                            allow_tf32=False,
                            tactic_dram_size_gib=1),
                        optimization=dict(simplify=False),
                        validation=dict(
                            atol=0, rtol=0, min_cosine_similarity=1,
                            fail_on_error=True),
                        package=dict(output_dir=str(root / 'bundle')),
                    )),
                encoding='utf-8')
            bundle = CompilePipeline(load_config(config)).execute()
            report = json.loads((bundle / 'manifest.json').read_text())
            self.assertTrue(report['validation']['passed'])
            state = json.loads((root / 'work/export-stage.json').read_text())
            self.assertEqual(state['input_dtypes'], {'x': 'torch.bfloat16'})
            self.assertNotEqual(state['pid'], os.getpid())
            with np.load(root / 'work/export-inputs.npz', allow_pickle=False) as arrays:
                self.assertEqual(arrays['x'].dtype, np.float32)
                np.testing.assert_array_equal(arrays['x'], [[1.25, -3.0]])


class ExportWorkerCPUTest(unittest.TestCase):

    def test_bf16_snapshot_in_a_real_export_process(self):
        import importlib.util
        import subprocess
        import sys
        if any(importlib.util.find_spec(name) is None for name in ('torch', 'onnx')):
            self.skipTest('CPU export dependencies are optional')
        import numpy as np
        import torch

        from fluxhermes.contracts import DeploymentConfig
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = DeploymentConfig.from_dict(
                dict(
                    name='cpu-bf16-snapshot',
                    work_dir=str(root),
                    codebase_config=dict(
                        type='python',
                        task='custom',
                        device='cpu',
                        python_paths=[str(Path(__file__).parent)],
                        source=dict(
                            type='python',
                            factory=f'{__name__}:bf16_identity_factory')),
                    inputs=[dict(name='x', dtype='bfloat16', shape=(1, 2))],
                    outputs=[dict(name='y', dtype='bfloat16', shape=(1, 2))],
                    ir_config=dict(type='onnx', opset=18, dynamo=False),
                    backend_config=dict(
                        type='tensorrt', precision='bf16', strongly_typed=True),
                    optimization=dict(simplify=False)))
            resolved = root / 'resolved_config.json'
            resolved.write_text(json.dumps(config.to_dict()))
            process = subprocess.run([
                sys.executable, '-B', '-m', 'fluxhermes.export_worker',
                str(resolved), 'cpu-snapshot-test'
            ],
                                     capture_output=True,
                                     text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            state = json.loads((root / 'export-stage.json').read_text())
            self.assertTrue(state['passed'])
            self.assertEqual(state['input_dtypes'], {'x': 'torch.bfloat16'})
            self.assertNotEqual(state['pid'], os.getpid())
            with np.load(root / 'export-inputs.npz', allow_pickle=False) as arrays:
                self.assertEqual(arrays['x'].dtype, np.float32)
                restored = torch.from_numpy(arrays['x']).to(torch.bfloat16)
                expected = torch.tensor([[1.25, -3.0]], dtype=torch.bfloat16)
                self.assertTrue(torch.equal(restored, expected))


class MemoryBudgetOptionTest(unittest.TestCase):

    def test_invalid_optimization_levels_fail_before_loading_tensorrt(self):
        from fluxhermes.builders.tensorrt import build_tensorrt_engine
        from fluxhermes.errors import ConfigurationError
        for value in (-1, 6, True, '0', 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ConfigurationError, 'builder_optimization_level'):
                build_tensorrt_engine('unused.onnx', 'unused.engine', [],
                                      {'builder_optimization_level': value})

    def test_invalid_tactic_budgets_fail_before_loading_tensorrt(self):
        from fluxhermes.builders.tensorrt import build_tensorrt_engine
        from fluxhermes.errors import ConfigurationError
        for value in (0, -1, float('nan'), float('inf'), True, '8'):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ConfigurationError, 'tactic_dram_size_gib'):
                build_tensorrt_engine('unused.onnx', 'unused.engine', [],
                                      {'tactic_dram_size_gib': value})


if __name__ == '__main__':
    unittest.main()
