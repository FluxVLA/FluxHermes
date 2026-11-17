import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from fluxhermes.cli.compile import main
from fluxhermes.config import load_config
from fluxhermes.pipeline import CompilePipeline


def mismatched_reference_factory(deployment, device):
    """Deliberate negative control, executed through real TensorRT."""
    import torch

    from fluxhermes.adapters.base import PreparedModel
    return PreparedModel(
        module=torch.nn.Identity(),
        inputs={'x': torch.ones(1, 2, device=device)},
        output_names=('y', ),
        reference_runner=lambda inputs: inputs['x'] + 1)


class PipelinePlanTest(unittest.TestCase):

    def test_dry_run_does_not_import_gpu_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / 'configs/pi05/pi05_aloha_fp16.py')
        plan = CompilePipeline(config).plan_dict()
        self.assertEqual(plan['precision'], 'fp16')
        self.assertEqual(plan['stages'][0]['name'], 'prepare_model')
        self.assertEqual(plan['codebase'], 'fluxvla')
        self.assertEqual(plan['backend'], 'tensorrt')
        self.assertTrue(plan['stages'][2]['enabled'])

    @unittest.skipUnless(
        os.environ.get('FLUXHERMES_RUN_TRT_TESTS') == '1',
        'opt-in real TensorRT integration')
    def test_real_pipeline_persists_numerical_failure(self):
        from fluxhermes.errors import ValidationError
        for strict, isolated in ((True, False), (False, False), (True, True), (False,
                                                                               True)):
            with self.subTest(
                    strict=strict,
                    isolated=isolated), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / 'deploy.py'
                path.write_text(
                    'deployment = ' + repr(
                        dict(
                            name='negative-control',
                            work_dir=str(root / 'work'),
                            codebase_config=dict(
                                type='python',
                                task='custom',
                                device='cuda:0',
                                isolate_export=isolated,
                                python_paths=[str(Path(__file__).parent)],
                                source=dict(
                                    type='python',
                                    device='cuda:0',
                                    factory=f'{__name__}:mismatched_reference_factory')
                            ),
                            inputs=[dict(name='x', dtype='float32', shape=(1, 2))],
                            outputs=[dict(name='y', dtype='float32', shape=(1, 2))],
                            backend_config=dict(type='tensorrt', precision='fp32'),
                            optimization=dict(simplify=False),
                            ir_config=dict(type='onnx', opset=18, dynamo=False),
                            validation=dict(
                                atol=.02,
                                rtol=.02,
                                min_cosine_similarity=.999,
                                fail_on_error=strict),
                            package=dict(output_dir=str(root / 'bundle')),
                        )),
                    encoding='utf-8')
                pipeline = CompilePipeline(load_config(path))
                if strict:
                    with self.assertRaises(ValidationError):
                        pipeline.execute()
                    report = json.loads((root / 'work/failure.json').read_text())
                    self.assertFalse(report['workflow_completed'])
                    self.assertFalse(report['numerical_passed'])
                    self.assertFalse(report['validation']['passed'])
                    self.assertEqual(report['stages'][-1]['name'], 'validate')
                    self.assertEqual(report['stages'][-1]['status'], 'failed')
                    self.assertFalse((root / 'bundle').exists())
                else:
                    bundle = pipeline.execute()
                    report = json.loads((root / 'work/result.json').read_text())
                    self.assertTrue(report['workflow_completed'])
                    self.assertFalse(report['numerical_passed'])
                    manifest = json.loads((bundle / 'manifest.json').read_text())
                    self.assertFalse(manifest['validation']['passed'])

    def test_cli_compile_defaults_to_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'deploy.py'
            path.write_text(
                "deployment = dict(\n"
                " name='demo', work_dir='work',\n"
                " source=dict(type='python', factory='missing:factory'),\n"
                " inputs=[dict(name='x', dtype='float32', shape=(1, 2))],\n"
                " outputs=[dict(name='y', dtype='float32', shape=(1, 2))],\n"
                ")\n",
                encoding='utf-8',
            )
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([str(path)])
            self.assertEqual(code, 0)
            self.assertIn('Dry run only', output.getvalue())


if __name__ == '__main__':
    unittest.main()
