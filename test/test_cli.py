"""CLI regression tests with real bundles/NPZ and a CPU-only engine fixture."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fluxhermes.bundle import create_bundle
from fluxhermes.cli import benchmark, compile, doctor, inspect_bundle, run_bundle
from fluxhermes.contracts import BundleManifest, TensorSpec

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {
    'compile': 'fluxhermes-compile',
    'inspect_bundle': 'fluxhermes-inspect',
    'doctor': 'fluxhermes-doctor',
    'run_bundle': 'fluxhermes-run',
    'benchmark': 'fluxhermes-benchmark',
}


class NumpyEngine:
    """Exercise CLI orchestration with a simple NumPy engine fixture."""

    def __init__(self):
        self.calls = []

    def infer(self, inputs, synchronize):
        self.calls.append((inputs, synchronize))
        return {'y': inputs['x'] * 2}


class CLITest(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def invoke(self, entry, argv):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            result = entry(argv)
        return result, output.getvalue(), errors.getvalue()

    def bundle(self, validated=True):
        engine = self.root / 'source.plan'
        engine.write_bytes(b'CLI fixture, not a TensorRT engine')
        manifest = BundleManifest(
            schema_version=2,
            name='cli-demo',
            created_at=datetime.now(timezone.utc).isoformat(),
            backend='tensorrt',
            precision='fp32',
            backend_files=('backend/model.plan', ),
            backend_metadata={},
            source={},
            inputs=(TensorSpec(name='x', dtype='float32', shape=(1, 2)), ),
            outputs=(TensorSpec(name='y', dtype='float32', shape=(1, 2)), ),
            runtime={'action_output': 'y'},
            validation={'passed': validated},
            files={},
        )
        return create_bundle(self.root / 'bundle', manifest,
                             {'backend/model.plan': engine})

    def config(self):
        path = self.root / 'deploy.py'
        path.write_text(
            "deployment = dict(\n"
            " name='cli-demo', work_dir='work',\n"
            " pipeline=[dict(type='PrepareModel')],\n"
            " source=dict(type='python', factory='missing:factory'),\n"
            " inputs=[dict(name='x', dtype='float32', shape=(1, 2))],\n"
            " outputs=[dict(name='y', dtype='float32', shape=(1, 2))],\n"
            ")\n",
            encoding='utf-8',
        )
        return path

    def test_scripts_run_from_an_uninstalled_checkout_and_another_directory(self):
        for module, prog in TOOLS.items():
            with self.subTest(tool=module):
                result = subprocess.run(
                    [
                        sys.executable, '-I', '-B',
                        str(ROOT / 'scripts' / f'{module}.py'), '--help'
                    ],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'usage: {prog}', result.stdout)

    def test_module_help_does_not_import_gpu_dependencies(self):
        for module in TOOLS:
            with self.subTest(tool=module):
                code = ('import runpy, sys\n'
                        f'sys.path.insert(0, {str(ROOT)!r})\n'
                        "sys.argv = ['tool', '--help']\n"
                        'try:\n'
                        f"    runpy.run_module('fluxhermes.cli.{module}', "
                        "run_name='__main__')\n"
                        'except SystemExit as error:\n'
                        '    assert error.code == 0\n'
                        "assert not {'torch', 'tensorrt', 'onnx', 'numpy', "
                        "'fluxhermes.pipeline'} & sys.modules.keys()\n")
                result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                                        cwd=self.root,
                                        capture_output=True,
                                        text=True,
                                        timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_script_and_module_compile_have_the_same_dry_run(self):
        config = self.config()
        expected = self.invoke(compile.main, [str(config)])
        self.assertEqual(expected[0], 0)
        self.assertEqual(expected[2], '')
        self.assertIn('Dry run only', expected[1])
        for command in ([str(ROOT / 'scripts/compile.py'),
                         str(config)], ['-m', 'fluxhermes.cli.compile',
                                        str(config)]):
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, '-B', *command],
                                        cwd=ROOT,
                                        capture_output=True,
                                        text=True,
                                        timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected[1])
        self.assertFalse((self.root / 'work').exists())

    def test_compile_execute_invokes_the_pipeline(self):
        with patch(
                'fluxhermes.pipeline.CompilePipeline.execute',
                return_value=self.root / 'bundle') as execute:
            result, output, errors = self.invoke(compile.main,
                                                 [str(self.config()), '--execute'])
        execute.assert_called_once_with()
        self.assertEqual(result, 0)
        self.assertEqual(errors, '')
        self.assertIn('Pipeline completed:', output)
        self.assertNotIn('Dry run only', output)

    def test_inspect_checks_real_bundle_integrity(self):
        bundle = self.bundle()
        result, output, errors = self.invoke(inspect_bundle.main, [str(bundle)])
        self.assertEqual(result, 0, errors)
        self.assertEqual(json.loads(output)['name'], 'cli-demo')
        (bundle / 'backend/model.plan').write_bytes(b'tampered')
        result, output, errors = self.invoke(inspect_bundle.main, [str(bundle)])
        self.assertEqual(result, 2)
        self.assertEqual(output, '')
        self.assertIn('checksum mismatch', errors)
        self.assertNotIn('Traceback', errors)

    def test_doctor_preserves_the_json_report(self):
        report = {'cuda': {'available': False}, 'modules': {'torch': None}}
        with patch('fluxhermes.utils.environment_report', return_value=report):
            result, output, errors = self.invoke(doctor.main, [])
        self.assertEqual(result, 0, errors)
        self.assertEqual(json.loads(output), report)

    def test_runtime_entries_reject_unvalidated_bundles_before_loading_inputs(self):
        args = [str(self.bundle(validated=False)), '--inputs', 'missing.npz']
        for entry in (run_bundle.main, benchmark.main):
            with self.subTest(tool=entry.__module__):
                with patch('fluxhermes.apis.build_backend_wrapper') as build:
                    result, output, errors = self.invoke(entry, args)
                self.assertEqual(result, 2)
                self.assertEqual(output, '')
                self.assertIn('validated', errors)
                self.assertNotIn('missing.npz', errors)
                build.assert_not_called()

    def test_run_round_trips_npz_and_forwards_runtime_options(self):
        bundle = self.bundle(validated=False)
        sample, output = self.root / 'inputs.npz', self.root / 'nested/outputs.npz'
        inputs = np.array([[1., 3.]], dtype=np.float32)
        np.savez(sample, x=inputs)
        args = [
            str(bundle), '--inputs',
            str(sample), '--output',
            str(output), '--profile', '2', '--device', 'cuda:1', '--allow-unvalidated',
            '--allow-target-mismatch'
        ]
        engine = NumpyEngine()
        with patch(
                'fluxhermes.apis.build_backend_wrapper', return_value=engine) as build:
            result, _, errors = self.invoke(run_bundle.main, args)
        self.assertEqual(result, 0, errors)
        with np.load(output, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['y'], inputs * 2)
        self.assertEqual(len(engine.calls), 1)
        self.assertTrue(engine.calls[0][1])
        self.assertEqual(build.call_args.kwargs, {
            'profile_index': 2,
            'device': 'cuda:1',
            'require_target_match': False
        })

    def test_benchmark_preserves_warmup_synchronization_and_latency_statistics(self):
        sample = self.root / 'inputs.npz'
        np.savez(sample, x=np.ones((1, 2), dtype=np.float32))
        args = [
            str(self.bundle()), '--inputs',
            str(sample), '--warmup', '2', '--repeats', '3'
        ]
        engine = NumpyEngine()
        with patch(
                'fluxhermes.apis.build_backend_wrapper',
                return_value=engine) as build, patch(
                    'fluxhermes.cli.benchmark.time.perf_counter',
                    side_effect=[0., .001, .002, .004, .005, .008]):
            result, output, errors = self.invoke(benchmark.main, args)
        self.assertEqual(result, 0, errors)
        report = json.loads(output)
        self.assertEqual((report['warmup'], report['repeats']), (2, 3))
        for key, value in {
                'mean_ms': 2,
                'p50_ms': 2,
                'p90_ms': 3,
                'p99_ms': 3,
                'min_ms': 1,
                'max_ms': 3
        }.items():
            self.assertAlmostEqual(report[key], value)
        self.assertEqual(len(engine.calls), 5)
        self.assertTrue(all(synchronized for _, synchronized in engine.calls))
        self.assertEqual(build.call_args.kwargs, {
            'profile_index': 0,
            'device': 'cuda:0',
            'require_target_match': True
        })


if __name__ == '__main__':
    unittest.main()
