"""Actual CLI execution of compiled PI0.5 ONNX Runtime bundles and fixtures."""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import subprocess
import sys
import unittest
import zipfile

import numpy as np
import test_multigraph_pipeline as pipeline_fixtures
import torch
from safetensors.torch import save_file

from fluxhermes.cli.main import _load_inputs
from fluxhermes.errors import FluxHermesError


class CLIExportRuntimeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        # Reuse actual compilation/fixture setup only. Importing the module,
        # rather than its TestCase, avoids duplicating discovered tests.
        pipeline_fixtures.MultiGraphPipelineTest.setUpClass.__func__(cls)
        cls.npz_path = cls.root / 'inputs.npz'
        np.savez(cls.npz_path, **{
            name: tensor.numpy()
            for name, tensor in cls.tensors.items()
        })

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, '-I', '-m', 'fluxhermes.cli.main', *map(str, args)],
            cwd=self.root,
            env={
                **os.environ, 'HF_HUB_OFFLINE': '1',
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1'
            },
            text=True,
            capture_output=True,
            check=False,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def assert_error(self, result, message):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def comparison(self, fixture=None, config=None, bundle=None):
        return self.cli('benchmark', self.bundle if bundle is None else bundle,
                        '--inputs', self.inputs_path if fixture is None else fixture,
                        '--device', 'cpu', '--warmup', 1, '--repeats', 2,
                        '--reference-config',
                        self.config.config_path if config is None else config)

    def bundle_with_metadata(self, name, manifest):
        destination = self.root / name
        shutil.copytree(self.bundle, destination)
        (destination / 'manifest.json').write_text(
            json.dumps(manifest), encoding='utf-8')
        return destination

    def reference_config(self, name, document):
        sdk_config = self.root / (name + '.json')
        sdk_config.write_text(json.dumps(document, indent=4), encoding='utf-8')
        config = copy.deepcopy(self.config.to_dict())
        config['codebase_config']['source']['deployment_config'] = str(sdk_config)
        hermes_config = self.root / (name + '.py')
        hermes_config.write_text(
            'deployment = ' + repr(config) + '\n', encoding='utf-8')
        return hermes_config

    def test_reference_comparison_measures_policy_parity_and_ratio_for_both_formats(
            self):
        for fixture in (self.npz_path, self.inputs_path):
            with self.subTest(format=fixture.suffix):
                result = self.comparison(fixture=fixture)
                self.assert_success(result)
                report = json.loads(result.stdout)
                reference = report['reference']
                comparison = report['comparison']
                self.assertEqual(reference['backend'], 'pytorch')
                self.assertEqual(reference['device'], report['device'])
                self.assertEqual(reference['warmup'], report['warmup'])
                self.assertEqual(reference['repeats'], report['repeats'])
                self.assertEqual(reference['timing_scope'], report['timing_scope'])
                self.assertTrue(reference['synchronize'])
                self.assertTrue(reference['inputs_preloaded'])
                self.assertFalse(reference['model_loading_included'])
                self.assertFalse(reference['host_to_device_transfer_included'])
                self.assertEqual(reference['torch_num_threads'], 1)
                self.assertTrue(comparison['parity']['passed'])
                self.assertEqual(comparison['parity']['atol'],
                                 self.manifest.validation['atol'])
                self.assertTrue(comparison['reference_released_before_backend_load'])
                self.assertEqual(comparison['measurement_order'],
                                 ['reference', 'backend'])
                for name in ('recipe', 'source_config_sha256', 'checkpoint_sha256',
                             'inputs_sha256'):
                    self.assertEqual(reference['provenance'][name],
                                     self.manifest.source[name])
                for field in ('mean_ms', 'p50_ms', 'p90_ms', 'p99_ms', 'min_ms',
                              'max_ms'):
                    self.assertGreater(reference[field], 0)
                ratio = comparison['reference_over_backend_mean_ratio']
                self.assertGreater(ratio, 0)
                self.assertEqual(ratio, reference['mean_ms'] / report['mean_ms'])

    def test_reference_comparison_requires_exact_cli_fixture_for_unused_state_values(
            self):
        for variant in ('values', 'dtype', 'names'):
            with self.subTest(variant=variant):
                values = {
                    name: value.numpy().copy()
                    for name, value in self.tensors.items()
                }
                if variant == 'values':
                    # PI0.5 ignores state values, but a fixed-fixture timing
                    # comparison must still reject different user input data.
                    values['states'][0, 0] += 1
                    message = 'values/device differ'
                elif variant == 'dtype':
                    values['images'] = values['images'].astype(np.float64)
                    message = 'dtype/shape differs'
                else:
                    del values['states']
                    message = 'input names differ'
                path = self.root / f'reference-mismatch-{variant}.npz'
                np.savez(path, **values)
                result = self.comparison(fixture=path)
                self.assert_error(result, message)
                self.assertEqual(result.stdout, '')

    def test_reference_comparison_rejects_semantically_changed_source_config(self):
        document = copy.deepcopy(self.document)
        document['inference']['num_steps'] += 1
        config = self.reference_config('changed-source-config', document)
        result = self.comparison(config=config)
        self.assert_error(result, 'source provenance mismatch for source_config_sha256')
        self.assertEqual(result.stdout, '')

    def test_reference_comparison_accepts_canonical_equivalent_config_formatting(self):
        document = dict(reversed(tuple(self.document.items())))
        config = self.reference_config('reformatted-source-config', document)
        result = self.comparison(config=config)
        self.assert_success(result)
        self.assertTrue(json.loads(result.stdout)['comparison']['parity']['passed'])

    def test_reference_comparison_checks_all_four_source_provenance_bindings(self):
        for field in ('recipe', 'source_config_sha256', 'checkpoint_sha256',
                      'inputs_sha256'):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest.to_dict())
                manifest['source'][
                    field] = 'wrong-recipe' if field == 'recipe' else 'f' * 64
                bundle = self.bundle_with_metadata('wrong-provenance-' + field,
                                                   manifest)
                result = self.comparison(bundle=bundle)
                self.assert_error(result, 'source provenance mismatch for ' + field)
                self.assertEqual(result.stdout, '')
        manifest = copy.deepcopy(self.manifest.to_dict())
        del manifest['source']['source_config_sha256']
        bundle = self.bundle_with_metadata('missing-source-provenance', manifest)
        self.assert_error(
            self.comparison(bundle=bundle), 'requires source_config_sha256')

    def test_reference_comparison_requires_recorded_tolerances_and_actual_parity(self):
        manifest = copy.deepcopy(self.manifest.to_dict())
        del manifest['validation']['rtol']
        bundle = self.bundle_with_metadata('missing-parity-tolerance', manifest)
        self.assert_error(self.comparison(bundle=bundle), 'validation.rtol')
        # Real ORT and PyTorch agree within deployment tolerance, not bitwise.
        # Tightening the recorded tolerance exercises genuine parity failure,
        # without replacing either runtime or manufacturing candidate outputs.
        self.assertGreater(
            self.manifest.validation['outputs']['actions']['max_abs_error'], 0)
        manifest = copy.deepcopy(self.manifest.to_dict())
        manifest['validation'].update(atol=0, rtol=0, min_cosine_similarity=1)
        bundle = self.bundle_with_metadata('strict-parity-tolerance', manifest)
        result = self.comparison(bundle=bundle)
        self.assert_error(result, 'Reference/backend policy parity failed')
        self.assertEqual(result.stdout, '')

    def test_actual_reference_modules_are_freed_before_backend_construction(self):
        # Observe actual function return/call events without replacing any
        # factory, model, execution method, tensor, or timing implementation.
        script = '''
import sys
import weakref
from fluxhermes.cli.main import main
state = {'references': [], 'source_calls': 0, 'backend_loaded': False}
def observe(frame, event, value):
    name = frame.f_code.co_name
    if (event == 'return' and name == 'prepare_model'
            and type(value).__name__ == 'PreparedDeployment'):
        state['references'].append(weakref.ref(value))
        for graph in value.graphs.values():
            state['references'].extend(
                weakref.ref(module) for module in graph.module.modules())
    if (event == 'call' and name == 'run_reference'
            and type(frame.f_locals.get('self')).__name__ == 'PreparedDeployment'):
        state['source_calls'] += 1
    if event == 'call' and name == 'build_backend_wrapper':
        assert state['source_calls'] == 4, state
        assert state['references'] and all(ref() is None for ref in state['references'])
        state['backend_loaded'] = True
sys.setprofile(observe)
try:
    result = main(sys.argv[1:])
finally:
    sys.setprofile(None)
assert result == 0
assert state['backend_loaded']
'''
        result = subprocess.run(
            [
                sys.executable,
                '-I',
                '-c',
                script,
                'benchmark',
                str(self.bundle),
                '--inputs',
                str(self.inputs_path),
                '--device',
                'cpu',
                '--warmup',
                '1',
                '--repeats',
                '2',
                '--reference-config',
                str(self.config.config_path),
            ],
            cwd=self.root,
            env={
                **os.environ, 'HF_HUB_OFFLINE': '1',
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1'
            },
            text=True,
            capture_output=True,
            check=False)
        self.assert_success(result)
        self.assertTrue(json.loads(result.stdout)['comparison']['parity']['passed'])

    def test_run_npz_and_safetensors_match_actual_reference_without_mutation(self):
        for fixture in (self.npz_path, self.inputs_path):
            with self.subTest(format=fixture.suffix):
                output = self.root / f'cli-{fixture.suffix[1:]}.npz'
                before = fixture.read_bytes()
                result = self.cli('run', self.bundle, '--inputs', fixture, '--output',
                                  output, '--device', 'cpu')
                self.assert_success(result)
                self.assertIn(str(output), result.stdout)
                with np.load(output, allow_pickle=False) as archive:
                    self.assertEqual(archive.files, ['actions'])
                    self.assertEqual(archive['actions'].dtype, np.float32)
                    torch.testing.assert_close(
                        torch.from_numpy(archive['actions']),
                        self.expected,
                        atol=1e-5,
                        rtol=1e-5)
                self.assertEqual(before, fixture.read_bytes())

    def test_run_retains_npz_suffix_and_existing_runtime_flags(self):
        output = self.root / 'output-without-suffix'
        result = self.cli('run', self.bundle, '--inputs', self.npz_path, '--output',
                          output, '--device', 'cpu', '--profile', 0,
                          '--allow-target-mismatch', '--allow-unvalidated')
        self.assert_success(result)
        actual_path = output.with_name(output.name + '.npz')
        self.assertTrue(actual_path.is_file())
        self.assertIn(str(actual_path), result.stdout)

    def test_benchmark_reports_actual_synchronized_preloaded_policy_scope(self):
        for fixture in (self.npz_path, self.inputs_path):
            with self.subTest(format=fixture.suffix):
                result = self.cli('benchmark', self.bundle, '--inputs', fixture,
                                  '--device', 'cpu', '--warmup', 1, '--repeats', 3)
                self.assert_success(result)
                report = json.loads(result.stdout)
                self.assertEqual(report['backend'], 'onnxruntime')
                self.assertEqual(report['device'], 'cpu')
                self.assertEqual(report['profile_index'], 0)
                self.assertEqual(report['input_format'], fixture.suffix[1:])
                self.assertEqual(report['timing_scope'],
                                 'synchronized_policy_inference')
                self.assertTrue(report['synchronize'])
                self.assertTrue(report['inputs_preloaded'])
                self.assertFalse(report['host_to_device_transfer_included'])
                self.assertFalse(report['input_loading_included'])
                self.assertFalse(report['backend_loading_included'])
                self.assertEqual(report['warmup'], 1)
                self.assertEqual(report['repeats'], 3)
                for key in ('mean_ms', 'p50_ms', 'p90_ms', 'p99_ms', 'min_ms',
                            'max_ms'):
                    self.assertGreater(report[key], 0)
                self.assertLessEqual(report['min_ms'], report['mean_ms'])
                self.assertLessEqual(report['mean_ms'], report['max_ms'])

    def test_benchmark_allows_zero_warmup_and_rejects_invalid_counts_early(self):
        result = self.cli('benchmark', self.bundle, '--inputs', self.inputs_path,
                          '--device', 'cpu', '--warmup', 0, '--repeats', 1)
        self.assert_success(result)
        self.assertEqual(json.loads(result.stdout)['warmup'], 0)
        for warmup, repeats, message in ((-1, 1, '--warmup'), (0, 0, '--repeats'),
                                         (1, -3, '--repeats')):
            with self.subTest(warmup=warmup, repeats=repeats):
                result = self.cli('benchmark', self.root / 'nonexistent-bundle',
                                  '--inputs', self.inputs_path, '--device', 'cpu',
                                  '--warmup', warmup, '--repeats', repeats)
                self.assert_error(result, message)

    def test_fixture_loader_preserves_names_dtypes_values_and_cpu_device(self):
        tensors = {
            'bool_values': torch.tensor([True, False]),
            'int64_values': torch.tensor([0, 2**40], dtype=torch.int64),
            'float32_values': torch.tensor([0.25, -1.5], dtype=torch.float32),
            'float64_values': torch.tensor([0.125, -2.5], dtype=torch.float64),
            'float16_values': torch.tensor([0.5], dtype=torch.float16),
            'scalar': torch.tensor(3.25, dtype=torch.float32),
        }
        npz = self.root / 'typed-fixture.npz'
        safe = self.root / 'typed-fixture.safetensors'
        np.savez(npz, **{name: tensor.numpy() for name, tensor in tensors.items()})
        save_file(tensors, safe)
        for path in (npz, safe):
            with self.subTest(format=path.suffix):
                actual = _load_inputs(str(path), device='cpu')
                self.assertEqual(set(actual), set(tensors))
                for name, expected in tensors.items():
                    self.assertEqual(actual[name].dtype, expected.dtype)
                    self.assertEqual(actual[name].device, torch.device('cpu'))
                    torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)

    def test_safetensors_bfloat16_is_preserved_not_silently_converted(self):
        path = self.root / 'native-bfloat16.safetensors'
        expected = {'native_bf16': torch.tensor([1.0, -0.5], dtype=torch.bfloat16)}
        save_file(expected, path)
        actual = _load_inputs(str(path), device='cpu')
        self.assertEqual(actual['native_bf16'].dtype, torch.bfloat16)
        torch.testing.assert_close(
            actual['native_bf16'], expected['native_bf16'], rtol=0, atol=0)
        invalid = dict(self.tensors)
        invalid['images'] = invalid['images'].bfloat16()
        save_file(invalid, path)
        result = self.cli('run', self.bundle, '--inputs', path, '--device', 'cpu',
                          '--output', self.root / 'bad-bf16.npz')
        self.assert_error(result, 'dtype')

    def test_cli_rejects_wrong_input_names_dtypes_and_tokens(self):
        for variant in ('missing', 'extra', 'dtype', 'vocabulary'):
            with self.subTest(variant=variant):
                values = {
                    name: tensor.numpy().copy()
                    for name, tensor in self.tensors.items()
                }
                if variant == 'missing':
                    del values['noise']
                    message = 'missing'
                elif variant == 'extra':
                    values['extra'] = np.zeros((1, ), dtype=np.float32)
                    message = 'unknown'
                elif variant == 'dtype':
                    values['img_masks'] = values['img_masks'].astype(np.int64)
                    message = 'dtype'
                else:
                    values['lang_tokens'][0, -1] = 64
                    message = 'lang_tokens'
                fixture = self.root / f'invalid-{variant}.npz'
                output = self.root / f'invalid-{variant}-output.npz'
                np.savez(fixture, **values)
                result = self.cli('run', self.bundle, '--inputs', fixture, '--device',
                                  'cpu', '--output', output)
                self.assert_error(result, message)
                self.assertFalse(output.exists())

    def test_loader_rejects_object_arrays_duplicate_names_and_malformed_archives(self):
        object_path = self.root / 'object.npz'
        np.savez(object_path, unsafe=np.array([{'value': 1}], dtype=object))
        with self.assertRaisesRegex(FluxHermesError, 'allow_pickle=False'):
            _load_inputs(str(object_path), device='cpu')
        duplicate = self.root / 'duplicate.npz'
        data = io.BytesIO()
        np.save(data, np.array([1.0], dtype=np.float32), allow_pickle=False)
        with zipfile.ZipFile(duplicate, 'w') as archive:
            archive.writestr('value.npy', data.getvalue())
            with self.assertWarnsRegex(UserWarning, 'Duplicate name'):
                archive.writestr('value.npy', data.getvalue())
        with self.assertRaisesRegex(FluxHermesError, 'unique'):
            _load_inputs(str(duplicate), device='cpu')
        for suffix, content in (('.npz', b'PK\x03\x04truncated'), ('.safetensors',
                                                                   b'truncated')):
            path = self.root / ('malformed' + suffix)
            path.write_bytes(content)
            result = self.cli('run', self.bundle, '--inputs', path, '--device', 'cpu')
            self.assert_error(result, 'cannot load')

    def test_loader_device_format_and_unsupported_dtype_errors_are_actionable(self):
        with self.assertRaisesRegex(FluxHermesError, 'device'):
            _load_inputs(str(self.npz_path), device='meta')
        with self.assertRaisesRegex(FluxHermesError, '.npz or .safetensors'):
            _load_inputs(str(self.root / 'unsupported.txt'), device='cpu')
        strings = self.root / 'strings.npz'
        np.savez(strings, text=np.array(['not-a-numeric-tensor']))
        with self.assertRaisesRegex(FluxHermesError, 'NumPy dtype'):
            _load_inputs(str(strings), device='cpu')
        result = self.cli('run', self.bundle, '--inputs',
                          self.root / 'missing.safetensors', '--device', 'cpu')
        self.assert_error(result, 'cannot load')

    def test_fresh_cli_process_does_not_import_models_or_training_stack(self):
        script = '''
import sys
from fluxhermes.cli.main import main
for module in ('torch', 'transformers', 'fluxvla'):
    assert module not in sys.modules, module
result = main(sys.argv[1:])
assert result == 0
for module in ('fluxvla', 'mmengine', 'transformers', 'timm',
               'fluxvla_export.recipes.pi05.model',
               'fluxvla_export.recipes.pi05.reference'):
    assert module not in sys.modules, module
'''
        result = subprocess.run(
            [
                sys.executable,
                '-I',
                '-c',
                script,
                'run',
                str(self.bundle),
                '--inputs',
                str(self.inputs_path),
                '--device',
                'cpu',
                '--output',
                str(self.root / 'isolated-cli.npz'),
            ],
            cwd=self.root,
            env={
                **os.environ, 'HF_HUB_OFFLINE': '1',
                'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1'
            },
            text=True,
            capture_output=True,
            check=False)
        self.assert_success(result)


if __name__ == '__main__':
    unittest.main()
