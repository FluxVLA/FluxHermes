"""CPU regression tests for diagnostics, not full-model/build integration.

Pipeline fixtures explicitly replace export/backend/package boundaries; numerical
verdicts still come from real NumPy comparisons. Optional runner tests use real
CPU Torch tensors and skip when Torch is absent, never emulating Torch.
"""
import importlib.util
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import numpy as np

from fluxhermes.adapters.base import PreparedModel
from fluxhermes.config import load_config
from fluxhermes.errors import ValidationError
from fluxhermes.pipeline import CompilePipeline
from fluxhermes.validation.parity import compare_outputs, validate_backend

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


def numeric_report(passed=True):
    return compare_outputs({'y': np.array([1., 2.])},
                           {'y': np.array([1., 2.]) + (0 if passed else 1)},
                           atol=0.,
                           rtol=0.,
                           min_cosine_similarity=-1.)


class CompareDiagnosticsTest(unittest.TestCase):

    def test_finite_metrics_are_computed_from_actual_arrays(self):
        report = numeric_report(False)
        self.assertFalse(report['passed'])
        metrics = report['outputs']['y']
        self.assertEqual(metrics['max_abs_error'], 1.)
        self.assertEqual(metrics['max_rel_error'], 1.)
        self.assertAlmostEqual(metrics['cosine_similarity'], 8. / np.sqrt(65.))
        self.assertTrue(metrics['finite'])
        json.dumps(report, allow_nan=False)

    def test_invalid_thresholds_attach_structured_report(self):
        for key in ('atol', 'rtol', 'min_cosine_similarity'):
            values = [float('inf'), -float('inf'), float('nan')]
            values += [-.01] if key != 'min_cosine_similarity' else [-1.01, 1.01]
            for value in values:
                options = dict(atol=0., rtol=0., min_cosine_similarity=1.)
                options[key] = value
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValidationError) as caught:
                        compare_outputs({'y': np.ones(2)}, {'y': np.ones(2)}, **options)
                    report = caught.exception.report
                    self.assertIs(report['passed'], False)
                    self.assertEqual(report['outputs'], {})
                    self.assertEqual(report['error']['type'], 'ValidationError')
                    json.dumps(report, allow_nan=False)

    def test_cosine_endpoints_and_independent_tolerance_gate(self):
        reference = {'y': np.array([1., 0.])}
        opposite = {'y': np.array([-1., 0.])}
        self.assertTrue(compare_outputs(reference, opposite, 2., 0., -1.)['passed'])
        self.assertFalse(compare_outputs(reference, opposite, 2., 0., 0.)['passed'])
        self.assertFalse(compare_outputs(reference, opposite, 0., 0., -1.)['passed'])
        self.assertTrue(compare_outputs(reference, reference, 0., 0., 1.)['passed'])

    def test_empty_or_wrong_names_have_error_and_empty_prefix(self):
        for reference, candidate in [({}, {}), ({
                'y': np.ones(1)
        }, {}), ({}, {
                'y': np.ones(1)
        }), ({
                'y': np.ones(1)
        }, {
                'z': np.ones(1)
        })]:
            with self.subTest(names=(list(reference), list(candidate))):
                with self.assertRaises(ValidationError) as caught:
                    compare_outputs(reference, candidate, 0., 0., -1.)
                report = caught.exception.report
                self.assertFalse(report['passed'])
                self.assertEqual(report['outputs'], {})
                self.assertEqual(report['error']['expected_names'], sorted(reference))
                self.assertEqual(report['error']['actual_names'], sorted(candidate))
                json.dumps(report, allow_nan=False)

    def test_shape_failure_preserves_only_completed_output_prefix(self):
        with self.assertRaises(ValidationError) as caught:
            compare_outputs({
                'first': np.ones(2),
                'bad': np.ones(2),
                'last': np.ones(1)
            }, {
                'first': np.ones(2),
                'bad': np.ones((1, 2)),
                'last': np.ones(1)
            }, 0., 0., -1.)
        report = caught.exception.report
        self.assertFalse(report['passed'])
        self.assertEqual(list(report['outputs']), ['first'])
        self.assertTrue(report['outputs']['first']['passed'])
        self.assertEqual(report['error']['output'], 'bad')
        self.assertEqual(report['error']['expected_shape'], [2])
        self.assertEqual(report['error']['actual_shape'], [1, 2])
        json.dumps(report, allow_nan=False)

    def test_nonfinite_either_side_has_nullable_json_metrics(self):
        for side in ('reference', 'candidate'):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(side=side, value=value):
                    arrays = {
                        'reference': {
                            'y': np.ones(2)
                        },
                        'candidate': {
                            'y': np.ones(2)
                        }
                    }
                    arrays[side]['y'][0] = value
                    report = compare_outputs(
                        **arrays, atol=0., rtol=0., min_cosine_similarity=-1.)
                    self.assertIs(report['passed'], False)
                    metrics = report['outputs']['y']
                    self.assertIs(metrics['finite'], False)
                    self.assertIs(metrics['passed'], False)
                    for key in ('max_abs_error', 'max_rel_error', 'cosine_similarity'):
                        self.assertIsNone(metrics[key])
                    json.dumps(report, allow_nan=False)


@unittest.skipUnless(TORCH_AVAILABLE, 'Torch absent: real CPU runner tests skipped')
class BackendExceptionDiagnosticsTest(unittest.TestCase):

    def setUp(self):
        import torch
        self.torch = torch
        self.prepared = PreparedModel(torch.nn.Identity(),
                                      {'x': torch.tensor([[1., 2.]])}, ('y', ))

    def backend(self, error=None):

        class CPUBackend:

            def infer(inner, inputs, synchronize=False):
                if not synchronize:
                    raise AssertionError('expected synchronized validation')
                if error is not None:
                    raise error
                return {'y': inputs['x'].clone()}

        return CPUBackend()

    def test_real_cpu_identity_and_negative_control(self):
        self.assertTrue(validate_backend(self.prepared, self.backend(), {})['passed'])
        self.prepared.reference_runner = lambda inputs: inputs['x'] + 1
        report = validate_backend(self.prepared, self.backend(),
                                  {'fail_on_error': False})
        self.assertFalse(report['passed'])
        with self.assertRaises(ValidationError) as caught:
            validate_backend(self.prepared, self.backend(), {})
        self.assertEqual(caught.exception.report, report)

    def test_single_runner_errors_keep_identity_type_and_report(self):
        for side in ('reference', 'candidate'):
            for strict in (False, True):
                with self.subTest(side=side, strict=strict):
                    error = LookupError('injected ' + side + ' failure')

                    def fail(inputs):
                        raise error

                    self.prepared.reference_runner = (
                        fail if side == 'reference' else None)
                    backend = self.backend(error if side == 'candidate' else None)
                    with self.assertRaises(LookupError) as caught:
                        validate_backend(self.prepared, backend,
                                         {'fail_on_error': strict})
                    self.assertIs(caught.exception, error)
                    self.assertFalse(error.report['passed'])
                    self.assertEqual(error.report['error'], {
                        'type': 'LookupError',
                        'message': str(error)
                    })
                    json.dumps(error.report, allow_nan=False)

    def test_manifest_parse_and_schema_errors_have_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            for text, error_type in [('{', json.JSONDecodeError),
                                     ('{"samples": []}', ValidationError)]:
                for strict in (False, True):
                    with self.subTest(text=text, strict=strict):
                        path.write_text(text, encoding='utf-8')
                        with self.assertRaises(error_type) as caught:
                            validate_backend(self.prepared, self.backend(), {
                                'samples_manifest': str(path),
                                'fail_on_error': strict
                            })
                        self.assertIs(type(caught.exception), error_type)
                        report = caught.exception.report
                        self.assertFalse(report['passed'])
                        self.assertEqual(report['error']['type'], error_type.__name__)
                        json.dumps(report, allow_nan=False)


class PipelineReportControlTest(unittest.TestCase):
    """Real transform orchestration; injected non-inference stage boundaries."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        config_path = self.root / 'deploy.py'
        config_path.write_text(
            'deployment = ' + repr(
                dict(
                    name='cpu-report-control',
                    work_dir=str(self.root / 'work'),
                    pipeline=[
                        dict(type='PrepareModel'),
                        dict(type='ExportONNX'),
                        dict(type='OptimizeONNX'),
                        dict(type='LowerPrecision'),
                        dict(type='ConvertBackend'),
                        dict(type='ValidateBackend'),
                        dict(type='PackageBundle'),
                    ],
                    source=dict(type='python', device='cpu', factory='unused:factory'),
                    inputs=[dict(name='x', dtype='float32', shape=(1, 2))],
                    outputs=[dict(name='y', dtype='float32', shape=(1, 2))],
                    backend_config=dict(type='tensorrt', precision='fp32'),
                    optimization=dict(simplify=False),
                    validation=dict(fail_on_error=False),
                )),
            encoding='utf-8')
        self.pipeline = CompilePipeline(load_config(config_path))
        self.work = self.pipeline.config.work_dir
        self.verdict = True
        self.failure = None
        self.running = []
        self.prepared = SimpleNamespace(metadata={})
        stack = ExitStack()
        self.addCleanup(stack.close)

        def build_processor(*args):
            # Observe the on-disk state before any run stage completes.
            self.running.append(self.read_report('result'))
            if self.failure == 'prepare':
                raise LookupError('injected prepare failure')
            return SimpleNamespace(
                prepare_model=lambda: self.prepared, runtime_metadata=metadata)

        def metadata():
            if self.failure == 'metadata':
                raise LookupError('injected metadata failure')
            return {}

        def package(*args, **kwargs):
            if self.failure == 'package':
                raise LookupError('injected package failure')
            return self.root / 'control-only-bundle'

        manager = SimpleNamespace(
            to_backend=lambda *args: SimpleNamespace(files=[], metadata={}),
            build_wrapper=lambda *args, **kwargs: object())
        replacements = dict(
            build_task_processor=build_processor,
            export_onnx=lambda *args: self.root / 'control-only.onnx',
            optimize_onnx=lambda path, *args: path,
            lower_onnx_precision=lambda path, *args: path,
            get_backend_manager=lambda name: manager,
            validate_backend=lambda *args: numeric_report(self.verdict),
            create_bundle=package)
        for name, replacement in replacements.items():
            stack.enter_context(
                patch('fluxhermes.transforms.stages.' + name, replacement))

    def read_report(self, name):
        return json.loads((self.work / (name + '.json')).read_text(encoding='utf-8'))

    def assert_running(self, report):
        self.assertEqual(report['status'], 'running')
        self.assertFalse(report['workflow_completed'])
        self.assertIsNone(report['numerical_passed'])
        UUID(report['run_id'])

    def assert_failure(self, verdict):
        result = self.read_report('result')
        self.assertEqual(result, self.read_report('failure'))
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['workflow_completed'])
        self.assertIs(result['numerical_passed'], verdict)
        self.assertEqual(result['error']['type'], 'LookupError')
        self.assertEqual(result['run_id'], self.running[-1]['run_id'])
        json.dumps(result, allow_nan=False)
        return result

    def test_success_then_failure_resets_records_validation_and_current_result(self):
        self.pipeline.execute()
        success = self.read_report('result')
        self.assertEqual(success['status'], 'completed')
        self.assertTrue(success['workflow_completed'])
        self.assertTrue(success['numerical_passed'])
        self.assertEqual(success['run_id'], self.running[-1]['run_id'])
        self.failure = 'prepare'
        with self.assertRaisesRegex(LookupError, 'prepare'):
            self.pipeline.execute()
        failed = self.assert_failure(None)
        self.assertIsNone(failed['validation'])
        self.assertNotEqual(success['run_id'], failed['run_id'])
        self.assertEqual([s['name'] for s in failed['stages']], ['prepare_model'])
        for report in self.running:
            self.assert_running(report)

    def test_failure_then_success_retains_distinguishable_historical_failure(self):
        self.failure = 'prepare'
        with self.assertRaises(LookupError):
            self.pipeline.execute()
        failure = self.assert_failure(None)
        self.failure = None
        self.pipeline.execute()
        result = self.read_report('result')
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['workflow_completed'])
        self.assertNotEqual(result['run_id'], failure['run_id'])
        self.assertEqual(self.read_report('failure'), failure)
        self.assertTrue(
            all(stage['status'] == 'completed' for stage in result['pipeline']))
        self.assertEqual(
            len({s['name']
                 for s in result['pipeline']}), len(result['pipeline']))
        for report in self.running:
            self.assert_running(report)

    def test_attached_validation_error_report_survives_pipeline_failure(self):
        report = numeric_report(False)
        error = ValidationError('injected strict numerical failure')
        error.report = report
        with patch('fluxhermes.transforms.stages.validate_backend', side_effect=error):
            with self.assertRaises(ValidationError) as caught:
                self.pipeline.execute()
        self.assertIs(caught.exception, error)
        result = self.read_report('result')
        self.assertEqual(result, self.read_report('failure'))
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['workflow_completed'])
        self.assertFalse(result['numerical_passed'])
        self.assertEqual(result['validation'], report)
        self.assertEqual(result['run_id'], self.running[-1]['run_id'])
        self.assertEqual(result['stages'][-1]['name'], 'validate')
        self.assertEqual(result['stages'][-1]['status'], 'failed')

    def test_downstream_failure_retains_real_numeric_verdict(self):
        for verdict in (True, False):
            for location in ('metadata', 'package', 'after_hook'):
                with self.subTest(verdict=verdict, location=location):
                    self.verdict, self.failure = verdict, location

                    def after_hook(**kwargs):
                        # The numerical report must already be saved before this hook.
                        self.assertEqual(self.pipeline.context['validation'], {
                            **numeric_report(verdict), 'enabled': True
                        })
                        if self.failure == 'after_hook':
                            raise LookupError('injected after hook failure')

                    if location == 'after_hook':
                        self.pipeline.pipeline_manager.add_hook('validate', after_hook)
                    with self.assertRaises(LookupError):
                        self.pipeline.execute()
                    report = self.assert_failure(verdict)
                    self.assertEqual(report['validation']['outputs'],
                                     numeric_report(verdict)['outputs'])
                    if location == 'after_hook':
                        self.assertEqual(report['stages'][-1]['name'], 'validate')
                        self.assertEqual(report['stages'][-1]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
