"""main@3db1262 hook ABI, exercised with real Torch -> ONNX -> ORT CPU."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_transforms import CPU_EXPORT_AVAILABLE, CPUManager, CPUWrapper

from fluxhermes.adapters.base import PreparedModel
from fluxhermes.backend.base import BackendBuildResult
from fluxhermes.bundle import verify_bundle
from fluxhermes.codebase.base import BaseTaskProcessor
from fluxhermes.config import load_config
from fluxhermes.contracts import BundleManifest, DeploymentConfig
from fluxhermes.pipeline import CompilePipeline


def enrich_metadata(*, stage, result, record):
    # The public hook used by the independent PR8 reproducer.
    result.metadata['hook_observed'] = 'yes'


def legacy_values(root, **overrides):
    values = dict(
        name='legacy-hooks',
        work_dir=str(root / 'work'),
        source=dict(
            type='python',
            device='cpu',
            factory='test_transforms:tiny_factory',
            kwargs=dict(checkpoint=str(root / 'weights.pt'))),
        backend_config=dict(type='pipeline_test_cpu', precision='fp32'),
        inputs=[dict(name='x', dtype='float32', shape=(1, 2))],
        outputs=[dict(name='y', dtype='float32', shape=(1, 2))],
        export=dict(opset=18, dynamo=False, external_data=False),
        validation=dict(enabled=True, atol=.05, rtol=.05),
        package=dict(include_onnx=True))
    values.update(overrides)
    return values


class LegacyHookConfigTest(unittest.TestCase):

    def test_save_load_preserves_implicit_and_explicit_hook_abi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for explicit in (False, True):
                with self.subTest(explicit=explicit):
                    values = legacy_values(root)
                    if explicit:
                        values['pipeline'] = [dict(type='PrepareModel')]
                    config = DeploymentConfig.from_dict(values)
                    serialized = config.to_dict()
                    self.assertEqual('pipeline' in serialized, explicit)
                    path = root / 'deploy.py'
                    path.write_text(
                        'deployment = ' + repr(serialized), encoding='utf-8')
                    reloaded = load_config(path)
                    self.assertEqual(reloaded.pipeline, config.pipeline)
                    self.assertEqual('pipeline' in reloaded.to_dict(), explicit)


@unittest.skipUnless(CPU_EXPORT_AVAILABLE, 'Optional Torch/ONNX/ORT CPU dependencies')
class LegacyHookCPUTest(unittest.TestCase):

    def setUp(self):
        import torch

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        torch.save(
            dict(
                weight=torch.tensor([[2., .5], [-.25, 1.]]),
                bias=torch.tensor([1., -.5])), self.root / 'weights.pt')

    def pipeline(self, **overrides):
        return CompilePipeline(
            DeploymentConfig.from_dict(legacy_values(self.root, **overrides)))

    def test_configured_metadata_hook_and_processor_stage_survive_json_roundtrip(self):
        pipeline = self.pipeline(pipeline_hooks=[
            dict(stage='prepare_model', hook=f'{__name__}:enrich_metadata')
        ])
        config = DeploymentConfig.from_dict(
            json.loads(json.dumps(pipeline.config.to_dict())))
        pipeline = CompilePipeline(config)
        processors = []
        pipeline.pipeline_manager.add_hook('build_task_processor',
                                           lambda **kw: processors.append(kw['result']))
        manifest = verify_bundle(pipeline.execute())
        self.assertTrue(manifest.validation['passed'])
        self.assertEqual(manifest.source['hook_observed'], 'yes')
        self.assertEqual(len(processors), 1)
        self.assertIsInstance(processors[0], BaseTaskProcessor)

    def check_all_stages(self, precision='fp32', validation=True, relative_work=False):
        import onnx

        work_dir = self.root / 'work'
        if relative_work:
            work_dir = Path(os.path.relpath(work_dir))
        pipeline = self.pipeline(
            work_dir=str(work_dir),
            backend_config=dict(type='pipeline_test_cpu', precision=precision),
            quantization=dict(mode='int8' if precision == 'int8' else 'none'),
            validation=dict(enabled=validation, atol=.05, rtol=.05))
        config, work = pipeline.config, pipeline.config.work_dir
        expected = [
            'build_task_processor', 'prepare_model', 'export_ir', 'optimize_ir',
            'lower_precision'
        ]
        if precision == 'int8':
            expected.append('quantize')
        expected.append('build_backend')
        if validation:
            expected.append('validate')
        expected.append('package')
        events, results, records = [], {}, {}
        asset = self.root / 'hook-asset.txt'
        asset.write_text('asset added by before hook', encoding='utf-8')

        def before(*, stage, args, kwargs, record):
            events.append((stage, 'before'))
            self.assertEqual(record.name, stage)
            self.assertEqual(record.status, 'running')
            records[stage] = record
            self.assertIsInstance(args, tuple)
            self.assertEqual(set(kwargs), {'assets'} if stage == 'package' else set())
            if stage == 'build_task_processor':
                self.assertEqual(args, (config, 'cpu'))
                self.assertIs(args[0], config)
            elif stage == 'prepare_model':
                self.assertEqual(args, ())
                self.assertTrue(results['build_task_processor'].hook_marker)
            elif stage == 'export_ir':
                self.assertEqual(len(args), 3)
                self.assertIs(args[0], results['prepare_model'])
                self.assertEqual(args[0].metadata['prepared_hook'], 'yes')
                self.assertEqual(args[1], work / 'model.onnx')
                self.assertIs(args[2], config.ir)
                args[2]['opset'] = 17
            elif stage in ('optimize_ir', 'lower_precision', 'quantize'):
                previous = dict(
                    optimize_ir='export_ir',
                    lower_precision='optimize_ir',
                    quantize='lower_precision')[stage]
                self.assertIs(args[0], results[previous])
                filename = {
                    'optimize_ir': 'model.optimized.onnx',
                    'lower_precision': f'model.{precision}.onnx',
                    'quantize': 'model.int8.onnx'
                }[stage]
                self.assertEqual(args[1], work / filename)
                self.assertEqual(len(args), 3 if stage == 'optimize_ir' else 4)
                options = (
                    config.quantization if stage == 'quantize' else config.optimization)
                self.assertIs(args[-1], options)
                if stage == 'lower_precision':
                    self.assertEqual(args[2], precision)
                    self.assertEqual(options['hook_marker'], 'optimize')
                elif stage == 'optimize_ir':
                    # A mutation must be visible to the next historical stage.
                    options['hook_marker'] = 'optimize'
                    self.assertNotIn('simplify', options)
                else:
                    self.assertIs(args[2], results['prepare_model'].calibration_source)
            elif stage == 'build_backend':
                self.assertEqual(len(args), 3)
                self.assertIsInstance(args[0], list)
                self.assertEqual(args[0],
                                 [results.get('quantize', results['lower_precision'])])
                self.assertEqual(args[1], work)
                self.assertIs(args[2], config)
            elif stage == 'validate':
                self.assertEqual(len(args), 3)
                self.assertIs(args[0], results['prepare_model'])
                self.assertIsInstance(args[1], CPUWrapper)
                self.assertIs(args[2], config.validation)
            elif stage == 'package':
                self.assertEqual(len(args), 3)
                self.assertEqual(args[0], (work / 'bundle').resolve())
                self.assertIsInstance(args[1], BundleManifest)
                self.assertIsInstance(args[2], dict)
                self.assertEqual(args[1].source['prepared_hook'], 'yes')
                self.assertEqual(args[1].source['processor_hook'], 'yes')
                self.assertEqual(args[1].backend_metadata['backend_hook'], 'yes')
                self.assertEqual(args[1].validation['enabled'], validation)
                if validation:
                    self.assertEqual(args[1].validation['validation_hook'], 'yes')
                else:
                    self.assertEqual(args[1].validation['reason'],
                                     'validation was disabled')
                self.assertEqual(
                    set(args[1].metadata), {
                        'codebase_config', 'backend_config', 'ir_config',
                        'quantization', 'optimization', 'pipeline'
                    })
                self.assertEqual([r['name'] for r in args[1].metadata['pipeline']],
                                 expected[:-1])
                self.assertTrue(
                    all(r['status'] == 'completed'
                        for r in args[1].metadata['pipeline']))
                args[1].source['package_hook'] = 'yes'
                args[2]['hook/extra.txt'] = asset
                # Exercise kwargs replacement, not only mutation of an existing list.
                kwargs['assets'] = [asset]
            return 'ignored before return'

        def after(*, stage, result, record):
            events.append((stage, 'after'))
            self.assertIs(record, records[stage])
            self.assertEqual(record.status, 'completed')
            results[stage] = result
            if stage == 'build_task_processor':
                self.assertIsInstance(result, BaseTaskProcessor)
                result.hook_marker = True
                result.runtime_metadata = lambda: dict(processor_hook='yes')
            elif stage == 'prepare_model':
                self.assertIsInstance(result, PreparedModel)
                result.metadata['prepared_hook'] = 'yes'
            elif stage in ('export_ir', 'optimize_ir', 'lower_precision', 'quantize',
                           'package'):
                self.assertIsInstance(result, Path)
                self.assertTrue(result.exists())
            elif stage == 'build_backend':
                self.assertIsInstance(result, BackendBuildResult)
                result.metadata['backend_hook'] = 'yes'
            elif stage == 'validate':
                self.assertIsInstance(result, dict)
                self.assertTrue(result['passed'])
                self.assertNotIn('enabled', result)
                result['validation_hook'] = 'yes'
            return 'ignored after return'

        for stage in expected:
            pipeline.pipeline_manager.add_hook(stage, before, when='before')
            pipeline.pipeline_manager.add_hook(stage, after)
        # No extra context-only rewrite stage existed in the historical API.
        pipeline.pipeline_manager.add_hook(
            'rewrite', lambda **kw: self.fail('legacy rewrite hook'))
        manifest = verify_bundle(pipeline.execute())
        self.assertEqual(events, [(stage, when) for stage in expected
                                  for when in ('before', 'after')])
        self.assertEqual([record.name for record in pipeline.pipeline_manager.records],
                         expected)
        self.assertEqual(manifest.source['package_hook'], 'yes')
        self.assertEqual(manifest.validation['passed'], validation)
        self.assertTrue((results['package'] / 'hook/extra.txt').is_file())
        self.assertTrue((results['package'] / 'assets/hook-asset.txt').is_file())
        self.assertEqual(
            onnx.load(str(work / 'model.onnx')).opset_import[0].version, 17)
        for record in records.values():
            self.assertEqual(record.status, 'completed')
            self.assertGreater(record.duration_s, 0)

    def test_all_fp32_before_after_contracts_and_mutations(self):
        self.check_all_stages()

    def test_all_int8_before_after_contracts_with_real_quantization(self):
        self.check_all_stages(precision='int8')

    @unittest.skipUnless(
        importlib.util.find_spec('onnxconverter_common') is not None,
        'Optional FP16 converter')
    def test_all_fp16_before_after_contracts_with_real_lowering(self):
        self.check_all_stages(precision='fp16')

    def test_validation_disabled_skips_hooks_but_keeps_other_legacy_stages(self):
        self.check_all_stages(validation=False)

    def check_validation_toggle(self, initial, target):
        pipeline = self.pipeline(validation=dict(enabled=initial, atol=.05, rtol=.05))
        self.assertEqual('validate' in [step['name'] for step in pipeline.plan()],
                         initial)
        events = []

        def select_validation(*, stage, args, kwargs, record):
            self.assertIs(args[2], pipeline.config)
            args[2].validation['enabled'] = target

        def before(*, stage, args, kwargs, record):
            self.assertEqual(len(args), 3)
            self.assertIs(args[2], pipeline.config.validation)
            events.append('before')

        def after(*, stage, result, record):
            self.assertTrue(result['passed'])
            self.assertNotIn('enabled', result)
            events.append('after')

        pipeline.pipeline_manager.add_hook(
            'build_backend', select_validation, when='before')
        pipeline.pipeline_manager.add_hook('validate', before, when='before')
        pipeline.pipeline_manager.add_hook('validate', after)
        # Instrument the real wrapper constructor; inference is not replaced.
        with patch.object(
                CPUManager, 'build_wrapper', wraps=CPUManager.build_wrapper) as wrapper:
            manifest = verify_bundle(pipeline.execute())
        self.assertEqual(wrapper.call_count, int(target))
        self.assertEqual(events, ['before', 'after'] if target else [])
        self.assertEqual(manifest.validation['enabled'], target)
        self.assertEqual(manifest.validation['passed'], target)
        self.assertEqual(
            'validate' in [r.name for r in pipeline.pipeline_manager.records], target)
        self.assertEqual('validate' in [step['name'] for step in pipeline.plan()],
                         target)
        if not target:
            self.assertEqual(
                manifest.validation,
                dict(enabled=False, passed=False, reason='validation was disabled'))

    def test_legacy_before_hook_can_enable_initially_disabled_validation(self):
        self.check_validation_toggle(False, True)

    def test_legacy_before_hook_can_disable_initially_enabled_validation(self):
        self.check_validation_toggle(True, False)

    def test_relative_work_dir_preserves_old_stage_path_arguments(self):
        self.check_all_stages(relative_work=True)

    def test_relative_package_paths_preserve_old_cwd_and_config_resolution(self):
        for from_file in (False, True):
            with self.subTest(from_file=from_file):
                asset = self.root / 'asset.txt'
                asset.write_text('configured legacy asset', encoding='utf-8')
                values = legacy_values(
                    self.root,
                    package=dict(
                        output_dir='custom-bundle'
                        if from_file else os.path.relpath(self.root / 'custom-bundle'),
                        assets=['asset.txt' if from_file else os.path.relpath(asset)]))
                if from_file:
                    path = self.root / 'deploy.py'
                    path.write_text('deployment = ' + repr(values), encoding='utf-8')
                    config = load_config(path)
                else:
                    config = DeploymentConfig.from_dict(values)
                pipeline = CompilePipeline(config)
                observed = []

                def before(*, stage, args, kwargs, record):
                    self.assertEqual(args[0], self.root / 'custom-bundle')
                    self.assertEqual(kwargs['assets'], [asset])
                    observed.append(stage)

                pipeline.pipeline_manager.add_hook('package', before, when='before')
                bundle = pipeline.execute()
                self.assertTrue(verify_bundle(bundle).validation['passed'])
                self.assertEqual(observed, ['package'])
                self.assertTrue((bundle / 'assets/asset.txt').is_file())

    def test_explicit_pipeline_keeps_context_contract_after_roundtrip(self):
        implicit = self.pipeline().config
        values = implicit.to_dict()
        values['pipeline'] = implicit.pipeline
        # Explicit stages ignore the old validation.enabled switch.
        values['validation']['enabled'] = False
        pipeline = CompilePipeline(DeploymentConfig.from_dict(values))
        pipeline = CompilePipeline(
            DeploymentConfig.from_dict(pipeline.config.to_dict()))
        events = []

        def before(*, stage, args, kwargs, record):
            self.assertEqual(len(args), 1)
            self.assertIs(args[0], pipeline.context)
            self.assertEqual(kwargs, {})
            self.assertEqual(record.status, 'running')
            events.append((stage, 'before'))

        def after(*, stage, result, record):
            self.assertIs(result, pipeline.context)
            self.assertEqual(record.status, 'completed')
            events.append((stage, 'after'))

        stages = [
            'prepare_model', 'rewrite', 'export_ir', 'optimize_ir', 'lower_precision',
            'build_backend', 'validate', 'package'
        ]
        for stage in stages:
            pipeline.pipeline_manager.add_hook(stage, before, when='before')
            pipeline.pipeline_manager.add_hook(stage, after)
        pipeline.pipeline_manager.add_hook(
            'build_task_processor', lambda **kw: self.fail('explicit processor hook'))
        self.assertTrue(verify_bundle(pipeline.execute()).validation['passed'])
        self.assertEqual([record.name for record in pipeline.pipeline_manager.records],
                         stages)
        self.assertEqual(len(events), len(stages) * 2)

    def test_legacy_hook_failure_is_attributed_to_its_stage(self):
        for stage, when in (('build_task_processor', 'after'), ('export_ir', 'before')):
            with self.subTest(stage=stage, when=when):
                pipeline = self.pipeline()

                def fail(**kwargs):
                    raise LookupError('deliberate legacy hook failure')

                pipeline.pipeline_manager.add_hook(stage, fail, when=when)
                with self.assertRaisesRegex(LookupError, 'deliberate legacy hook'):
                    pipeline.execute()
                report = json.loads(
                    (pipeline.config.work_dir / 'failure.json').read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['stages'][-1]['name'], stage)
                self.assertEqual(report['stages'][-1]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
