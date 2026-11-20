"""Config composition tests, with real Torch/ONNX/ORT CPU integration."""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fluxhermes.adapters.base import PreparedModel
from fluxhermes.backend.base import (BACKENDS, BackendBuildResult, BaseBackendManager,
                                     BaseBackendWrapper)
from fluxhermes.bundle import verify_bundle
from fluxhermes.config import load_config
from fluxhermes.contracts import DeploymentConfig
from fluxhermes.core import FunctionRewriter
from fluxhermes.errors import ConfigurationError, ValidationError
from fluxhermes.pipeline import CompilePipeline
from fluxhermes.transforms import TRANSFORMS, BaseTransform
from fluxhermes.utils import sha256_file

CPU_EXPORT_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ('torch', 'onnx', 'onnxruntime'))
_PREPARES = 0


def tiny_factory(deployment, device, checkpoint, random_inputs=False):
    import torch

    global _PREPARES
    _PREPARES += 1
    module = torch.nn.Linear(2, 2).to(device).eval()
    module.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True))
    inputs = {
        'x': (torch.rand(1, 2, device=device)
              if random_inputs else torch.tensor([[.25, -.75]], device=device))
    }
    return PreparedModel(
        module,
        inputs, ('y', ),
        calibration_source=lambda: iter([inputs]),
        metadata=dict(checkpoint_sha256=sha256_file(checkpoint)))


class CPUWrapper(BaseBackendWrapper):

    def __init__(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=['CPUExecutionProvider'])

    def infer(self, inputs, synchronize=False):
        arrays = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in inputs.items()
        }
        names = [output.name for output in self.session.get_outputs()]
        return dict(zip(names, self.session.run(names, arrays)))


@BACKENDS.register_module('pipeline_test_cpu')
class CPUManager(BaseBackendManager):
    """Test backend executing actual ONNX graphs with ONNX Runtime CPU."""
    backend_name = 'pipeline_test_cpu'

    @classmethod
    def is_available(cls):
        return CPU_EXPORT_AVAILABLE

    @classmethod
    def get_version(cls):
        import onnxruntime
        return onnxruntime.__version__

    @classmethod
    def to_backend(cls, ir_files, work_dir, config):
        import onnx

        output = work_dir / 'cpu-backend.onnx'
        onnx.save_model(onnx.load(str(ir_files[0])), str(output))
        wrapper = CPUWrapper(output)
        return BackendBuildResult((output, ),
                                  dict(providers=wrapper.session.get_providers()))

    @classmethod
    def build_wrapper(cls, backend_files, device, **kwargs):
        return CPUWrapper(backend_files[0])


@TRANSFORMS.register_module()
class ReferenceDrift(BaseTransform):
    """Inject a real numerical negative control after model preparation."""
    name = 'reference_drift'
    requires = ('prepared', )
    provides = ('drift', )
    invalidates = ('validation', 'bundle')

    def __init__(self, offset):
        self.offset = offset

    def __call__(self, data):
        module = data['prepared'].module
        data['prepared'].reference_runner = lambda inputs: module(inputs['x']
                                                                  ) + self.offset
        return {'drift': self.offset}


def rewrite_target(value):
    return value + 1


@TRANSFORMS.register_module()
class ObserveRewrite(BaseTransform):
    name = 'observe_rewrite'
    provides = ('observed', )

    def __call__(self, data):
        return {**data, 'observed': rewrite_target(1)}


@TRANSFORMS.register_module()
class FailTransform(BaseTransform):
    name = 'fail_transform'

    def __call__(self, data):
        raise LookupError('deliberate transform failure')


class CompositionFixture(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def config(self, pipeline=None, **overrides):
        values = dict(
            name='composition',
            work_dir=str(self.root / 'work'),
            inputs=[dict(name='x', dtype='float32', shape=(1, 2))],
            outputs=[dict(name='y', dtype='float32', shape=(1, 2))],
            backend_config=dict(type='pipeline_test_cpu', precision='fp32'))
        if pipeline is not None:
            values['pipeline'] = pipeline
        values.update(overrides)
        return DeploymentConfig.from_dict(values)

    def source(self):
        return dict(
            type='python',
            task='custom',
            device='cpu',
            source=dict(type='python', factory='unused:factory'))


class CompositionTest(CompositionFixture):

    def test_bad_compositions_fail_before_loading_or_creating_artifacts(self):
        chains = [
            [],
            'PrepareModel',
            [{}],
            [dict(type='UnknownTransform')],
            [dict(type='ExportONNX', path='wrong-option.onnx')],
            [dict(type='ConvertBackend')],
            [dict(type='ValidateBackend')],
            [dict(type='PrepareReference')],
            [dict(type='Rewrite', transforms=[dict(type='ExportONNX')])],
            [dict(type='IsolatedExport', transforms=[dict(type='LoadONNX', path='x')])],
        ]
        for chain in chains:
            with self.subTest(chain=chain), patch(
                    'fluxhermes.transforms.stages.build_task_processor') as prepare:
                with self.assertRaises(ConfigurationError):
                    CompilePipeline(self.config(chain, codebase_config=self.source()))
                prepare.assert_not_called()
                self.assertFalse((self.root / 'work').exists())
        with self.assertRaisesRegex(ConfigurationError, 'source'):
            CompilePipeline(self.config([dict(type='PrepareModel')]))

    def test_replacing_graph_discards_precision_and_backend_dependencies(self):
        for precision in ('fp16', 'int8'):
            for replacement in (dict(type='LoadONNX',
                                     path='new.onnx'), dict(type='ExportONNX')):
                chain = [
                    dict(type='PrepareModel'),
                    dict(type='LoadONNX', path='old.onnx', precision=precision),
                    replacement,
                    dict(type='ConvertBackend')
                ]
                with self.subTest(precision=precision, replacement=replacement):
                    with self.assertRaisesRegex(ConfigurationError, 'requires'):
                        CompilePipeline(
                            self.config(
                                chain,
                                codebase_config=self.source(),
                                backend_config=dict(
                                    type='pipeline_test_cpu', precision=precision)))
        with self.assertRaisesRegex(ConfigurationError, 'backend_result'):
            CompilePipeline(
                self.config([
                    dict(type='LoadONNX', path='old.onnx'),
                    dict(type='ConvertBackend'),
                    dict(type='LoadONNX', path='new.onnx'),
                    dict(type='PackageBundle')
                ]))

    def test_precision_declaration_must_match_backend(self):
        with self.assertRaisesRegex(ConfigurationError, 'must match'):
            CompilePipeline(
                self.config([dict(type='LoadONNX', path='x', precision='fp16')]))

    def test_backend_only_config_roundtrip(self):
        config = self.config([dict(type='LoadONNX', path='existing.onnx')])
        reloaded = DeploymentConfig.from_dict(config.to_dict())
        self.assertEqual(reloaded.source, {})
        self.assertEqual(reloaded.pipeline, config.pipeline)
        self.assertEqual(CompilePipeline(reloaded).plan()[0]['type'], 'LoadONNX')

    def test_plan_and_execution_follow_nested_configuration(self):
        chain = [
            dict(type='Rewrite', transforms=[dict(type='ObserveRewrite')]),
            dict(type='ObserveRewrite')
        ]
        pipeline = CompilePipeline(self.config(chain))
        plan = pipeline.plan_dict()
        self.assertEqual(plan['stages'][0]['type'], 'Rewrite')
        self.assertEqual(plan['stages'][0]['transforms'][0]['type'], 'ObserveRewrite')
        self.assertEqual(plan['pipeline'], tuple(chain))
        original = rewrite_target
        rewriter = FunctionRewriter()
        rewriter.register_rewriter(f'{__name__}.rewrite_target')(
            lambda ctx, value: ctx.origin_func(value) * 10)
        observed = []
        pipeline.pipeline_manager.add_hook(
            'observe_rewrite',
            lambda **kwargs: observed.append(kwargs['result']['observed']))
        with patch('fluxhermes.transforms.stages.FUNCTION_REWRITER', rewriter):
            pipeline.execute()
        self.assertEqual(observed, [20, 2])
        self.assertIs(rewrite_target, original)
        self.assertEqual(
            [stage['name'] for stage in pipeline.pipeline_manager.report()],
            ['rewrite', 'observe_rewrite', 'observe_rewrite'])

    def test_transform_failure_restores_rewrites_and_writes_records(self):
        rewriter = FunctionRewriter()
        rewriter.register_rewriter(f'{__name__}.rewrite_target')(
            lambda ctx, value: ctx.origin_func(value) * 10)
        original = rewrite_target
        pipeline = CompilePipeline(
            self.config([
                dict(
                    type='Rewrite',
                    transforms=[
                        dict(type='ObserveRewrite'),
                        dict(type='FailTransform')
                    ])
            ]))
        with patch('fluxhermes.transforms.stages.FUNCTION_REWRITER', rewriter):
            with self.assertRaisesRegex(LookupError, 'deliberate'):
                pipeline.execute()
        self.assertIs(rewrite_target, original)
        report = json.loads((self.root / 'work/result.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual([stage['status'] for stage in report['stages']],
                         ['failed', 'completed', 'failed'])

    def test_missing_later_rewrite_restores_already_applied_patches(self):
        rewriter = FunctionRewriter()
        for target in ('rewrite_target', 'missing_rewrite_target'):
            rewriter.register_rewriter(f'{__name__}.{target}')(
                lambda ctx, value: ctx.origin_func(value) * 10)
        original = rewrite_target
        with self.assertRaises(AttributeError):
            with rewriter.context('python', 'pipeline_test_cpu', 'onnx'):
                self.fail('Invalid rewrite setup must fail before entering its scope')
        self.assertIs(rewrite_target, original)


@unittest.skipUnless(CPU_EXPORT_AVAILABLE, 'Optional Torch/ONNX/ORT CPU dependencies')
class CPUCompositionTest(CompositionFixture):

    def setUp(self):
        super().setUp()
        import torch

        global _PREPARES
        _PREPARES = 0
        self.checkpoint = self.root / 'weights.pt'
        torch.save(
            dict(
                weight=torch.tensor([[2., .5], [-.25, 1.]]),
                bias=torch.tensor([1., -.5])), self.checkpoint)

    def write_config(self, chain, **overrides):
        values = dict(
            name='cpu-composition',
            work_dir=str(self.root / 'work'),
            pipeline=chain,
            codebase_config=dict(
                type='python',
                task='custom',
                device='cpu',
                python_paths=[str(Path(__file__).parent)],
                source=dict(
                    type='python',
                    factory=f'{__name__}:tiny_factory',
                    kwargs=dict(checkpoint=str(self.checkpoint)))),
            inputs=[dict(name='x', dtype='float32', shape=(1, 2))],
            outputs=[dict(name='y', dtype='float32', shape=(1, 2))],
            backend_config=dict(
                type='pipeline_test_cpu',
                precision='fp32',
                python_paths=[str(Path(__file__).parent)],
                custom_modules=[__name__]),
            ir_config=dict(type='onnx', opset=18, dynamo=False, external_data=False),
            validation=dict(atol=1e-6, rtol=1e-6, min_cosine_similarity=.9999),
            package=dict(include_onnx=True))
        values.update(overrides)
        path = self.root / 'deploy.py'
        path.write_text('deployment = ' + repr(values), encoding='utf-8')
        return CompilePipeline(load_config(path))

    def export_chain(self):
        return [
            dict(type='PrepareModel'),
            dict(
                type='Rewrite',
                transforms=[
                    dict(
                        type='ExportONNX',
                        output='graphs/export.onnx',
                        options=dict(opset=17))
                ])
        ]

    def test_export_only_loads_real_weights_and_honors_transform_options(self):
        import numpy as np
        import onnx

        pipeline = self.write_config(self.export_chain())
        with patch.object(
                CPUManager, 'to_backend', side_effect=AssertionError('unused backend')):
            graph = pipeline.execute()
        self.assertEqual(graph, self.root / 'work/graphs/export.onnx')
        model = onnx.load(str(graph))
        onnx.checker.check_model(model)
        self.assertEqual(model.opset_import[0].version, 17)
        np.testing.assert_allclose(
            CPUWrapper(graph).infer(pipeline.context['prepared'].inputs)['y'],
            [[1.125, -1.3125]])
        self.assertEqual(_PREPARES, 1)
        self.assertEqual(pipeline.context['prepared'].metadata['checkpoint_sha256'],
                         sha256_file(self.checkpoint))
        report = json.loads((self.root / 'work/result.json').read_text())
        self.assertTrue(report['workflow_completed'])
        self.assertIsNone(report['numerical_passed'])
        self.assertNotIn('build_backend',
                         [stage['name'] for stage in report['pipeline']])

    def test_existing_onnx_to_backend_without_source_model(self):
        graph = self.write_config(self.export_chain()).execute()
        pipeline = self.write_config([
            dict(type='LoadONNX', path='work/graphs/export.onnx'),
            dict(type='ConvertBackend'),
            dict(type='PackageBundle')
        ],
                                     codebase_config={},
                                     work_dir=str(self.root / 'conversion'))
        bundle = pipeline.execute()
        manifest = verify_bundle(bundle)
        self.assertEqual(_PREPARES, 1)
        self.assertNotIn('prepared', pipeline.context)
        self.assertFalse(manifest.validation['enabled'])
        self.assertFalse(manifest.validation['passed'])
        self.assertEqual(manifest.source['onnx_sha256'], sha256_file(graph))
        self.assertTrue((bundle / 'backend/cpu-backend.onnx').is_file())
        self.assertTrue((bundle / 'model/model.onnx').is_file())

    def test_explicit_fp16_lowering_and_reexport_clears_precision_marker(self):
        import onnx

        pipeline = self.write_config(
            self.export_chain() + [dict(type='LowerPrecision')],
            backend_config=dict(type='pipeline_test_cpu', precision='fp16'))
        graph = pipeline.execute()
        model = onnx.load(str(graph))
        onnx.checker.check_model(model)
        self.assertIn(onnx.TensorProto.FLOAT16,
                      [value.data_type for value in model.graph.initializer])
        self.assertTrue(pipeline.context['precision_ready'])
        pipeline = self.write_config(
            self.export_chain() +
            [dict(type='LowerPrecision'),
             dict(type='ExportONNX', output='fresh.onnx')],
            backend_config=dict(type='pipeline_test_cpu', precision='fp16'))
        graph = pipeline.execute()
        self.assertNotIn('precision_ready', pipeline.context)
        model = onnx.load(str(graph))
        self.assertEqual({value.data_type
                          for value in model.graph.initializer},
                         {onnx.TensorProto.FLOAT})

    def test_explicit_int8_quantization_runs_calibration_and_parity(self):
        import onnx

        chain = self.export_chain() + [
            dict(type='QuantizeONNX'),
            dict(type='ConvertBackend'),
            dict(type='ValidateBackend'),
            dict(type='PackageBundle')
        ]
        pipeline = self.write_config(
            chain,
            backend_config=dict(type='pipeline_test_cpu', precision='int8'),
            validation=dict(atol=.05, rtol=.05, min_cosine_similarity=.999))
        manifest = verify_bundle(pipeline.execute())
        self.assertTrue(manifest.validation['passed'])
        model = onnx.load(str(pipeline.context['graph_path']))
        operators = {node.op_type for node in model.graph.node}
        self.assertTrue({'QuantizeLinear', 'DequantizeLinear'} <= operators)

    @unittest.skipUnless(
        importlib.util.find_spec('onnxsim') is not None, 'Optional ONNX simplifier')
    def test_explicit_optimizer_produces_graph_with_real_parity(self):
        pipeline = self.write_config(self.export_chain() + [
            dict(
                type='OptimizeONNX',
                output='simplified.onnx',
                options=dict(external_data=False)),
            dict(type='ConvertBackend'),
            dict(type='ValidateBackend'),
            dict(type='PackageBundle')
        ])
        manifest = verify_bundle(pipeline.execute())
        self.assertTrue(manifest.validation['passed'])
        self.assertEqual(pipeline.context['graph_path'],
                         self.root / 'work/simplified.onnx')

    def test_complete_source_backend_parity_and_bundle(self):
        chain = self.export_chain() + [
            dict(type='ConvertBackend'),
            dict(type='ValidateBackend'),
            dict(type='PackageBundle')
        ]
        pipeline = self.write_config(chain)
        bundle = pipeline.execute()
        manifest = verify_bundle(bundle)
        self.assertTrue(manifest.validation['passed'])
        self.assertEqual(manifest.backend_metadata['providers'],
                         ['CPUExecutionProvider'])
        self.assertEqual(manifest.metadata['transforms'], chain)
        report = json.loads((self.root / 'work/result.json').read_text())
        self.assertTrue(report['numerical_passed'])
        self.assertEqual([stage['name'] for stage in report['pipeline']], [
            'prepare_model', 'rewrite', 'export_ir', 'build_backend', 'validate',
            'package'
        ])

    def test_custom_transform_and_strict_numerical_failure(self):
        chain = self.export_chain() + [
            dict(type='ConvertBackend'),
            dict(type='ReferenceDrift', offset=1.),
            dict(type='ValidateBackend'),
            dict(type='PackageBundle')
        ]
        pipeline = self.write_config(chain)
        with self.assertRaises(ValidationError):
            pipeline.execute()
        report = json.loads((self.root / 'work/result.json').read_text())
        self.assertFalse(report['workflow_completed'])
        self.assertFalse(report['numerical_passed'])
        self.assertEqual(pipeline.context['drift'], 1.)
        self.assertEqual(report['stages'][-1]['name'], 'validate')
        self.assertFalse((self.root / 'work/bundle').exists())
        pipeline = self.write_config(chain, validation=dict(fail_on_error=False))
        manifest = verify_bundle(pipeline.execute())
        self.assertFalse(manifest.validation['passed'])

    def test_rebuilding_backend_discards_previous_validation(self):
        chain = self.export_chain() + [
            dict(type='ConvertBackend'),
            dict(type='ValidateBackend'),
            dict(type='ConvertBackend'),
            dict(type='PackageBundle')
        ]
        pipeline = self.write_config(chain)
        manifest = verify_bundle(pipeline.execute())
        self.assertFalse(manifest.validation['passed'])
        self.assertIsNone(
            json.loads(
                (self.root / 'work/result.json').read_text())['numerical_passed'])

    def test_export_after_hook_failure_restores_rewrite_scope(self):
        rewriter = FunctionRewriter()
        rewriter.register_rewriter(f'{__name__}.rewrite_target')(
            lambda ctx, value: ctx.origin_func(value) * 10)
        original = rewrite_target
        pipeline = self.write_config(self.export_chain())

        def fail_hook(**kwargs):
            self.assertEqual(rewrite_target(1), 20)
            raise LookupError('export hook failure')

        pipeline.pipeline_manager.add_hook('export_ir', fail_hook)
        with patch('fluxhermes.transforms.stages.FUNCTION_REWRITER', rewriter):
            with self.assertRaisesRegex(LookupError, 'export hook'):
                pipeline.execute()
        self.assertIs(rewrite_target, original)

    def test_isolated_export_reloads_exact_snapshot_and_merges_records(self):
        import numpy as np

        chain = [
            dict(type='IsolatedExport', transforms=self.export_chain()),
            dict(type='ConvertBackend'),
            dict(type='PrepareReference'),
            dict(type='ValidateBackend'),
            dict(type='PackageBundle')
        ]
        config = dict(
            type='python',
            task='custom',
            device='cpu',
            python_paths=[str(Path(__file__).parent)],
            source=dict(
                type='python',
                factory=f'{__name__}:tiny_factory',
                kwargs=dict(checkpoint=str(self.checkpoint), random_inputs=True)))
        pipeline = self.write_config(chain, codebase_config=config)
        manifest = verify_bundle(pipeline.execute())
        self.assertTrue(manifest.validation['passed'])
        state = json.loads((self.root / 'work/export-stage.json').read_text())
        self.assertEqual(state['run_id'], pipeline.context['run_id'])
        self.assertNotEqual(state['pid'], os.getpid())
        self.assertEqual(_PREPARES, 1)  # Only the explicit reference reload runs here.
        with np.load(
                self.root / 'work/export-inputs.npz', allow_pickle=False) as arrays:
            np.testing.assert_array_equal(
                pipeline.context['prepared'].inputs['x'].numpy(), arrays['x'])
        self.assertEqual(
            [stage['name'] for stage in pipeline.pipeline_manager.report()], [
                'isolated_export', 'prepare_model', 'rewrite', 'export_ir',
                'build_backend', 'prepare_reference', 'validate', 'package'
            ])

    def test_isolated_failure_preserves_parent_and_child_diagnostics(self):
        pipeline = self.write_config([
            dict(
                type='IsolatedExport',
                transforms=self.export_chain() + [dict(type='FailTransform')])
        ])
        with self.assertRaisesRegex(ConfigurationError, 'Export worker failed'):
            pipeline.execute()
        report = json.loads((self.root / 'work/result.json').read_text())
        self.assertFalse(report['workflow_completed'])
        self.assertEqual(report['stages'][0]['status'], 'failed')
        self.assertEqual(report['stages'][-1]['name'], 'fail_transform')
        self.assertEqual(report['stages'][-1]['status'], 'failed')

    def test_stale_worker_report_cannot_satisfy_new_run(self):
        pipeline = self.write_config(
            [dict(type='IsolatedExport', transforms=self.export_chain())])
        pipeline.config.work_dir.mkdir()
        (pipeline.config.work_dir / 'export-stage.json').write_text(
            json.dumps(dict(run_id='previous-run', passed=True, stages=[])))
        with patch(
                'fluxhermes.transforms.stages.subprocess.run',
                return_value=SimpleNamespace(returncode=0)):
            with self.assertRaisesRegex(ConfigurationError, 'different run'):
                pipeline.execute()


if __name__ == '__main__':
    unittest.main()
