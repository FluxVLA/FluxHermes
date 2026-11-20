"""Compatibility for the implicit pipeline in main@3db1262 deployment configs."""
import copy
import runpy
import tempfile
import unittest
from pathlib import Path

from fluxhermes.config import load_config
from fluxhermes.contracts import DeploymentConfig
from fluxhermes.errors import ConfigurationError
from fluxhermes.pipeline import CompilePipeline

ROOT = Path(__file__).resolve().parents[1]


def legacy_config(precision='fp16'):
    # The current PI05 templates retain the historical tensor/source contract.
    # Restore the exact control fields removed when explicit pipelines were added.
    data = runpy.run_path(str(ROOT /
                              f'configs/pi05/pi05_aloha_{precision}.py'))['deployment']
    data.pop('pipeline')
    data['optimization']['simplify'] = False
    data.setdefault('quantization',
                    {})['mode'] = 'int8' if precision == 'int8' else 'none'
    data['validation']['enabled'] = True
    return data


class LegacyConfigTest(unittest.TestCase):

    def test_old_pi05_configs_load_and_plan_without_a_pipeline(self):
        for precision in ('fp16', 'int8'):
            with self.subTest(
                    precision=precision), tempfile.TemporaryDirectory() as tmp:
                data = legacy_config(precision)
                path = Path(tmp) / 'deploy.py'
                path.write_text('deployment = ' + repr(data), encoding='utf-8')
                config = load_config(path)
                pipeline = CompilePipeline(config)
                expected = ['PrepareModel', 'Rewrite', 'OptimizeONNX', 'LowerPrecision']
                if precision == 'int8':
                    expected.append('QuantizeONNX')
                expected.extend(['ConvertBackend', 'ValidateBackend', 'PackageBundle'])
                self.assertEqual([step['type'] for step in pipeline.plan()], expected)
                self.assertEqual(config.pipeline[1]['transforms'],
                                 [dict(type='ExportONNX')])
                # OptimizeONNX defaults to simplify=True; the old pipeline did not.
                self.assertFalse(config.pipeline[2]['options']['simplify'])
                self.assertEqual(config.precision, precision)
                self.assertFalse(config.work_dir.exists())
                self.assertEqual(
                    DeploymentConfig.from_dict(config.to_dict()).pipeline,
                    config.pipeline)

    def test_legacy_defaults_do_not_mutate_the_callers_configuration(self):
        data = legacy_config()
        data.pop('optimization')
        data.pop('quantization')
        data.pop('validation')
        original = copy.deepcopy(data)
        config = DeploymentConfig.from_dict(data)
        steps = [step['type'] for step in CompilePipeline(config).plan()]
        self.assertIn('ValidateBackend', steps)
        self.assertNotIn('QuantizeONNX', steps)
        self.assertFalse(config.pipeline[2]['options']['simplify'])
        self.assertEqual(data, original)

    def test_legacy_simplification_and_disabled_validation_are_honored(self):
        data = legacy_config()
        data['optimization']['simplify'] = True
        data['validation']['enabled'] = False
        config = DeploymentConfig.from_dict(data)
        self.assertTrue(config.pipeline[2]['options']['simplify'])
        # Retain the execution slot so a legacy hook can enable validation later;
        # plan() still reflects the current switch, and CPU tests assert execution.
        self.assertIn('ValidateBackend', [step['type'] for step in config.pipeline])
        self.assertNotIn('ValidateBackend',
                         [step['type'] for step in CompilePipeline(config).plan()])
        self.assertEqual(config.pipeline[-1]['type'], 'PackageBundle')

    def test_legacy_quantization_mode_is_validated_before_building(self):
        for precision, mode, message in (('fp16', 'int4',
                                          'Unsupported quantization mode'),
                                         ('int8', 'none',
                                          'requires quantization.mode=int8'),
                                         ('fp16', 'int8',
                                          'requires backend precision=int8')):
            with self.subTest(precision=precision, mode=mode):
                data = legacy_config(precision)
                data['quantization']['mode'] = mode
                with self.assertRaisesRegex(ConfigurationError, message):
                    CompilePipeline(DeploymentConfig.from_dict(data))

    def test_explicit_pipeline_takes_precedence_over_legacy_switches(self):
        data = legacy_config('int8')
        data['pipeline'] = [dict(type='PrepareModel')]
        data['quantization']['mode'] = 'none'
        config = DeploymentConfig.from_dict(data)
        self.assertEqual(config.pipeline, tuple(data['pipeline']))
        self.assertEqual([step['type'] for step in CompilePipeline(config).plan()],
                         ['PrepareModel'])

    def test_explicit_empty_or_invalid_pipelines_still_fail(self):
        for value in ([], (), None, 'PrepareModel'):
            with self.subTest(pipeline=value):
                data = legacy_config()
                data['pipeline'] = value
                with self.assertRaises(ConfigurationError):
                    CompilePipeline(DeploymentConfig.from_dict(data))


if __name__ == '__main__':
    unittest.main()
