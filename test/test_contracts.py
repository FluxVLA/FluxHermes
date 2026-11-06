import unittest
from pathlib import Path

from fluxhermes.contracts import DeploymentConfig, TensorSpec
from fluxhermes.errors import ConfigurationError


class TensorSpecTest(unittest.TestCase):

    def test_static_tensor(self):
        spec = TensorSpec(name='images', dtype='float32', shape=(1, 3, 224, 224))
        self.assertEqual(spec.shape, (1, 3, 224, 224))

    def test_dynamic_tensor_requires_profile(self):
        with self.assertRaises(ConfigurationError):
            TensorSpec(name='tokens', dtype='int64', shape=(1, -1))

    def test_profile_order_is_validated(self):
        with self.assertRaises(ConfigurationError):
            TensorSpec(
                name='tokens',
                dtype='int64',
                shape=(1, -1),
                min_shape=(1, 16),
                opt_shape=(1, 8),
                max_shape=(1, 32),
            )


class DeploymentConfigTest(unittest.TestCase):

    def test_normalizes_relative_work_dir(self):
        path = Path('/tmp/configs/deploy.py')
        config = DeploymentConfig.from_dict(
            {
                'name': 'test',
                'work_dir': '../work',
                'source': {'type': 'python', 'factory': 'x:y'},
                'inputs': [
                    {'name': 'x', 'dtype': 'float32', 'shape': (1, 2)},
                ],
                'outputs': [
                    {'name': 'y', 'dtype': 'float32', 'shape': (1, 2)},
                ],
            },
            config_path=path,
        )
        self.assertEqual(config.work_dir, Path('/tmp/work').resolve())

    def test_rejects_unknown_precision(self):
        with self.assertRaises(ConfigurationError):
            DeploymentConfig.from_dict({
                'name': 'test',
                'work_dir': '/tmp/work',
                'source': {'type': 'python', 'factory': 'x:y'},
                'inputs': [
                    {'name': 'x', 'dtype': 'float32', 'shape': (1, 2)},
                ],
                'outputs': [
                    {'name': 'y', 'dtype': 'float32', 'shape': (1, 2)},
                ],
                'tensorrt': {'precision': 'int4'},
            })

    def test_resolves_timing_cache_relative_to_config(self):
        config = DeploymentConfig.from_dict(
            {
                'name': 'test',
                'work_dir': '../work',
                'source': {'type': 'python', 'factory': 'x:y'},
                'inputs': [
                    {'name': 'x', 'dtype': 'float32', 'shape': (1, 2)},
                ],
                'outputs': [
                    {'name': 'y', 'dtype': 'float32', 'shape': (1, 2)},
                ],
                'tensorrt': {'timing_cache': '../cache/model.timing'},
            },
            config_path=Path('/tmp/configs/deploy.py'),
        )
        self.assertEqual(
            config.tensorrt['timing_cache'],
            str(Path('/tmp/cache/model.timing').resolve()))


if __name__ == '__main__':
    unittest.main()
