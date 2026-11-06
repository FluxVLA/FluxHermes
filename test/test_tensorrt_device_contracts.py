"""Dependency-free strict TensorRT profile and target metadata preflight."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from fluxhermes.backend.tensorrt.manager import TensorRTBackendManager
from fluxhermes.backend.tensorrt.wrapper import TensorRTWrapper
from fluxhermes.errors import ArtifactError, ConfigurationError
from fluxhermes.runtime.tensorrt import (TensorRTEngine, _validate_profile_index,
                                         _validate_target_metadata)


class TensorRTDevicePreflightTest(unittest.TestCase):

    def test_import_does_not_initialize_torch_or_tensorrt(self):
        code = '''
import sys
import fluxhermes.builders.tensorrt
import fluxhermes.backend.tensorrt.manager
import fluxhermes.backend.tensorrt.wrapper
import fluxhermes.runtime.tensorrt
assert 'torch' not in sys.modules
assert 'tensorrt' not in sys.modules
'''
        subprocess.run([sys.executable, '-I', '-c', code], check=True)

    def test_profile_indices_are_not_coerced(self):
        self.assertEqual(_validate_profile_index(0, 0), 0)
        self.assertEqual(_validate_profile_index(1, 2), 1)
        for value in (-1, True, False, 0.0, 1.9, '0', None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ConfigurationError, 'profile_index'):
                    TensorRTEngine('unused.plan', profile_index=value)
        for value, count in ((1, 0), (1, 1), (2, 2)):
            with self.subTest(value=value, count=count):
                with self.assertRaisesRegex(ConfigurationError, 'range'):
                    _validate_profile_index(value, count)

    def test_manager_does_not_coerce_profile_or_strict_target_flags(self):
        for value in ('0', True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ConfigurationError, 'profile_index'):
                    TensorRTBackendManager.build_wrapper(
                        (Path('unused.plan'), ), device='cuda:0',
                        profile_index=value)
        for value in ('false', 0, 1, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ConfigurationError, 'require_target_match'):
                    TensorRTBackendManager.build_wrapper(
                        (Path('unused.plan'), ), device='cuda:0',
                        require_target_match=value)

    def test_strict_target_metadata_requires_all_actual_facts(self):
        metadata = {
            'tensorrt_version': '10.0.0', 'device': 'metadata-test-device',
            'compute_capability': [8, 0], 'cuda_version': '12.6',
        }
        _validate_target_metadata(metadata)
        for field in metadata:
            with self.subTest(field=field):
                incomplete = dict(metadata)
                del incomplete[field]
                with self.assertRaisesRegex(ArtifactError, field):
                    _validate_target_metadata(incomplete)
        for capability in (None, [], [8], [8, 0, 1], [True, 0], [0, 0],
                           [8, -1], ['8', '0']):
            with self.subTest(capability=capability):
                with self.assertRaisesRegex(ArtifactError,
                                            'compute_capability'):
                    _validate_target_metadata(dict(
                        metadata, compute_capability=capability))

    def test_strict_wrappers_validate_before_deserialization(self):
        for metadata in (None, {}, {'tensorrt_version': '10.0.0'}):
            with self.subTest(metadata=metadata):
                for constructor in (TensorRTWrapper, TensorRTEngine):
                    with self.assertRaises(ArtifactError):
                        constructor(
                            'unused.plan', target_metadata=metadata,
                            require_target_match=True)


if __name__ == '__main__':
    unittest.main()
