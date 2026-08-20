import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fluxhermes.bundle import create_bundle
from fluxhermes.contracts import BundleManifest, TensorSpec
from fluxhermes.errors import ArtifactError
from fluxhermes.runtime import RobotPolicy, TensorRTEngine


class RuntimeSafetyTest(unittest.TestCase):

    @staticmethod
    def _manifest(validated=False):
        return BundleManifest(
            schema_version=2,
            name='demo',
            created_at=datetime.now(timezone.utc).isoformat(),
            backend='tensorrt',
            precision='fp16',
            backend_files=('backend/model.plan',),
            backend_metadata={'tensorrt_version': '10.0'},
            source={},
            inputs=(TensorSpec(name='x', dtype='float32', shape=(1, 2)),),
            outputs=(TensorSpec(name='y', dtype='float32', shape=(1, 2)),),
            runtime={'action_output': 'y'},
            validation={'passed': validated},
            files={},
        )

    def test_robot_policy_rejects_unvalidated_bundle_before_loading_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / 'model.plan'
            engine.write_bytes(b'not-a-real-engine')
            bundle = create_bundle(
                root / 'bundle',
                self._manifest(validated=False),
                {'backend/model.plan': engine},
            )
            with self.assertRaisesRegex(ArtifactError, 'unvalidated'):
                RobotPolicy(bundle)

    def test_target_check_rejects_tensorrt_version_mismatch(self):
        engine = TensorRTEngine.__new__(TensorRTEngine)
        engine.trt = types.SimpleNamespace(__version__='10.0.0')
        engine.torch = types.SimpleNamespace()
        engine.device = types.SimpleNamespace(index=0)
        with self.assertRaisesRegex(ArtifactError, 'TensorRT version mismatch'):
            engine._check_target({'tensorrt_version': '10.1.0'})


if __name__ == '__main__':
    unittest.main()
