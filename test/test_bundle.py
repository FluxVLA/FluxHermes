import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fluxhermes.bundle import create_bundle, verify_bundle
from fluxhermes.contracts import BundleManifest, TensorSpec
from fluxhermes.errors import ArtifactError
from fluxhermes.utils import write_json


def manifest():
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
        validation={'passed': True},
        files={},
    )


class BundleTest(unittest.TestCase):

    def test_create_verify_and_detect_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / 'source.plan'
            engine.write_bytes(b'engine')
            bundle = create_bundle(
                root / 'bundle',
                manifest(),
                {'backend/model.plan': engine},
            )
            loaded = verify_bundle(bundle)
            self.assertTrue(loaded.validation['passed'])
            (bundle / 'backend/model.plan').write_bytes(b'changed')
            with self.assertRaises(ArtifactError):
                verify_bundle(bundle)

    def test_copies_asset_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / 'source.plan'
            engine.write_bytes(b'engine')
            assets = root / 'processor'
            assets.mkdir()
            (assets / 'stats.json').write_text('{}', encoding='utf-8')
            bundle = create_bundle(
                root / 'bundle',
                manifest(),
                {'backend/model.plan': engine},
                assets=[assets],
            )
            verify_bundle(bundle)
            self.assertTrue((bundle / 'assets/processor/stats.json').is_file())

    def test_rejects_unsafe_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / 'source.plan'
            engine.write_bytes(b'engine')
            with self.assertRaises(ArtifactError):
                create_bundle(
                    root / 'bundle',
                    manifest(),
                    {'../outside.plan': engine},
                )

    def test_reads_legacy_single_engine_manifest(self):
        value = manifest().to_dict()
        value['schema_version'] = 1
        value['engine'] = {
            'path': 'backend/model.plan',
            'tensorrt_version': '10.0',
        }
        value.pop('backend_files')
        value.pop('backend_metadata')
        loaded = BundleManifest.from_dict(value)
        self.assertEqual(loaded.backend_files, ('backend/model.plan',))

    def test_rejects_backend_file_without_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            (root / 'backend').mkdir()
            (root / 'backend/model.plan').write_bytes(b'engine')
            write_json(root / 'manifest.json', manifest().to_dict())
            with self.assertRaisesRegex(ArtifactError, 'not covered'):
                verify_bundle(root)


if __name__ == '__main__':
    unittest.main()
