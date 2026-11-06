"""Real BF16-storage SDK integration through Hermes, without model replicas."""

import copy
import hashlib
import json
import unittest

import onnx
import test_export_package_adapter as fixtures
import torch
from fluxvla_export.config import config_to_document
from fluxvla_export.config import load_config as load_sdk_config
from fluxvla_export.recipes.pi05.model import build_pi05_model
from safetensors.torch import save_file

from fluxhermes.adapters.fluxvla_export import FluxVLAExportAdapter
from fluxhermes.exporters import export_onnx
from fluxhermes.multigraph import _export_model
from fluxhermes.pipeline import CompilePipeline
from fluxhermes.runtime.policy import RobotPolicy
from fluxhermes.validation import compare_outputs


class BF16CheckpointStorageTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fixtures.ExportPackageAdapterTest.setUpClass.__func__(cls)
        cls.stored_state = {
            key: value.to(torch.bfloat16) if value.is_floating_point() else value
            for key, value in cls.model.state_dict().items()
        }
        cls.stored_path = cls.root / 'bf16-storage.safetensors'
        save_file(cls.stored_state, cls.stored_path)
        cls.stored_hash = hashlib.sha256(cls.stored_path.read_bytes()).hexdigest()

    def config_and_reference(self, compute):
        label = compute or 'float32'
        document = copy.deepcopy(self.document)
        document['checkpoint'].update(
            sha256=self.stored_hash, storage_dtype='bfloat16')
        document['precision']['autocast_dtype'] = compute
        path = self.root / f'storage-{label}.json'
        path.write_text(json.dumps(document))
        sdk_config = load_sdk_config(path, recipe='pi05-v1')
        with torch.random.fork_rng(devices=[]):
            native = build_pi05_model(sdk_config).eval()
        # This is the pre-existing PyTorch loading operation, not the SDK loader.
        native.load_state_dict(self.stored_state, strict=True)
        self.assertTrue(all(p.dtype == torch.float32 for p in native.parameters()))
        with torch.no_grad(), torch.autocast(
                'cpu', dtype=torch.bfloat16, enabled=compute is not None):
            expected = native(**self.tensors)
        config = fixtures.ExportPackageAdapterTest.deployment(
            self,
            work_dir=str(self.root / f'work-storage-{label}'),
            codebase_config=dict(
                type='fluxvla', task='export_package', device='cpu',
                source=dict(type='fluxvla_export', recipe='pi05-v1',
                            deployment_config=str(path),
                            checkpoint=str(self.stored_path),
                            inputs=str(self.inputs_path))),
            backend_config=dict(type='onnxruntime', device='cpu', precision='fp32'),
            validation=dict(enabled=True, fail_on_error=True, atol=1e-5,
                            rtol=1e-5, min_cosine_similarity=0.99999))
        return config, sdk_config, expected

    def test_storage_is_bound_and_reference_matches_native_loading(self):
        for compute in (None, 'bfloat16'):
            with self.subTest(compute=compute):
                config, sdk_config, expected = self.config_and_reference(compute)
                prepared = FluxVLAExportAdapter(config.source, config).prepare('cpu')
                actual = prepared.run_reference()['actions']
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                document = config_to_document(sdk_config)
                self.assertEqual(document['checkpoint']['storage_dtype'], 'bfloat16')
                encoded = json.dumps(document, sort_keys=True,
                                     separators=(',', ':')).encode()
                digest = hashlib.sha256(encoded).hexdigest()
                self.assertEqual(prepared.metadata['source_config_sha256'], digest)
                self.assertEqual(prepared.metadata['checkpoint_sha256'],
                                 self.stored_hash)
                self.assertEqual(
                    prepared.metadata['precision_policy']['parameter_dtype'], 'float32')
                if compute is not None:
                    for name, graph in prepared.graphs.items():
                        output = export_onnx(
                            _export_model(graph), self.root / f'stored-{name}.onnx',
                            graph.export_options)
                        onnx.checker.check_model(str(output), full_check=True)
                        self.assertTrue(graph.export_options['optimize'])
        self.assertEqual(hashlib.sha256(self.stored_path.read_bytes()).hexdigest(),
                         self.stored_hash)

    def test_bf16_storage_fp32_compute_normal_pipeline_and_reload(self):
        config, _, expected = self.config_and_reference(None)
        config.config_path.write_text('deployment = ' + repr(config.to_dict()) + '\n')
        bundle = CompilePipeline(config).execute()
        policy = RobotPolicy(bundle, device='cpu')
        self.assertTrue(policy.manifest.validation['passed'])
        result = compare_outputs(
            {'actions': expected}, {'actions': policy.predict(self.tensors)},
            atol=1e-5, rtol=1e-5, min_cosine_similarity=0.99999)
        self.assertTrue(result['passed'], result)
        self.assertEqual(hashlib.sha256(self.stored_path.read_bytes()).hexdigest(),
                         self.stored_hash)


if __name__ == '__main__':
    unittest.main()
