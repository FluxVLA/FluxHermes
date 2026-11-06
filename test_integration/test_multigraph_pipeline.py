"""Actual SDK -> two ONNX graphs -> ORT -> relocated robot bundle tests."""

import copy
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import test_export_package_adapter as fixtures
import torch
from safetensors.torch import save_file

from fluxhermes.apis import build_backend_wrapper
from fluxhermes.bundle import verify_bundle
from fluxhermes.config import load_config
from fluxhermes.contracts import BundleManifest, DeploymentConfig
from fluxhermes.errors import ArtifactError, ConfigurationError
from fluxhermes.pipeline import CompilePipeline
from fluxhermes.runtime.policy import RobotPolicy


class MultiGraphPipelineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Reuse real model+checkpoint fixture creation, not adapter test methods.
        fixtures.ExportPackageAdapterTest.setUpClass.__func__(cls)
        case = cls()
        config = fixtures.ExportPackageAdapterTest.deployment(
            case,
            backend_config={
                'type': 'onnxruntime',
                'precision': 'fp32',
                'device': 'cpu'
            },
            validation={
                'enabled': True,
                'atol': 1e-5,
                'rtol': 1e-5,
                'min_cosine_similarity': 0.99999
            },
            package={'include_onnx': True},
        )
        config.config_path.write_text('deployment = ' + repr(config.to_dict()) + '\n')
        cls.config = load_config(config.config_path)
        cls.bundle = CompilePipeline(cls.config).execute()
        cls.manifest = verify_bundle(cls.bundle)
        with torch.inference_mode():
            cls.expected = cls.model(**cls.tensors)

    def test_full_pipeline_reports_real_cpu_graph_and_policy_validation(self):
        manifest = self.manifest
        self.assertEqual(manifest.backend, 'onnxruntime')
        self.assertTrue(manifest.validation['passed'])
        self.assertEqual(set(manifest.validation['graphs']), {'prefill', 'denoise'})
        self.assertEqual(
            set(manifest.backend_metadata['graphs']), {'prefill', 'denoise'})
        for name, record in manifest.backend_metadata['graphs'].items():
            self.assertTrue(manifest.validation['graphs'][name]['passed'])
            self.assertEqual(record['metadata']['execution_provider'],
                             'CPUExecutionProvider')
            self.assertTrue(
                all(path.startswith(f'backend/{name}/') for path in record['files']))
        self.assertTrue(
            any(path.startswith('model/prefill/') for path in manifest.files))
        self.assertTrue(
            any(path.startswith('model/denoise/') for path in manifest.files))
        policy = RobotPolicy(self.bundle, device='cpu')
        snapshot = {name: value.clone() for name, value in self.tensors.items()}
        actual = policy.predict(self.tensors)
        torch.testing.assert_close(actual, self.expected, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.equal(actual, policy.predict(self.tensors)))
        for name in snapshot:
            self.assertTrue(torch.equal(snapshot[name], self.tensors[name]))

    def test_relocated_bundle_fresh_process_needs_no_model_or_training_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            bundle = root / 'bundle'
            shutil.copytree(self.bundle, bundle)
            inputs = root / 'inputs.safetensors'
            expected = root / 'expected.safetensors'
            save_file(self.tensors, inputs)
            save_file({'actions': self.expected}, expected)
            script = '''
import sys
import torch
from safetensors.torch import load_file
from fluxhermes.runtime.policy import RobotPolicy
policy = RobotPolicy(sys.argv[1], device='cpu')
actual = policy.predict(load_file(sys.argv[2]))
expected = load_file(sys.argv[3])['actions']
torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
for name in ('fluxvla', 'mmengine', 'transformers', 'timm',
             'fluxvla_export.recipes.pi05.model',
             'fluxvla_export.recipes.pi05.reference'):
    assert name not in sys.modules, name
print('relocated ONNX bundle: real inference without model/training imports')
'''
            subprocess.run([
                sys.executable, '-I', '-c', script,
                str(bundle),
                str(inputs),
                str(expected)
            ],
                           check=True,
                           cwd=root,
                           env={
                               **os.environ, 'HF_HUB_OFFLINE': '1'
                           })

    def test_real_unvalidated_bundle_is_refused_without_overwriting_existing(self):
        document = self.config.to_dict()
        document['work_dir'] = str(self.root / 'unvalidated-work')
        document['package']['output_dir'] = str(self.root / 'unvalidated-bundle')
        document['validation']['enabled'] = False
        path = self.root / 'unvalidated.py'
        path.write_text('deployment = ' + repr(document) + '\n')
        config = load_config(path)
        bundle = CompilePipeline(config).execute()
        self.assertFalse(verify_bundle(bundle).validation['passed'])
        with self.assertRaisesRegex(ArtifactError, 'unvalidated'):
            RobotPolicy(bundle, device='cpu')
        # Rebuilding at the same destination cannot remove even generated data.
        before = (bundle / 'manifest.json').read_bytes()
        with self.assertRaisesRegex(ArtifactError, 'refusing to replace'):
            CompilePipeline(config).execute()
        self.assertEqual((bundle / 'manifest.json').read_bytes(), before)

    def test_corrupted_graph_partition_is_rejected(self):
        data = self.manifest.to_dict()
        data['backend_metadata'] = copy.deepcopy(data['backend_metadata'])
        records = data['backend_metadata']['graphs']
        records['denoise']['files'] = records['prefill']['files']
        malformed = BundleManifest.from_dict(data)
        with self.assertRaisesRegex(ArtifactError, 'partition'):
            build_backend_wrapper(malformed, self.bundle, device='cpu')

    def test_unknown_source_path_injection_fails_at_config_not_import(self):
        data = self.config.to_dict()
        for section in ('source', 'codebase'):
            altered = copy.deepcopy(data)
            destination = altered['codebase_config']
            if section == 'source':
                destination = destination['source']
            destination['python_paths'] = ['/untrusted-training-checkout']
            with self.assertRaisesRegex(ConfigurationError, 'not allowed'):
                DeploymentConfig.from_dict(altered)

    def test_executor_metadata_and_backend_device_errors_are_explicit(self):
        policy = RobotPolicy(self.bundle, device='cpu')
        bad_inputs = dict(self.tensors)
        bad_inputs['lang_tokens'] = torch.full_like(bad_inputs['lang_tokens'], 64)
        with self.assertRaisesRegex(ValueError, 'vocab|token'):
            policy.predict(bad_inputs)
        with self.assertRaisesRegex(ConfigurationError, 'cpu'):
            RobotPolicy(self.bundle, device='cuda:0')


if __name__ == '__main__':
    unittest.main()
