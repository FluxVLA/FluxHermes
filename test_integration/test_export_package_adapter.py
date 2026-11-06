"""Real installed-SDK integration; no training imports or mocked models."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from fluxvla_export import PreparedDeployment
from fluxvla_export.config import load_config as load_sdk_config
from fluxvla_export.recipes.pi05.model import build_pi05_model
from safetensors.torch import save_file

from fluxhermes.adapters.fluxvla_export import (FluxVLAExportAdapter,
                                                tensor_specs_from_sdk)
from fluxhermes.apis import build_task_processor
from fluxhermes.codebase import get_codebase
from fluxhermes.contracts import DeploymentConfig
from fluxhermes.errors import ArtifactError, ConfigurationError


def _tiny_sdk_document():
    """Explicit synthetic test data, not a training config or model replica."""
    gemma = dict(
        type='condition_gemma', hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, hidden_activation='gelu_pytorch_tanh',
        max_position_embeddings=64, vocab_size=64, rms_norm_eps=1e-6,
        rope_theta=10000.0, attention_bias=False, attention_dropout=0.0,
        use_cache=True, use_adarms=True, adarms_cond_dim=None,
        bos_token_id=2, eos_token_id=1, pad_token_id=0)

    def linear(in_dim, out_dim):
        return dict(type='linear', in_dim=in_dim, out_dim=out_dim, bias=True)

    return dict(
        schema_version=1, recipe='pi05-v1',
        checkpoint=dict(
            format='fluxvla-native', kind='dense', sha256='0' * 64),
        inference=dict(mode='plain', num_steps=2, rtc=False, solver='euler'),
        inputs=dict(
            format='safetensors', sha256='0' * 64,
            tensors={
                'images': dict(dtype='float32', shape=[2, 6, 8, 8]),
                'img_masks': dict(dtype='bool', shape=[2, 2]),
                'lang_tokens': dict(dtype='int64', shape=[2, 4]),
                'lang_masks': dict(dtype='bool', shape=[2, 4]),
                'states': dict(dtype='float32', shape=[2, 4]),
                'noise': dict(dtype='float32', shape=[2, 3, 4]),
            }),
        model=dict(
            type='pi05', action_dim=4, action_horizon=3, state_dim=4,
            state_projection=False, proj_width=16,
            backbone=gemma, expert=dict(gemma, adarms_cond_dim=16),
            vision=dict(
                type='siglip', hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_channels=3,
                image_size=8, patch_size=4,
                hidden_activation='gelu_pytorch_tanh', layer_norm_eps=1e-6,
                attention_dropout=0.0, vision_use_head=False),
            projector=linear(16, 16), action_in_proj=linear(4, 16),
            action_out_proj=linear(16, 4), time_mlp_in=linear(16, 16),
            time_mlp_out=linear(16, 16)),
        precision=dict(
            parameter_dtype='float32', autocast_dtype=None,
            attention_backend='sdpa', adarms_fp32=True,
            attention_math_fp32=True, backbone_fp32_layers=[0],
            expert_fp32_layers=[0], flow_fp32=True, residual_fp32=True,
            vision_residual_fp32=True, vision_stem_fp32=True),
        preprocessing=dict(
            id='synthetic-hermes-sdk-integration-v1',
            image_transform_id='seeded-packed-two-view-rgb',
            normalization_id='already-normalized-test-tensors',
            normalization_sha256=hashlib.sha256(
                b'test normalization').hexdigest(),
            tokenizer_id='literal-test-token-ids',
            tokenizer_sha256=hashlib.sha256(b'test tokenizer').hexdigest(),
            prompt_id='literal-token-fixture', state_in_prompt=False,
            state_semantics='synthetic-pi05-unused-continuous-state',
            action_semantics='synthetic-padded-four-dimensional-actions',
            view_order=['synthetic-view-0', 'synthetic-view-1']))


class ExportPackageAdapterTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name).resolve()
        cls.json_path = cls.root / 'deployment.json'
        cls.checkpoint_path = cls.root / 'weights.safetensors'
        cls.inputs_path = cls.root / 'inputs.safetensors'
        document = _tiny_sdk_document()
        cls.json_path.write_text(json.dumps(document), encoding='utf-8')
        initial = load_sdk_config(cls.json_path, recipe='pi05-v1')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9127)
            cls.model = build_pi05_model(initial)
        generator = torch.Generator(device='cpu').manual_seed(3271)
        cls.tensors = {}
        for spec in initial.input_specs:
            dtype = getattr(torch, spec.dtype)
            if dtype == torch.bool:
                value = torch.ones(spec.shape, dtype=dtype)
                value[0, -1] = False
            elif dtype == torch.int64:
                value = torch.randint(
                    0, initial.model.vocab_size, spec.shape,
                    generator=generator)
            else:
                value = torch.randn(spec.shape, generator=generator)
            cls.tensors[spec.name] = value
        save_file(dict(cls.model.state_dict()), cls.checkpoint_path)
        save_file(cls.tensors, cls.inputs_path)
        for key, path in (('checkpoint', cls.checkpoint_path),
                          ('inputs', cls.inputs_path)):
            document[key]['sha256'] = hashlib.sha256(
                path.read_bytes()).hexdigest()
        cls.json_path.write_text(json.dumps(document), encoding='utf-8')
        cls.sdk_config = load_sdk_config(cls.json_path, recipe='pi05-v1')
        cls.document = document

    def deployment(self, **changes):
        inputs = [spec.to_dict() for spec in tensor_specs_from_sdk(
            self.sdk_config.input_specs)]
        document = dict(
            name='real-tiny-pi05', work_dir=str(self.root / 'work'),
            codebase_config=dict(
                type='fluxvla', task='export_package', device='cpu',
                source=dict(
                    type='fluxvla_export', recipe='pi05-v1',
                    deployment_config='deployment.json',
                    checkpoint='weights.safetensors',
                    inputs='inputs.safetensors')),
            backend_config=dict(type='tensorrt', precision='fp32'),
            ir_config=dict(type='onnx', opset=18), inputs=inputs,
            outputs=[dict(name='actions', dtype='float32', shape=(2, 3, 4))],
            optimization=dict(fp16_conversion=False),
            quantization=dict(mode='none'))
        document.update(changes)
        return DeploymentConfig.from_dict(
            document, config_path=self.root / 'hermes.py')

    def test_registered_task_preserves_legacy_and_native_graphs(self):
        codebase = get_codebase('fluxvla')
        self.assertIn('predict_action', codebase.task_registry.module_dict)
        self.assertIn('export_package', codebase.task_registry.module_dict)
        config = self.deployment()
        processor = build_task_processor(config, device='cpu')
        prepared = processor.prepare_model()
        self.assertIsInstance(prepared, PreparedDeployment)
        self.assertEqual(prepared.recipe, 'pi05-v1')
        self.assertEqual(tuple(prepared.graphs), ('prefill', 'denoise'))
        self.assertEqual(tuple(prepared.policy_inputs),
                         tuple(spec.name for spec in config.inputs))
        self.assertEqual(prepared.metadata['precision_policy'],
                         self.document['precision'])
        self.assertEqual(
            prepared.metadata['runtime']['entrypoint'],
            'fluxvla_export.recipes.pi05.executor:build_executor')
        self.assertEqual(processor.runtime_metadata(), {
            'codebase': 'fluxvla', 'task': 'export_package',
            'recipe': 'pi05-v1'})
        with torch.inference_mode():
            expected = self.model(**self.tensors)
        outputs = prepared.run_reference()
        self.assertEqual(tuple(outputs), ('actions',))
        torch.testing.assert_close(outputs['actions'], expected)
        self.assertTrue(bool(torch.isfinite(outputs['actions']).all()))

    def test_import_and_registration_are_dependency_light(self):
        code = '''
import sys
before = tuple(sys.path)
from fluxhermes.adapters.fluxvla_export import FluxVLAExportAdapter
from fluxhermes.codebase import get_codebase
get_codebase('fluxvla')
assert tuple(sys.path) == before
for name in ('torch', 'transformers', 'fluxvla_export', 'fluxvla', 'mmengine'):
    assert name not in sys.modules, name
'''
        subprocess.run([sys.executable, '-I', '-c', code], check=True)

    def test_real_preparation_never_imports_training_or_injects_paths(self):
        config = self.deployment()
        before = tuple(sys.path)
        prepared = FluxVLAExportAdapter(config.source, config).prepare('cpu')
        self.assertIsInstance(prepared, PreparedDeployment)
        self.assertEqual(tuple(sys.path), before)
        self.assertFalse(any(
            name == 'fluxvla' or name.startswith('fluxvla.')
            or name == 'mmengine' or name.startswith('mmengine.')
            for name in sys.modules))

    def test_relative_paths_are_resolved_from_hermes_config(self):
        config = self.deployment()
        adapter = FluxVLAExportAdapter(config.source, config)
        self.assertEqual(adapter._artifact_path('deployment_config'),
                         self.json_path)
        self.assertEqual(adapter._artifact_path('checkpoint'),
                         self.checkpoint_path)
        self.assertEqual(adapter._artifact_path('inputs'), self.inputs_path)
        with self.assertRaisesRegex(ConfigurationError, 'relative paths'):
            adapter = FluxVLAExportAdapter(
                config.source, replace(config, config_path=None))
            adapter.prepare('cpu')

    def test_relative_and_tilde_paths_in_clean_process(self):
        code = '''
import sys
from pathlib import Path
from fluxhermes.adapters.fluxvla_export import FluxVLAExportAdapter
from fluxhermes.contracts import DeploymentConfig
config = DeploymentConfig.from_dict({
    'name': 'paths', 'work_dir': '.',
    'codebase_config': {'type': 'fluxvla', 'task': 'export_package',
        'source': {'recipe': 'pi05-v1',
                   'deployment_config': '~/deployment.json',
                   'checkpoint': 'weights.safetensors',
                   'inputs': 'inputs.safetensors'}},
    'backend_config': {'type': 'tensorrt', 'precision': 'fp32'},
    'inputs': [{'name': 'x', 'dtype': 'float32', 'shape': (1,)}],
    'outputs': [{'name': 'y', 'dtype': 'float32', 'shape': (1,)}],
}, config_path=Path(sys.argv[1]) / 'hermes.py')
adapter = FluxVLAExportAdapter(config.source, config)
root = Path(sys.argv[1])
assert adapter._artifact_path('deployment_config') == root / 'deployment.json'
assert adapter._artifact_path('checkpoint') == root / 'weights.safetensors'
'''
        subprocess.run(
            [sys.executable, '-I', '-c', code, str(self.root)],
            env={**os.environ, 'HOME': str(self.root)}, check=True)

    def test_fp32_example_derives_specs_from_explicit_sdk_json(self):
        example = (Path(__file__).resolve().parents[1] / 'configs' / 'pi05'
                   / 'pi05_export_fp32.py')
        code = '''
import sys
from fluxhermes.config import load_config
config = load_config(sys.argv[1])
assert config.codebase['task'] == 'export_package'
assert config.precision == 'fp32'
assert config.optimization['fp16_conversion'] is False
assert config.ir['do_constant_folding'] is False
assert config.quantization['mode'] == 'none'
assert config.inputs[0].shape == (2, 6, 8, 8)
assert config.outputs[0].shape == (2, 3, 4)
assert 'torch' not in sys.modules
assert 'fluxvla' not in sys.modules
assert 'mmengine' not in sys.modules
'''
        environment = {
            **os.environ,
            'PI05_DEPLOYMENT_JSON': str(self.json_path),
            'PI05_CHECKPOINT': str(self.checkpoint_path),
            'PI05_EXPORT_INPUTS': str(self.inputs_path),
        }
        subprocess.run(
            [sys.executable, '-I', '-c', code, str(example)],
            env=environment, check=True)

    def test_fp32_example_requires_artifact_environment(self):
        example = (Path(__file__).resolve().parents[1] / 'configs' / 'pi05'
                   / 'pi05_export_fp32.py')
        environment = {
            key: value for key, value in os.environ.items()
            if key not in {'PI05_DEPLOYMENT_JSON', 'PI05_CHECKPOINT',
                           'PI05_EXPORT_INPUTS'}
        }
        code = '''
import sys
from fluxhermes.config import load_config
try:
    load_config(sys.argv[1])
except ValueError as error:
    assert 'PI05_DEPLOYMENT_JSON' in str(error), error
else:
    raise AssertionError('explicit artifact environment is required')
'''
        subprocess.run(
            [sys.executable, '-I', '-c', code, str(example)],
            env=environment, check=True)

    def test_rejects_source_checkout_and_training_configuration(self):
        config = self.deployment()
        for field in ('python_paths', 'fluxvla_root', 'model_section',
                      'sample_factory', 'strict', 'config'):
            with self.subTest(field=field):
                source = dict(config.source, **{field: 'not-supported'})
                with self.assertRaisesRegex(ConfigurationError, 'unknown='):
                    FluxVLAExportAdapter(source, config)
        source = dict(config.source, deployment_config='training.py')
        with self.assertRaisesRegex(ConfigurationError, 'deployment JSON'):
            FluxVLAExportAdapter(source, config).prepare('cpu')

    def test_rejects_missing_source_fields_and_legacy_source_type(self):
        config = self.deployment()
        for field in ('recipe', 'deployment_config', 'checkpoint', 'inputs'):
            with self.subTest(field=field):
                source = dict(config.source)
                del source[field]
                with self.assertRaisesRegex(ConfigurationError, 'missing='):
                    FluxVLAExportAdapter(source, config)
        with self.assertRaisesRegex(ConfigurationError, 'source.type'):
            FluxVLAExportAdapter(dict(config.source, type='fluxvla'), config)

    def test_rejects_global_fp16_int8_and_quantization(self):
        for precision in ('fp16', 'int8'):
            with self.subTest(precision=precision):
                with self.assertRaisesRegex(ConfigurationError, 'precision'):
                    config = self.deployment(backend_config=dict(
                        type='tensorrt', precision=precision))
                    FluxVLAExportAdapter(config.source, config).prepare('cpu')
        for option in (
            dict(optimization=dict(fp16_conversion=True)),
            dict(quantization=dict(mode='int8')),
            dict(quantization=dict(mode='qdq')),
        ):
            with self.subTest(option=option):
                with self.assertRaises(ConfigurationError):
                    config = self.deployment(**option)
                    FluxVLAExportAdapter(config.source, config).prepare('cpu')

    def test_bfloat16_policy_requires_constant_folding_disabled(self):
        document = copy.deepcopy(self.document)
        document['precision']['autocast_dtype'] = 'bfloat16'
        config = self.deployment()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bf16-deployment.json'
            path.write_text(json.dumps(document), encoding='utf-8')
            source = dict(config.source, deployment_config=str(path))
            incompatible = replace(
                config, ir=dict(config.ir, do_constant_folding=True))
            with self.assertRaisesRegex(
                    ConfigurationError, 'do_constant_folding=False'):
                FluxVLAExportAdapter(source, incompatible).prepare('cpu')
            compatible = replace(
                config, ir=dict(config.ir, do_constant_folding=False))
            prepared = FluxVLAExportAdapter(source, compatible).prepare('cpu')
            self.assertEqual(prepared.metadata['precision_policy'],
                             document['precision'])
            self.assertTrue(bool(torch.isfinite(
                prepared.run_reference()['actions']).all()))

    def test_requires_exact_policy_names_order_dtype_shape_and_profiles(self):
        config = self.deployment()
        replacements = (
            tuple(reversed(config.inputs)),
            (replace(config.inputs[0], name='renamed'), *config.inputs[1:]),
            (replace(config.inputs[0], dtype='float16'), *config.inputs[1:]),
            (replace(config.inputs[0], shape=(2, 3, 8, 8)),
             *config.inputs[1:]),
            (replace(config.inputs[0], min_shape=(2, 6, 8, 8),
                     opt_shape=(2, 6, 8, 8), max_shape=(2, 6, 8, 8)),
             *config.inputs[1:]),
        )
        for inputs in replacements:
            with self.subTest(inputs=inputs):
                changed = replace(config, inputs=inputs)
                with self.assertRaisesRegex(
                        ConfigurationError, 'Policy inputs'):
                    adapter = FluxVLAExportAdapter(changed.source, changed)
                    adapter.prepare('cpu')

    def test_requires_exact_policy_output_contract(self):
        config = self.deployment()
        for spec in (replace(config.outputs[0], name='other'),
                     replace(config.outputs[0], dtype='float16'),
                     replace(config.outputs[0], shape=(2, 2, 4))):
            with self.subTest(spec=spec):
                changed = replace(config, outputs=(spec, ))
                with self.assertRaisesRegex(
                        ConfigurationError, 'Policy outputs'):
                    adapter = FluxVLAExportAdapter(changed.source, changed)
                    adapter.prepare('cpu')

    def test_preserves_sdk_dynamic_profiles_during_conversion(self):
        from fluxvla_export import TensorSpec

        source = TensorSpec(
            'x', 'bfloat16', (1, -1),
            min_shape=(1, 2), opt_shape=(1, 4), max_shape=(1, 8))
        actual, = tensor_specs_from_sdk((source, ))
        for field in ('name', 'dtype', 'shape', 'min_shape', 'opt_shape',
                      'max_shape'):
            self.assertEqual(getattr(actual, field), getattr(source, field))

    def test_corrupt_real_artifact_is_not_hidden_or_unpickled(self):
        config = self.deployment()
        with tempfile.TemporaryDirectory() as directory:
            changed_path = Path(directory) / 'weights.safetensors'
            raw = self.checkpoint_path.read_bytes()
            changed_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            source = dict(config.source, checkpoint=str(changed_path))
            with self.assertRaisesRegex(ArtifactError, 'SHA-256 mismatch'):
                FluxVLAExportAdapter(source, config).prepare('cpu')


if __name__ == '__main__':
    unittest.main()
