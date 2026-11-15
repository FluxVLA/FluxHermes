"""Interface tests only. Tiny doubles are NOT full-model parity evidence."""
import copy
import runpy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mmengine.config import Config

from fluxhermes.adapters.groot_n17 import (_NAMES, _check_inputs, _check_specs,
                                           _make_wrapper, _native_checkpoint_keys,
                                           _native_model_config, _reference_runner,
                                           _verify_attention, prepare_groot_n17)
from fluxhermes.contracts import DeploymentConfig
from fluxhermes.errors import ConfigurationError

try:
    import torch
except ImportError:
    torch = None


def configuration():
    path = Path(
        __file__).resolve().parents[1] / 'configs/gr00t/gr00t_n17_libero_bf16.py'
    return copy.deepcopy(runpy.run_path(str(path))['deployment'])


def samples(length=148):
    return dict(
        images=np.zeros((1, 512, 1536), np.float32),
        lang_tokens=np.zeros((1, length), np.int64),
        position_ids=np.zeros((3, 1, length), np.int64),
        states=np.zeros((1, 1, 132), np.float32),
        embodiment_ids=np.zeros((1, ), np.int32),
        noise=np.zeros((1, 40, 132), np.float32))


class TinyAttention:

    def __init__(self):
        self.config = SimpleNamespace(_attn_implementation='sdpa')


class TinyVLM:

    def __init__(self):
        self.layers = [TinyAttention(), TinyAttention()]
        self.modes = []

    def named_modules(self):
        return [(str(i), item) for i, item in enumerate(self.layers)]

    def set_attn_implementation(self, mode):
        self.modes.append(mode)
        for layer in self.layers:
            layer.config._attn_implementation = mode


class ContractTests(unittest.TestCase):

    def setUp(self):
        self.specs = _check_specs(DeploymentConfig.from_dict(configuration()))

    def test_valid_dynamic_lengths(self):
        for length in (138, 142, 148, 151, 161, 180):
            _check_inputs(samples(length), self.specs)

    def test_bad_values(self):
        cases = []
        for length in (137, 181):
            cases.append(samples(length))
        for field, value in [('images', np.zeros((2, 512, 1536), np.float32)),
                             ('states', np.zeros((1, 1, 132), np.float64)),
                             ('position_ids', np.zeros((3, 1, 149), np.int64)),
                             ('noise', np.full((1, 40, 132), np.nan, np.float32))]:
            case = samples()
            case[field] = value
            cases.append(case)
        case = samples()
        case['lang_masks'] = np.ones((1, 148))
        cases.append(case)
        case = samples()
        del case['noise']
        cases.append(case)
        for case in cases:
            with self.assertRaises(ConfigurationError):
                _check_inputs(case, self.specs)

    def test_profile_and_dtype_rejected(self):
        for field, value in [('dtype', 'int32'), ('max_shape', (1, 162))]:
            config = configuration()
            config['inputs'][1][field] = value
            with self.assertRaises(ConfigurationError):
                _check_specs(DeploymentConfig.from_dict(config))

    def test_closed_factory_kwargs(self):
        with self.assertRaises(TypeError):
            prepare_groot_n17(
                deployment=None,
                device='cpu',
                fluxvla_root='.',
                config='x',
                checkpoint='y',
                sample_inputs='z',
                ignored=True)

    def test_config_export_flags(self):
        config = configuration()
        self.assertTrue(config['validation']['fail_on_error'])
        self.assertIn('samples_manifest', config['validation'])
        self.assertFalse(config['backend_config']['allow_tf32'])
        self.assertTrue(config['ir_config']['normalize_full_slices'])
        self.assertFalse(config['ir_config']['dynamo'])

    def test_mixed_attention_detected(self):
        vlm = TinyVLM()
        vlm.layers[1].config._attn_implementation = 'flash_attention_2'
        with self.assertRaisesRegex(RuntimeError, 'Attention 1'):
            _verify_attention(vlm, 'sdpa')

    def test_reference_keeps_saved_sdpa_mode(self):
        for failure in ('none', 'forward'):
            vlm = TinyVLM()

            class TinyWrapper:
                model = SimpleNamespace(vlm_backbone=SimpleNamespace(vlm=vlm))

                def __call__(self, *values):
                    self_test.assertEqual(vlm.layers[0].config._attn_implementation,
                                          'sdpa')
                    if failure == 'forward':
                        raise RuntimeError('forward failure')
                    return values[-1]

            self_test = self
            runner = _reference_runner(TinyWrapper(), self.specs,
                                       SimpleNamespace(inference_mode=nullcontext))
            if failure == 'none':
                self.assertEqual(runner(samples()).shape, (1, 40, 132))
            else:
                with self.assertRaises(RuntimeError):
                    runner(samples())
            self.assertTrue(
                all(layer.config._attn_implementation == 'sdpa'
                    for layer in vlm.layers))

    def test_native_fa2_reference_restores_sdpa_after_success_and_failure(self):
        for fail in (False, True):
            vlm = TinyVLM()

            def predict(*values):
                self.assertEqual(vlm.layers[0].config._attn_implementation,
                                 'flash_attention_2')
                if fail:
                    raise RuntimeError('reference failed')
                return values[-1]

            predict.model = SimpleNamespace(vlm_backbone=SimpleNamespace(vlm=vlm))
            runner = _reference_runner(predict, self.specs,
                                       SimpleNamespace(inference_mode=nullcontext),
                                       'flash_attention_2')
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'reference failed'):
                    runner(samples())
            else:
                runner(samples())
            self.assertEqual(vlm.modes, ['flash_attention_2', 'sdpa'])


class NativeConfigTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = Config(
            dict(
                model=dict(
                    type='GrootN17VLA',
                    model_path='/training/official',
                    processor_path='/training/processor',
                    load_metadata=True,
                    qwen3_runtime='compat_457',
                    vlm_backbone=dict(
                        type='GrootN17Qwen3Backbone',
                        use_flash_attention=False,
                        model_name='/training/Cosmos-Reason2-2B',
                        select_layer=16,
                        load_bf16=False,
                        reproject_vision=True,
                        trainable_params_fp32=False,
                        tune_llm=False,
                        tune_top_llm_layers=0,
                        tune_visual=False,
                        transformers_loading_kwargs=dict(
                            local_files_only=True, trust_remote_code=True)),
                    vla_head=dict(type='GrootN17ActionHead',
                                  tune_diffusion_model=True))))
        self.architecture = dict(
            model_type='qwen3_vl', text_config=dict(hidden_size=2048))
        path = self.root / 'configs/gr00tn17' / (
            'gr00tn17_qwen3vl_2b_libero_10_full_finetune.py')
        path.parent.mkdir(parents=True)
        Config(
            dict(
                model=dict(
                    vlm_backbone=dict(
                        type='GrootN17Qwen3Backbone',
                        model_config=self.architecture,
                        use_flash_attention=True,
                        load_bf16=False),
                    vla_head=dict(type='GrootN17ActionHead')))).dump(str(path))

    def test_saved_config_reuses_source_architecture_without_mutation(self):
        before = copy.deepcopy(self.cfg.to_dict())
        model, report = _native_model_config(self.cfg, self.root)
        self.assertEqual(self.cfg.to_dict(), before)
        self.assertIsNone(model['model_path'])
        self.assertIsNone(model['processor_path'])
        self.assertFalse(model['load_metadata'])
        self.assertFalse(model['load_pretrained_weights'])
        backbone = model['vlm_backbone']
        self.assertEqual(backbone['model_config'], self.architecture)
        self.assertEqual(backbone['select_layer'], 16)
        self.assertFalse(backbone['use_flash_attention'])
        self.assertFalse(backbone['load_bf16'])
        self.assertEqual(len(report['removed_legacy_backbone_options']), 6)
        self.assertTrue(Path(report['architecture_source']).is_file())

    def test_inline_architecture_is_not_replaced(self):
        inline = dict(model_type='qwen3_vl', text_config=dict(hidden_size=1024))
        self.cfg.model.vlm_backbone.model_config = inline
        model, report = _native_model_config(self.cfg, self.root / 'not-needed')
        self.assertEqual(model['vlm_backbone']['model_config'], inline)
        self.assertIsNone(report['architecture_source'])

    def test_wrong_architecture_and_attention_fail(self):
        for key, value in [('model_name', '/unknown/backbone'),
                           ('type', 'OtherBackbone'), ('use_flash_attention', None)]:
            cfg = copy.deepcopy(self.cfg)
            cfg.model.vlm_backbone[key] = value
            with self.subTest(key=key), self.assertRaises(ConfigurationError):
                _native_model_config(cfg, self.root)

    def test_unsupported_legacy_values_fail_even_with_inline_architecture(self):
        for key, value in [('model_name', '/unrelated/model'),
                           ('trainable_params_fp32', 'true'),
                           ('transformers_loading_kwargs', dict(torch_dtype='float16'))
                           ]:
            cfg = copy.deepcopy(self.cfg)
            cfg.model.vlm_backbone.model_config = self.architecture
            cfg.model.vlm_backbone[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ConfigurationError, key):
                _native_model_config(cfg, self.root)

    def test_flat_native_config_uses_source_components(self):
        cfg = Config(
            dict(
                model=dict(
                    type='GrootN17VLA',
                    assembly_runtime='native',
                    load_mode='native_safe',
                    processor_runtime='native',
                    backbone_model_path='/old/Cosmos-Reason2-2B',
                    use_flash_attention=False,
                    qwen3_runtime='compat_457')))
        original = copy.deepcopy(cfg.to_dict())
        model, report = _native_model_config(cfg, self.root)
        self.assertEqual(cfg.to_dict(), original)
        self.assertEqual(model['vlm_backbone']['model_config'], self.architecture)
        self.assertEqual(model['vla_head']['type'], 'GrootN17ActionHead')
        self.assertFalse(model['vlm_backbone']['use_flash_attention'])
        self.assertNotIn('processor_runtime', model)
        self.assertEqual(len(report['removed_legacy_model_options']), 4)
        cfg.model.assembly_runtime = 'official'
        with self.assertRaisesRegex(ConfigurationError, 'assembly_runtime'):
            _native_model_config(cfg, self.root)

    def test_native_fa2_and_training_precision_flags_are_preserved_or_recorded(self):
        self.cfg.model.vlm_backbone.use_flash_attention = True
        self.cfg.model.vlm_backbone.trainable_params_fp32 = True
        model, report = _native_model_config(self.cfg, self.root)
        self.assertTrue(model['vlm_backbone']['use_flash_attention'])
        self.assertNotIn('trainable_params_fp32', model['vlm_backbone'])
        self.assertTrue(
            report['removed_legacy_backbone_options']['trainable_params_fp32'])

    def test_unknown_backbone_options_are_not_silently_dropped(self):
        self.cfg.model.vlm_backbone.unsupported_option = True
        model, _ = _native_model_config(self.cfg, self.root)
        self.assertIn('unsupported_option', model['vlm_backbone'])


@unittest.skipIf(torch is None,
                 'Torch unavailable; real tensor boundary tests require Torch')
class TensorBoundaryTests(unittest.TestCase):

    def test_checkpoint_aliases_preserve_original_tensors(self):
        weight = torch.tensor([1.25, -3.0], dtype=torch.float32)
        for prefix in ('vlm_backbone.', 'n17_backbone.'):
            old, new = prefix + 'model.layer.weight', prefix + 'vlm.layer.weight'
            state = {old: weight, 'vla_head.model.weight': weight}
            mapped, count = _native_checkpoint_keys(state)
            self.assertEqual(count, 1)
            self.assertIs(mapped[new], weight)
            self.assertIs(mapped['vla_head.model.weight'], weight)
            self.assertIn(old, state)
            self.assertNotIn(new, state)
            unchanged, count = _native_checkpoint_keys(mapped)
            self.assertEqual(count, 0)
            self.assertIs(unchanged[new], weight)
            with self.assertRaisesRegex(ConfigurationError, 'Conflicting checkpoint'):
                _native_checkpoint_keys({old: weight, new: weight.clone()})

    def test_hook_noise_and_instance_restore(self):

        class VLM(torch.nn.Module):

            def forward(self, *, position_ids=None):
                self.positions = position_ids

        class Head(torch.nn.Module):

            def _sample_initial_actions(self):
                raise AssertionError('Native sampler should be temporarily overridden')

        class Model(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.vlm_backbone = torch.nn.Module()
                self.vlm_backbone.vlm = VLM()
                self.vla_head = Head()
                self.fail = False

            def predict_action(self, **inputs):
                self.last = inputs
                self.vlm_backbone.vlm()
                result = self.vla_head._sample_initial_actions().add_(1)
                if self.fail:
                    raise RuntimeError('predict failure')
                return result

        model = Model()
        wrapper = _make_wrapper(model, torch)
        inputs = {k: torch.from_numpy(v) for k, v in samples().items()}
        for local in (False, True):

            def sentinel():
                return None

            if local:
                model.vla_head._sample_initial_actions = sentinel
            for failure in (False, True):
                model.fail = failure
                if failure:
                    with self.assertRaises(RuntimeError):
                        wrapper(*(inputs[k] for k in _NAMES))
                else:
                    output = wrapper(*(inputs[k] for k in _NAMES))
                    self.assertTrue(torch.equal(output, inputs['noise'] + 1))
                self.assertEqual(len(model.vlm_backbone.vlm._forward_pre_hooks), 0)
                self.assertEqual('_sample_initial_actions' in model.vla_head.__dict__,
                                 local)
                if local:
                    self.assertIs(model.vla_head._sample_initial_actions, sentinel)
                self.assertEqual(model.last['states'].dtype, torch.bfloat16)
                self.assertIs(model.vlm_backbone.vlm.positions, inputs['position_ids'])
                self.assertTrue(
                    torch.equal(inputs['noise'], torch.zeros_like(inputs['noise'])))
        self.assertEqual(wrapper.image_grid_thw.tolist(), [[[1, 16, 16], [1, 16, 16]]])


if __name__ == '__main__':
    unittest.main()
