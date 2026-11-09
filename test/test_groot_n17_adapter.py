"""Interface tests only. Tiny doubles are NOT full-model parity evidence."""
import copy
from contextlib import nullcontext
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fluxhermes.adapters.groot_n17 import (
    _NAMES, _check_inputs, _check_specs, _make_wrapper,
    _reference_runner, _set_attention, prepare_groot_n17,
)
from fluxhermes.contracts import DeploymentConfig
from fluxhermes.errors import ConfigurationError

try:
    import torch
except ImportError:
    torch = None


def configuration():
    path = Path(__file__).resolve().parents[1] / 'configs/gr00t/gr00t_n17_libero_bf16.py'
    return copy.deepcopy(runpy.run_path(str(path))['deployment'])


def samples(length=148):
    return dict(images=np.zeros((1, 512, 1536), np.float32),
                lang_tokens=np.zeros((1, length), np.int64),
                position_ids=np.zeros((3, 1, length), np.int64),
                states=np.zeros((1, 1, 132), np.float32),
                embodiment_ids=np.zeros((1,), np.int32),
                noise=np.zeros((1, 40, 132), np.float32))


class TinyAttention:
    def __init__(self):
        self.config = SimpleNamespace(_attn_implementation='sdpa')


class TinyVLM:
    def __init__(self):
        self.layers = [TinyAttention(), TinyAttention()]
        self.modes = []
        self.ignore_second = False

    def named_modules(self):
        return [(str(i), item) for i, item in enumerate(self.layers)]

    def set_attn_implementation(self, mode):
        self.modes.append(mode)
        for index, layer in enumerate(self.layers):
            if index == 1 and self.ignore_second and mode != 'sdpa':
                continue
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
            case = samples(); case[field] = value; cases.append(case)
        case = samples(); case['lang_masks'] = np.ones((1, 148)); cases.append(case)
        case = samples(); del case['noise']; cases.append(case)
        for case in cases:
            with self.assertRaises(ConfigurationError):
                _check_inputs(case, self.specs)

    def test_profile_and_dtype_rejected(self):
        for field, value in [('dtype', 'int32'), ('max_shape', (1, 162))]:
            config = configuration(); config['inputs'][1][field] = value
            with self.assertRaises(ConfigurationError):
                _check_specs(DeploymentConfig.from_dict(config))

    def test_closed_factory_kwargs(self):
        with self.assertRaises(TypeError):
            prepare_groot_n17(deployment=None, device='cpu', fluxvla_root='.',
                             config='x', checkpoint='y', sample_inputs='z', ignored=True)

    def test_config_export_flags(self):
        config = configuration()
        self.assertTrue(config['validation']['fail_on_error'])
        self.assertIn('samples_manifest', config['validation'])
        self.assertFalse(config['backend_config']['allow_tf32'])
        self.assertTrue(config['ir_config']['normalize_full_slices'])
        self.assertFalse(config['ir_config']['dynamo'])

    def test_partial_attention_setter_detected(self):
        vlm = TinyVLM(); vlm.ignore_second = True
        with self.assertRaisesRegex(RuntimeError, 'Attention 1'):
            _set_attention(vlm, 'flash_attention_2')

    def test_reference_modes_and_finally(self):
        for failure in ('none', 'forward', 'setter'):
            vlm = TinyVLM()
            vlm.ignore_second = failure == 'setter'

            class TinyWrapper:
                model = SimpleNamespace(vlm_backbone=SimpleNamespace(vlm=vlm))

                def __call__(self, *values):
                    self_test.assertEqual(vlm.layers[0].config._attn_implementation,
                                          'flash_attention_2')
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
            self.assertEqual(vlm.modes, ['flash_attention_2', 'sdpa'])
            self.assertTrue(all(layer.config._attn_implementation == 'sdpa'
                                for layer in vlm.layers))


@unittest.skipIf(torch is None, 'Torch unavailable; real tensor boundary tests require Torch')
class TensorBoundaryTests(unittest.TestCase):
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

        model = Model(); wrapper = _make_wrapper(model, torch)
        inputs = {k: torch.from_numpy(v) for k, v in samples().items()}
        for local in (False, True):
            sentinel = lambda: None
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
                self.assertEqual('_sample_initial_actions' in model.vla_head.__dict__, local)
                if local:
                    self.assertIs(model.vla_head._sample_initial_actions, sentinel)
                self.assertEqual(model.last['states'].dtype, torch.bfloat16)
                self.assertIs(model.vlm_backbone.vlm.positions, inputs['position_ids'])
                self.assertTrue(torch.equal(inputs['noise'], torch.zeros_like(inputs['noise'])))
        self.assertEqual(wrapper.image_grid_thw.tolist(), [[[1, 16, 16], [1, 16, 16]]])


if __name__ == '__main__':
    unittest.main()
