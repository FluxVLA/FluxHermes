"""Real CPU tensor tests of adapter boundaries, not full PI05 validation.

The test adapter substitutes only the heavy import/builder entry with a tiny
Torch module. MMEngine JSON parsing, safetensors loading, load_state_dict,
autocast and in-place tensor operations all execute normally.
"""

import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch
from mmengine import Config
from safetensors.torch import save_file

from fluxhermes.adapters.base import PreparedModel
from fluxhermes.adapters.fluxvla import FluxVLAAdapter
from fluxhermes.contracts import DeploymentConfig
from fluxhermes.errors import ConfigurationError


class SmallPolicy(torch.nn.Module):
    """Exercise mixed constructor dtypes and a real in-place noise update."""

    def __init__(self, openpi_fp32_flow=False):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2, bias=False)
        self.norm = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        self.register_buffer('counter', torch.tensor(0, dtype=torch.int64))
        self.openpi_fp32_flow = openpi_fp32_flow

    def predict_action(self, states, noise=None):
        velocity = self.projection(states)
        return {'actions': velocity if noise is None else noise.add_(velocity)}


class SmallModelAdapter(FluxVLAAdapter):
    """Avoid the native CUDA import stack; keep the actual loading path."""

    def _import_stack(self):
        return (Path(self.source['fluxvla_root']), torch, Config,
                lambda model_cfg: SmallPolicy(**model_cfg))


class FluxVLAAdapterTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.document = {
            'model': {'openpi_fp32_flow': False},
            'runner': {'enable_mixed_precision_training': True,
                       'mixed_precision_dtype': 'bf16'},
        }
        (self.root / 'training.json').write_text(json.dumps(self.document))
        self.weights = {
            'norm': torch.tensor([1.001, 0.999], dtype=torch.float32),
            'projection.weight': torch.tensor([[0.713, -0.213], [0.101, 1.033]]),
            'counter': torch.tensor(7, dtype=torch.int64),
        }
        save_file(self.weights, self.root / 'model.safetensors')
        self.source = {
            'type': 'fluxvla', 'fluxvla_root': str(self.root),
            'config': 'training.json', 'checkpoint': 'model.safetensors',
        }
        self.deployment = DeploymentConfig.from_dict({
            'name': 'adapter-boundary-test', 'work_dir': str(self.root / 'work'),
            'source': self.source,
            'inputs': [{'name': 'states', 'dtype': 'float32', 'shape': (1, 2)}],
            'outputs': [{'name': 'actions', 'dtype': 'float32', 'shape': (1, 2)}],
        })
        self.states = torch.tensor([[0.321, -0.789]])

    def adapter(self, **source):
        return SmallModelAdapter(dict(self.source, **source), self.deployment)

    def test_explicit_parameter_dtype_precedes_checkpoint_copy(self):
        model, cfg, path, _ = self.adapter(parameter_dtype='float32')._load_model('cpu')
        self.assertEqual(model.norm.dtype, torch.float32)
        torch.testing.assert_close(model.norm, self.weights['norm'], rtol=0, atol=0)
        self.assertFalse(torch.equal(self.weights['norm'],
                                     self.weights['norm'].bfloat16().float()))
        self.assertEqual(model.counter.dtype, torch.int64)
        self.assertEqual(model.counter.item(), 7)
        self.assertFalse(model.training)
        self.assertFalse(model.openpi_fp32_flow)
        self.assertEqual(cfg.to_dict(), self.document)
        self.assertEqual(path, self.root / 'model.safetensors')

    def test_default_and_none_preserve_constructor_parameter_dtypes(self):
        for options in ({}, {'parameter_dtype': None}):
            with self.subTest(options=options):
                model, _, _, _ = self.adapter(**options)._load_model('cpu')
                self.assertEqual(model.norm.dtype, torch.bfloat16)
                self.assertEqual(model.projection.weight.dtype, torch.float32)
                torch.testing.assert_close(model.norm, self.weights['norm'].bfloat16(),
                                           rtol=0, atol=0)

    def test_explicit_low_precision_is_never_inferred_from_runner(self):
        for name in ('float16', 'bfloat16'):
            with self.subTest(dtype=name):
                model, _, _, _ = self.adapter(parameter_dtype=name)._load_model('cpu')
                self.assertTrue(all(p.dtype == getattr(torch, name)
                                    for p in model.parameters()))
                self.assertEqual(model.counter.dtype, torch.int64)

    def test_native_safetensors_still_loads_strictly(self):
        for state in ({'norm': self.weights['norm']},
                      dict(self.weights, norm=torch.ones(3)),
                      dict(self.weights, extra=torch.ones(1))):
            with self.subTest(keys=tuple(state)):
                save_file(state, self.root / 'model.safetensors')
                with self.assertRaises(RuntimeError):
                    self.adapter(parameter_dtype='float32')._load_model('cpu')

    def test_invalid_precision_fields_fail_explicitly(self):
        for field, values in (
                ('parameter_dtype', ('bf16', 'int64', False, [])),
                ('autocast_dtype', ('float32', 'bf16', True, {}))):
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ConfigurationError, field):
                        self.adapter(**{field: value})._load_model('cpu')

    def test_explicit_bfloat16_autocast_matches_direct_torch_call(self):
        adapter = self.adapter(parameter_dtype='float32', autocast_dtype='bfloat16')
        model, _, _, _ = adapter._load_model('cpu')
        wrapper = adapter._make_wrapper(model, torch, 'cpu')
        with torch.inference_mode():
            with torch.autocast('cpu', dtype=torch.bfloat16):
                expected = model.predict_action(self.states)['actions']
            actual = wrapper(self.states)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        self.assertFalse(torch.is_autocast_enabled('cpu'))

    def test_omitted_autocast_preserves_caller_context(self):
        adapter = self.adapter()
        model, _, _, _ = adapter._load_model('cpu')
        wrapper = adapter._make_wrapper(model, torch)
        with torch.inference_mode():
            self.assertEqual(wrapper(self.states).dtype, torch.float32)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = wrapper(self.states)
                expected = model.predict_action(self.states)['actions']
                self.assertEqual(actual.dtype, torch.bfloat16)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_explicit_none_disables_autocast_and_restores_outer_context(self):
        adapter = self.adapter(autocast_dtype=None)
        model, _, _, _ = adapter._load_model('cpu')
        wrapper = adapter._make_wrapper(model, torch)
        with torch.inference_mode():
            expected = model.predict_action(self.states)['actions']
            with torch.autocast('cpu', dtype=torch.bfloat16):
                actual = wrapper(self.states)
                self.assertTrue(torch.is_autocast_enabled('cpu'))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_exception_restores_autocast(self):
        adapter = self.adapter(autocast_dtype=None)
        model, _, _, _ = adapter._load_model('cpu')
        wrapper = adapter._make_wrapper(model, torch)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with self.assertRaises(RuntimeError):
                wrapper(torch.ones(1, 3))
            self.assertTrue(torch.is_autocast_enabled('cpu'))

    def test_noise_binding_is_cloned_and_repeated_reference_is_identical(self):
        self.deployment = DeploymentConfig.from_dict({
            'name': 'noise-test', 'work_dir': str(self.root / 'work'),
            'source': self.source,
            'inputs': [dict(name=name, dtype='float32', shape=(1, 2))
                       for name in ('states', 'epsilon')],
            'outputs': [dict(name='actions', dtype='float32', shape=(1, 2))],
        })
        adapter = self.adapter(bindings={'states': 'states', 'noise': 'epsilon'})
        model, _, _, _ = adapter._load_model('cpu')
        noise = torch.tensor([[0.13, -0.37]])
        original = noise.clone()
        rng = torch.random.get_rng_state().clone()
        version = noise._version
        prepared = PreparedModel(
            module=adapter._make_wrapper(model, torch),
            inputs=OrderedDict(states=self.states, epsilon=noise),
            output_names=('actions',))
        with torch.inference_mode():
            expected = model.predict_action(self.states, noise.clone())['actions']
            first = prepared.run_reference()['actions']
            second = prepared.run_reference()['actions']
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(noise, original))
        self.assertEqual(noise._version, version)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_prepare_records_only_explicit_precision_fields(self):
        import numpy as np
        np.savez(self.root / 'sample.npz', states=self.states.numpy())
        for options in ({}, {'parameter_dtype': 'float32', 'autocast_dtype': None}):
            with self.subTest(options=options):
                adapter = self.adapter(
                    sample_inputs=str(self.root / 'sample.npz'), **options)
                prepared = adapter.prepare('cpu')
                for field in ('parameter_dtype', 'autocast_dtype'):
                    self.assertEqual(field in prepared.metadata, field in options)
                    if field in options:
                        self.assertEqual(prepared.metadata[field], options[field])
                self.assertEqual(tuple(prepared.run_reference()), ('actions',))


if __name__ == '__main__':
    unittest.main()
