"""CPU tensor coverage; no TensorRT, GPU, or mocked numerical outputs."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fluxhermes.adapters.base import PreparedModel
from fluxhermes.backend.base import BaseBackendWrapper
from fluxhermes.errors import ValidationError
from fluxhermes.validation.parity import validate_backend


class TinyPolicy(torch.nn.Module):
    def forward(self, observation, noise):
        return torch.sin(observation) + noise.square()


class TorchBackend(BaseBackendWrapper):
    def __init__(self, offset=0.0, mutate=False):
        self.calls = []
        self.offset = offset
        self.mutate = mutate

    def infer(self, inputs, synchronize=False):
        self.calls.append({name: value.clone() for name, value in inputs.items()})
        if not synchronize:
            raise AssertionError('Validation must synchronize.')
        result = TinyPolicy()(inputs['observation'], inputs['noise']) + self.offset
        if self.mutate:
            inputs['noise'].zero_()
        return {'actions': result}


class ValidationMatrixTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = PreparedModel(
            TinyPolicy(),
            {'observation': torch.ones(1, 2), 'noise': torch.ones(1, 2)},
            ('actions',),
        )
        self.options = {'atol': .02, 'rtol': .02, 'min_cosine_similarity': .999}

    def manifest(self, count=15):
        samples = []
        for index in range(count):
            name = f'sample-{index}.npz'
            np.savez(self.root / name,
                     observation=np.full((1, 2), index // 3 + 1, dtype=np.float64),
                     noise=np.full((1, 2), index % 3 + 1, dtype=np.float64))
            samples.append({'id': f'obs-{index // 3}-noise-{index % 3}',
                            'inputs': name})
        return self.write_manifest({'samples': samples})

    def write_manifest(self, document):
        path = self.root / 'samples.json'
        path.write_text(json.dumps(document), encoding='utf-8')
        return str(path)

    def test_fifteen_actual_comparisons_independent_reference_and_conversion(self):
        reference_calls = []

        def native(inputs):
            reference_calls.append({k: v.clone() for k, v in inputs.items()})
            return TinyPolicy()(inputs['observation'], inputs['noise'])

        # The export graph deliberately differs; native must remain authoritative.
        self.prepared.module = lambda *values: torch.zeros_like(values[0])
        self.prepared.reference_runner = native
        backend = TorchBackend()
        report = validate_backend(self.prepared, backend,
                                  dict(self.options, samples_manifest=self.manifest()))
        self.assertTrue(report['passed'])
        self.assertEqual(report['sample_count'], 15)
        self.assertEqual(len(reference_calls), 15)
        self.assertEqual(len(backend.calls), 15)
        self.assertEqual(len({sample['id'] for sample in report['samples']}), 15)
        for sample, reference, actual in zip(
                report['samples'], reference_calls, backend.calls):
            self.assertTrue(sample['passed'])
            self.assertEqual(sample['atol'], .02)
            for name in actual:
                torch.testing.assert_close(reference[name], actual[name])
                self.assertEqual(actual[name].dtype, self.prepared.inputs[name].dtype)
                self.assertEqual(actual[name].device, self.prepared.inputs[name].device)
                self.assertEqual(sample['inputs'][name]['shape'], [1, 2])
                self.assertEqual(sample['inputs'][name]['source_dtype'], 'float64')
                self.assertEqual(len(sample['inputs'][name]['sha256']), 64)
        hashes = {s['inputs']['noise']['sha256'] for s in report['samples']}
        self.assertEqual(len(hashes), 3)

    def test_single_compatibility_and_output_normalization(self):
        report = validate_backend(self.prepared, TorchBackend(), {})
        self.assertEqual(set(report), {'passed', 'atol', 'rtol',
                                      'min_cosine_similarity', 'outputs'})
        self.assertEqual((report['atol'], report['rtol'],
                          report['min_cosine_similarity']),
                         (.01, .01, .995))
        expected = self.prepared.run_reference()['actions']
        for output in (expected, [expected], (expected,), {'actions': expected}):
            self.prepared.reference_runner = lambda inputs, output=output: output
            torch.testing.assert_close(
                self.prepared.run_reference()['actions'], expected)
        self.prepared.reference_runner = None
        # Explicit mappings use prepared positional order, not the supplied order.
        supplied = {'noise': torch.full((1, 2), 3.),
                    'observation': torch.full((1, 2), 2.)}
        torch.testing.assert_close(self.prepared.run_reference(supplied)['actions'],
                                   TinyPolicy()(supplied['observation'],
                                                supplied['noise']))

    def test_mutating_runners_preserve_original_and_each_other_inputs(self):
        def native(inputs):
            result = TinyPolicy()(inputs['observation'], inputs['noise'])
            inputs['noise'].add_(100)
            return result

        self.prepared.reference_runner = native
        original = {k: v.clone() for k, v in self.prepared.inputs.items()}
        manifest = self.manifest(2)
        original_bytes = (self.root / 'sample-0.npz').read_bytes()
        for options in ({}, {'samples_manifest': manifest}):
            backend = TorchBackend(mutate=True)
            self.assertTrue(validate_backend(self.prepared, backend, options)['passed'])
            self.assertTrue(torch.equal(backend.calls[0]['noise'], torch.ones(1, 2)))
        for name, value in original.items():
            torch.testing.assert_close(self.prepared.inputs[name], value)
        self.assertEqual((self.root / 'sample-0.npz').read_bytes(), original_bytes)

    def test_invalid_manifest_and_sample_fields(self):
        invalid = [
            {}, {'samples': []}, {'samples': {}, 'extra': 1},
            {'samples': [{'id': '', 'inputs': 'x.npz'}]},
            {'samples': [{'id': '   ', 'inputs': 'x.npz'}]},
            {'samples': [{'id': 'a'}]}, {'samples': [{'inputs': 'x.npz'}]},
            {'samples': [{'id': 'a', 'inputs': ''}]},
            {'samples': [{'id': 'a', 'inputs': 'x.npz', 'extra': 1}]},
            {'samples': [{'id': 'a', 'inputs': 'x.npz'}] * 2},
        ]
        for document in invalid:
            with self.subTest(document=document), self.assertRaises(ValidationError):
                validate_backend(self.prepared, TorchBackend(),
                                 {'samples_manifest': self.write_manifest(document)})

    def test_missing_extra_npz_names_and_pickle_rejected(self):
        cases = [
            {'observation': np.ones((1, 2))},
            {'observation': np.ones((1, 2)), 'noise': np.ones((1, 2)),
             'extra': np.ones(1)},
            {'observation': np.ones((1, 2)),
             'noise': np.array([object()], dtype=object)},
        ]
        for arrays in cases:
            with self.subTest(names=list(arrays)):
                path = self.root / 'bad.npz'
                np.savez(path, **arrays)
                manifest = self.write_manifest(
                    {'samples': [{'id': 'bad', 'inputs': str(path)}]})
                backend = TorchBackend()
                with self.assertRaises((ValidationError, ValueError)):
                    validate_backend(self.prepared, backend,
                                     {'samples_manifest': manifest})
                self.assertEqual(backend.calls, [])

    def test_all_pass_aggregation_thresholds_and_both_failure_modes(self):
        manifest = self.manifest(3)
        for fail_on_error in (False, True):
            backend = TorchBackend(offset=.5)
            options = dict(self.options, samples_manifest=manifest,
                           fail_on_error=fail_on_error)
            if fail_on_error:
                with self.assertRaises(ValidationError) as caught:
                    validate_backend(self.prepared, backend, options)
                report = caught.exception.report
            else:
                report = validate_backend(self.prepared, backend, options)
            self.assertFalse(report['passed'])
            self.assertEqual(len(backend.calls), 3)
            self.assertEqual([s['passed'] for s in report['samples']],
                             [False, False, False])
            self.assertEqual(report['min_cosine_similarity'], .999)
        self.assertTrue(validate_backend(self.prepared, TorchBackend(offset=.001),
                                        dict(self.options,
                                             samples_manifest=manifest))['passed'])
        # One bad sample must fail the matrix even when subsequent samples pass.
        class FirstBad(TorchBackend):
            def infer(self, inputs, synchronize=False):
                self.offset = .5 if not self.calls else 0.
                return super().infer(inputs, synchronize)
        report = validate_backend(self.prepared, FirstBad(),
                                  dict(self.options, samples_manifest=manifest,
                                       fail_on_error=False))
        self.assertFalse(report['passed'])
        self.assertEqual([s['passed'] for s in report['samples']], [False, True, True])
        with self.assertRaises(ValidationError) as caught:
            validate_backend(self.prepared, TorchBackend(offset=1), {})
        self.assertFalse(caught.exception.report['passed'])
        self.assertFalse(validate_backend(self.prepared, TorchBackend(offset=1),
                                         {'fail_on_error': False})['passed'])

    def test_cosine_threshold_is_not_replaced_by_absolute_tolerance(self):
        class RotatedBackend(TorchBackend):
            def infer(self, inputs, synchronize=False):
                result = super().infer(inputs, synchronize)
                result['actions'][:, 0].neg_()
                return result

        options = dict(self.options, samples_manifest=self.manifest(2),
                       atol=100., rtol=100., fail_on_error=False)
        report = validate_backend(self.prepared, RotatedBackend(), options)
        self.assertFalse(report['passed'])
        self.assertTrue(all(not sample['passed'] for sample in report['samples']))
        self.assertEqual(report['min_cosine_similarity'], .999)
        options['min_cosine_similarity'] = -1.
        self.assertTrue(validate_backend(
            self.prepared, RotatedBackend(), options)['passed'])

    def test_actual_infer_exceptions_propagate_in_both_modes(self):
        class BrokenBackend(TorchBackend):
            def infer(self, inputs, synchronize=False):
                if self.calls:
                    raise RuntimeError('actual backend failure')
                return super().infer(inputs, synchronize)

        for fail_on_error in (False, True):
            with self.assertRaisesRegex(
                    RuntimeError, 'actual backend failure') as caught:
                validate_backend(self.prepared, BrokenBackend(),
                                 {'samples_manifest': self.manifest(3),
                                  'fail_on_error': fail_on_error})
            self.assertFalse(caught.exception.report['passed'])
            self.assertEqual(len(caught.exception.report['samples']), 2)
        def broken_native(inputs):
            raise RuntimeError('actual native failure')
        self.prepared.reference_runner = broken_native
        backend = TorchBackend()
        with self.assertRaisesRegex(RuntimeError, 'actual native failure'):
            validate_backend(self.prepared, backend,
                             {'samples_manifest': self.manifest(1),
                              'fail_on_error': False})
        self.assertEqual(backend.calls, [])


if __name__ == '__main__':
    unittest.main()
