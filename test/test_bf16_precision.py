"""BF16 config and genuine ONNX preservation tests; no TensorRT substitutes.

The ONNX tests require the real onnx package and fail on missing dependencies.
TensorRT network construction and TF32 flag readback are target-GPU checks.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from fluxhermes.builders.tensorrt import _resolve_strongly_typed, build_tensorrt_engine
from fluxhermes.contracts import DeploymentConfig
from fluxhermes.errors import ConfigurationError
from fluxhermes.optimization.onnx import lower_onnx_precision


class BF16ConfigTest(unittest.TestCase):

    @staticmethod
    def config(**options):
        return DeploymentConfig.from_dict({
            'name':
            'bf16-test',
            'work_dir':
            'work',
            'source': {
                'type': 'python',
                'factory': 'example:factory'
            },
            'inputs': [{
                'name': 'x',
                'dtype': 'float32',
                'shape': (2, )
            }],
            'outputs': [{
                'name': 'y',
                'dtype': 'float32',
                'shape': (2, )
            }],
            # These tests inspect dtype labels, not an implicit compile chain.
            'pipeline': [{
                'type': 'PrepareModel'
            }],
            'backend_config': {
                'type': 'tensorrt',
                **options
            },
        })

    def test_bf16_label_roundtrips(self):
        for strong in ('auto', True):
            with self.subTest(strong=strong):
                config = self.config(
                    precision='bf16', strongly_typed=strong, allow_tf32=False)
                self.assertEqual(config.precision, 'bf16')
                restored = DeploymentConfig.from_dict(config.to_dict())
                self.assertEqual(restored.backend, config.backend)

    def test_bf16_default_typing_is_resolved_by_builder(self):
        config = self.config(precision='bf16')
        self.assertNotIn('strongly_typed', config.backend)
        self.assertTrue(
            _resolve_strongly_typed(10, config.precision,
                                    config.backend.get('strongly_typed', 'auto')))

    def test_bf16_rejects_explicit_weak_typing(self):
        with self.assertRaisesRegex(ConfigurationError, 'strongly_typed=True'):
            self.config(precision='bf16', strongly_typed=False)

    def test_existing_defaults_are_unchanged(self):
        config = self.config()
        self.assertEqual(config.precision, 'fp16')
        self.assertNotIn('allow_tf32', config.backend)
        self.assertNotIn('strongly_typed', config.backend)
        for precision in ('fp32', 'fp16', 'int8'):
            with self.subTest(precision=precision):
                self.assertEqual(self.config(precision=precision).precision, precision)

    def test_tf32_accepts_only_explicit_booleans(self):
        for value in (False, True):
            self.assertIs(self.config(allow_tf32=value).backend['allow_tf32'], value)
        for value in (None, 0, 1, 0.0, 'false', 'true', [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ConfigurationError, 'allow_tf32'):
                    self.config(allow_tf32=value)

    def test_builder_rejects_invalid_tf32_before_dependency_loading(self):
        for value in (None, 0, 1, 'false'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ConfigurationError, 'allow_tf32'):
                    build_tensorrt_engine('unused.onnx', 'unused.plan', (),
                                          {'allow_tf32': value})


class BF16TypingTest(unittest.TestCase):

    def test_bf16_auto_always_requests_strong_typing(self):
        for major in (8, 10, 11):
            with self.subTest(major=major):
                self.assertTrue(_resolve_strongly_typed(major, 'bf16', 'auto'))
                self.assertTrue(_resolve_strongly_typed(major, 'bf16', True))
                with self.assertRaisesRegex(ConfigurationError, 'strongly_typed'):
                    _resolve_strongly_typed(major, 'bf16', False)

    def test_other_modes_keep_version_dependent_auto(self):
        for precision in ('fp32', 'fp16', 'int8'):
            with self.subTest(precision=precision):
                self.assertFalse(_resolve_strongly_typed(10, precision, 'auto'))
                self.assertTrue(_resolve_strongly_typed(11, precision, 'auto'))
                self.assertTrue(_resolve_strongly_typed(10, precision, True))
                self.assertFalse(_resolve_strongly_typed(11, precision, False))

    def test_invalid_strong_options_are_rejected(self):
        for value in (None, 0, 1, 'true', 'false', [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ConfigurationError, 'strongly_typed'):
                    _resolve_strongly_typed(10, 'bf16', value)


class BF16OnnxPreservationTest(unittest.TestCase):

    @staticmethod
    def make_graph(path, *, external):
        import onnx
        from onnx import TensorProto, helper

        # Genuine mixed BF16/FLOAT graph, including stored BF16 initializer bits.
        weights = [
            helper.make_tensor(
                'bf16_bias',
                TensorProto.BFLOAT16, [2],
                struct.pack('<HH', 0x3FC0, 0xC000),
                raw=True),
            helper.make_tensor(
                'fp32_bias',
                TensorProto.FLOAT, [2],
                struct.pack('<ff', 0.25, 0.5),
                raw=True),
        ]
        nodes = [
            helper.make_node('Cast', ['x'], ['xb'], to=TensorProto.BFLOAT16),
            helper.make_node('Add', ['xb', 'bf16_bias'], ['yb']),
            helper.make_node('Cast', ['yb'], ['yf'], to=TensorProto.FLOAT),
            helper.make_node('Add', ['yf', 'fp32_bias'], ['y']),
        ]
        graph = helper.make_graph(
            nodes,
            'mixed-dtype', [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2])],
            initializer=weights)
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid('', 22)], ir_version=10)
        if external:
            onnx.save_model(
                model,
                str(path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location='weights.bin',
                size_threshold=0)
        else:
            onnx.save_model(model, str(path))
        onnx.checker.check_model(str(path), full_check=True)

    def test_bf16_preserves_graph_and_external_weights_byte_for_byte(self):
        import onnx

        for external in (False, True):
            for conversion in (False, True):
                with self.subTest(external=external, fp16_conversion=conversion):
                    with tempfile.TemporaryDirectory(prefix='hermes-bf16-') as tmp:
                        root = Path(tmp)
                        source = root / 'model.onnx'
                        self.make_graph(source, external=external)
                        before = {p.name: p.read_bytes() for p in root.iterdir()}
                        if external:
                            self.assertIn('weights.bin', before)
                        output = root / 'lowered' / 'model.onnx'
                        actual = lower_onnx_precision(source, output, 'bf16',
                                                      {'fp16_conversion': conversion})
                        self.assertEqual(actual, source.resolve())
                        self.assertFalse(output.parent.exists())
                        self.assertEqual(
                            {p.name: p.read_bytes()
                             for p in root.iterdir()}, before)
                        model = onnx.load(str(actual), load_external_data=True)
                        onnx.checker.check_model(model, full_check=True)
                        self.assertEqual(
                            [weight.data_type for weight in model.graph.initializer],
                            [onnx.TensorProto.BFLOAT16, onnx.TensorProto.FLOAT])

    def test_existing_no_conversion_paths_and_invalid_precision(self):
        with tempfile.TemporaryDirectory(prefix='hermes-bf16-') as tmp:
            root = Path(tmp)
            source = root / 'model.onnx'
            self.make_graph(source, external=False)
            output = root / 'lowered.onnx'
            for precision in ('fp32', 'int8', 'fp16'):
                self.assertEqual(
                    lower_onnx_precision(source, output, precision,
                                         {'fp16_conversion': False}), source.resolve())
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                lower_onnx_precision(source, output, 'fp8', {})
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
