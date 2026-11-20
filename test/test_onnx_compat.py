"""Actual Torch 2.6/2.8 CPU exports/ORT checks and registry ownership tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn import functional as F
from torch.onnx import symbolic_opset11, symbolic_opset14
from torch.onnx._internal import registration

from fluxhermes.adapters.base import PreparedModel
from fluxhermes.errors import ArtifactError, ConfigurationError
from fluxhermes.exporters.compat import legacy_bf16_symbolics
from fluxhermes.exporters.onnx import _load_convert, export_onnx


class Attention(torch.nn.Module):

    def __init__(self, scale, amp=False):
        super().__init__()
        self.scale, self.amp = scale, amp

    def forward(self, q, k, v):
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=self.amp):
            return F.scaled_dot_product_attention(q, k, v, scale=self.scale)


class GroupedAttention(torch.nn.Module):

    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v, enable_gqa=True, is_causal=self.causal, scale=0.3)


class BooleanAssignment(torch.nn.Module):

    def forward(self, x, mask, values):
        x = x.clone()
        x[mask, :] = x[mask, :] + values
        return x


class PartialSlice(torch.nn.Module):

    def forward(self, x):
        return x[:, 1::2, :]


class Cumulative(torch.nn.Module):

    def __init__(self, dtype=None):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.cumsum(-1, dtype=self.dtype)


class Failing(torch.nn.Module):

    def forward(self, x):
        raise RuntimeError('deliberate forward failure')


class IgnoresInput(torch.nn.Module):

    def forward(self, x, unused):
        return x + 1


class ONNXCompatTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)
        torch.random.default_generator.manual_seed(7)
        self.count = 0

    def prepared(self, model, inputs):
        return PreparedModel(model.eval(), inputs, ('out', ))

    def export(self, prepared, **kwargs):
        self.count += 1
        options = dict(opset=18, legacy_bf16_compat=True, **kwargs)
        return export_onnx(prepared, self.root / f'model-{self.count}.onnx', options)

    def runtime(self, path, inputs):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=['CPUExecutionProvider'])
        return session.run(None, {k: v.numpy() for k, v in inputs.items()})[0]

    def assert_restored(self):
        for name, original in (
            ('aten::cumsum', symbolic_opset11.cumsum),
            ('aten::scaled_dot_product_attention',
             symbolic_opset14.scaled_dot_product_attention),
        ):
            group = registration.registry.get_function_group(name)
            self.assertIs(group.get(18), original)
            self.assertFalse(group._functions.overridden(18))

    def test_positive_scale_and_bf16_matrix(self):
        for dtype, amp in ((torch.float32, False), (torch.bfloat16, False),
                           (torch.float32, True)):
            inputs = dict(
                q=torch.randn(2, 2, 3, 8).to(dtype),
                k=torch.randn(2, 2, 5, 8).to(dtype),
                v=torch.randn(2, 2, 5, 8).to(dtype))
            for scale in (None, 0.3, 1.0):
                for fold in (False, True):
                    with self.subTest(dtype=dtype, amp=amp, scale=scale, fold=fold):
                        prepared = self.prepared(Attention(scale, amp), inputs)
                        path = self.export(prepared, do_constant_folding=fold)
                        onnx.checker.check_model(str(path), full_check=True)
                        if dtype == torch.float32 and not amp:
                            np.testing.assert_allclose(
                                self.runtime(path, inputs),
                                prepared.run_reference()['out'].numpy(),
                                rtol=1e-5,
                                atol=1e-6)
                        self.assert_restored()

    def test_grouped_query_attention_lowering(self):
        for dtype in (torch.float32, torch.bfloat16):
            for kv_heads in (1, 2, 4):
                for causal in (False, True):
                    with self.subTest(dtype=dtype, kv_heads=kv_heads, causal=causal):
                        inputs = dict(
                            q=torch.randn(2, 4, 3, 8).to(dtype),
                            k=torch.randn(2, kv_heads, 5, 8).to(dtype),
                            v=torch.randn(2, kv_heads, 5, 8).to(dtype))
                        prepared = self.prepared(GroupedAttention(causal), inputs)
                        prepared.dynamic_axes = dict(
                            q={
                                0: 'batch',
                                2: 'qlen'
                            },
                            k={
                                0: 'batch',
                                2: 'kvlen'
                            },
                            v={
                                0: 'batch',
                                2: 'kvlen'
                            },
                            out={
                                0: 'batch',
                                2: 'qlen'
                            })
                        path = self.export(prepared, do_constant_folding=False)
                        if dtype == torch.float32:
                            for batch, qlen, kvlen in ((2, 3, 5), (1, 4, 6)):
                                sample = dict(
                                    q=torch.randn(batch, 4, qlen, 8),
                                    k=torch.randn(batch, kv_heads, kvlen, 8),
                                    v=torch.randn(batch, kv_heads, kvlen, 8))
                                expected = prepared.module(*sample.values()).numpy()
                                np.testing.assert_allclose(
                                    self.runtime(path, sample),
                                    expected,
                                    rtol=1e-5,
                                    atol=1e-6)
                        self.assert_restored()

    def test_boolean_assignment_full_slice_normalization(self):
        from torch.onnx import utils
        original = utils._optimize_graph
        for shape, count in (((1, 142, 8), 128), ((2, 7, 4), 3), ((1, 142, 8), 142)):
            mask = torch.zeros(shape[:2], dtype=torch.bool)
            mask.flatten()[:count] = True
            inputs = dict(
                x=torch.randn(shape), mask=mask, values=torch.randn(count, shape[-1]))
            prepared = self.prepared(BooleanAssignment(), inputs)
            path = self.export(
                prepared, normalize_full_slices=True, do_constant_folding=False)
            np.testing.assert_array_equal(
                self.runtime(path, inputs),
                prepared.run_reference()['out'].numpy())
            self.assertGreater(
                prepared.metadata['jit_view_normalization']['identity_slices_removed'],
                0)
            self.assertIs(utils._optimize_graph, original)
            self.assert_restored()

    def test_view_normalization_preserves_partial_slices_and_restores_on_error(self):
        from torch.onnx import utils
        original = utils._optimize_graph
        inputs = dict(x=torch.randn(2, 7, 4))
        prepared = self.prepared(PartialSlice(), inputs)
        path = self.export(prepared, normalize_full_slices=True)
        np.testing.assert_array_equal(
            self.runtime(path, inputs),
            prepared.run_reference()['out'].numpy())
        with self.assertRaisesRegex(RuntimeError, 'deliberate forward failure'):
            self.export(self.prepared(Failing(), inputs), normalize_full_slices=True)
        self.assertIs(utils._optimize_graph, original)
        self.assert_restored()

    def test_bool_cumsum_explicit_dtype_and_numeric_runtime(self):
        for dtype in (None, torch.int32, torch.int64, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                inputs = {'x': torch.tensor([[True, False, True], [True, True, False]])}
                prepared = self.prepared(Cumulative(dtype), inputs)
                path = self.export(prepared)
                actual = self.runtime(path, inputs)
                expected = prepared.run_reference()['out'].numpy()
                self.assertEqual(actual.dtype, expected.dtype)
                np.testing.assert_array_equal(actual, expected)

    def test_nonbool_cumsum_delegate_unchanged(self):
        for dtype in (torch.float32, torch.int64):
            inputs = {'x': torch.tensor([[1, 2, 3]], dtype=dtype)}
            prepared = self.prepared(Cumulative(), inputs)
            np.testing.assert_array_equal(
                self.runtime(self.export(prepared), inputs),
                prepared.run_reference()['out'].numpy())

    def test_original_failure_without_opt_in_and_cleanup_after_failure(self):
        inputs = dict(
            q=torch.randn(1, 1, 3, 8).bfloat16(),
            k=torch.randn(1, 1, 3, 8).bfloat16(),
            v=torch.randn(1, 1, 3, 8).bfloat16())
        with self.assertRaisesRegex(RuntimeError, 'ComplexDouble'):
            export_onnx(
                self.prepared(Attention(0.3), inputs), self.root / 'raw.onnx', {})
        with self.assertRaisesRegex(RuntimeError, 'deliberate forward failure'):
            self.export(self.prepared(Failing(), {'x': torch.ones(1)}))
        self.assert_restored()

    def test_preexisting_custom_registration_is_retained(self):
        for opset in (11, 18):

            def custom(*args):
                raise RuntimeError('custom should not execute')

            torch.onnx.register_custom_op_symbolic('aten::cumsum', custom, opset)
            try:
                with self.assertRaisesRegex(ConfigurationError, 'registry conflict'):
                    with legacy_bf16_symbolics(torch, 18):
                        self.fail('conflicting registry entered')
                self.assertIs(
                    registration.registry.get_function_group('aten::cumsum').get(18),
                    custom)
            finally:
                torch.onnx.unregister_custom_op_symbolic('aten::cumsum', opset)
        self.assert_restored()

    def test_same_function_custom_registration_is_still_a_conflict(self):
        torch.onnx.register_custom_op_symbolic('aten::cumsum', symbolic_opset11.cumsum,
                                               18)
        try:
            with self.assertRaisesRegex(ConfigurationError, 'registry conflict'):
                with legacy_bf16_symbolics(torch, 18):
                    self.fail('same-function override entered')
            self.assertTrue(
                registration.registry.get_function_group(
                    'aten::cumsum')._functions.overridden(18))
        finally:
            torch.onnx.unregister_custom_op_symbolic('aten::cumsum', 18)

    def test_external_registry_change_is_not_deleted(self):

        def external(*args):
            raise RuntimeError('external should not execute')

        try:
            with self.assertRaisesRegex(ConfigurationError, 'changed during export'):
                with legacy_bf16_symbolics(torch, 18):
                    torch.onnx.register_custom_op_symbolic('aten::cumsum', external, 18)
            self.assertIs(
                registration.registry.get_function_group('aten::cumsum').get(18),
                external)
        finally:
            torch.onnx.unregister_custom_op_symbolic('aten::cumsum', 18)
        self.assert_restored()

    def test_explicit_options_and_strict_inputs(self):
        prepared = self.prepared(Cumulative(), {'x': torch.ones(2)})
        for options in ({
                'legacy_bf16_compat': 'true'
        }, {
                'legacy_bf16_compat': True,
                'dynamo': True
        }, {
                'legacy_bf16_compat': True,
                'opset': 19
        }, {
                'target_opset': '22'
        }, {
                'opset': 18,
                'target_opset': 17
        }):
            with self.subTest(options=options), self.assertRaises(ConfigurationError):
                export_onnx(prepared, self.root / 'invalid.onnx', options)
        prepared = self.prepared(IgnoresInput(), {
            'x': torch.ones(2),
            'unused': torch.ones(2)
        })
        with self.assertRaisesRegex(ArtifactError, 'Unexpected ONNX input names'):
            self.export(prepared)
        self.assertEqual(prepared.metadata['onnx_io']['pruned_inputs'], ['unused'])
        self.assert_restored()

    def test_default_fp32_path(self):
        inputs = {'x': torch.tensor([[1., 2., 3.]])}
        prepared = self.prepared(Cumulative(), inputs)
        path = export_onnx(prepared, self.root / 'default.onnx', {})
        np.testing.assert_array_equal(self.runtime(path, inputs), [[1., 3., 6.]])

    def test_bf16_conv_official_conversion(self):
        prepared = self.prepared(
            torch.nn.Conv2d(3, 4, 3).bfloat16(),
            {'x': torch.randn(2, 3, 8, 8).bfloat16()})
        path = self.export(prepared, target_opset=22, do_constant_folding=False)
        self.assertEqual(onnx.load(str(path)).opset_import[0].version, 22)
        onnx.checker.check_model(str(path), full_check=True)

    def test_external_data_survives_official_conversion(self):
        path = self.root / 'external.onnx'
        prepared = self.prepared(torch.nn.Linear(4, 3), {'x': torch.randn(2, 4)})
        export_onnx(prepared, path, {})
        model = onnx.load(str(path))
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location='weights.bin',
            size_threshold=0)
        weights = (self.root / 'weights.bin').read_bytes()
        converted = _load_convert(onnx, path, 22)
        self.assertTrue(
            all(not value.HasField('raw_data')
                for value in converted.graph.initializer))
        self.assertTrue(
            all(value.data_location == onnx.TensorProto.EXTERNAL
                for value in converted.graph.initializer))
        self.assertEqual((self.root / 'weights.bin').read_bytes(), weights)
        onnx.checker.check_model(str(path), full_check=True)
        np.testing.assert_allclose(
            self.runtime(path, prepared.inputs),
            prepared.run_reference()['out'].detach().numpy(),
            rtol=1e-5,
            atol=1e-6)

    def test_external_constant_attribute_survives_converter(self):
        payload = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = onnx.numpy_helper.from_array(payload, name='/Constant_attr::value')
        node = onnx.helper.make_node(
            'Constant', [], ['y'], value=tensor, name='/Constant')
        graph = onnx.helper.make_graph(
            [node], 'external-constant', [],
            [onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 3])])
        model = onnx.helper.make_model(
            graph, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 18)])
        path = self.root / 'constant.onnx'
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location='constant.bin',
            size_threshold=0,
            convert_attribute=True)
        before = onnx.load(str(path), load_external_data=False)
        original = before.graph.node[0].attribute[0].t
        self.assertEqual(original.data_location, onnx.TensorProto.EXTERNAL)
        weights = (self.root / 'constant.bin').read_bytes()
        # Reproduce the actual official-converter loss, independently of repair.
        broken = onnx.version_converter.convert_version(before, 22)
        broken_path = self.root / 'broken.onnx'
        onnx.save_model(broken, str(broken_path))
        with self.assertRaisesRegex(onnx.checker.ValidationError, 'one and only one'):
            onnx.checker.check_model(str(broken_path), full_check=True)
        converted = _load_convert(onnx, path, 22)
        self.assertEqual(converted.graph.node[0].attribute[0].t, original)
        self.assertEqual((self.root / 'constant.bin').read_bytes(), weights)
        onnx.checker.check_model(str(path), full_check=True)
        np.testing.assert_array_equal(self.runtime(path, {}), payload)

    def test_external_constant_in_nested_if_graph(self):
        branches = []
        for name, sign in (('then', 1), ('else', -1)):
            tensor = onnx.numpy_helper.from_array(
                np.array([sign, 2 * sign], dtype=np.float32), name=name + '/value')
            branches.append(
                onnx.helper.make_graph([
                    onnx.helper.make_node(
                        'Constant', [], [name + '_out'], value=tensor)
                ], name, [], [
                    onnx.helper.make_tensor_value_info(name + '_out',
                                                       onnx.TensorProto.FLOAT, [2])
                ]))
        graph = onnx.helper.make_graph([
            onnx.helper.make_node(
                'If', ['cond'], ['y'], then_branch=branches[0], else_branch=branches[1])
        ], 'nested', [
            onnx.helper.make_tensor_value_info('cond', onnx.TensorProto.BOOL, [])
        ], [onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2])])
        model = onnx.helper.make_model(
            graph, ir_version=10, opset_imports=[onnx.helper.make_opsetid('', 18)])
        path = self.root / 'nested.onnx'
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location='nested.bin',
            size_threshold=0,
            convert_attribute=True)
        weights = (self.root / 'nested.bin').read_bytes()
        _load_convert(onnx, path, 22)
        self.assertEqual((self.root / 'nested.bin').read_bytes(), weights)
        onnx.checker.check_model(str(path), full_check=True)
        for cond in (True, False):
            np.testing.assert_array_equal(
                self.runtime(path, {'cond': torch.tensor(cond)}),
                np.array([1, 2], dtype=np.float32) * (1 if cond else -1))

    def test_tensor_traversal_includes_sparse_and_repeated_attributes(self):
        from fluxhermes.exporters.onnx import (_restore_external_references,
                                               _walk_tensor_protos)
        model = onnx.ModelProto()
        sparse = model.graph.sparse_initializer.add()
        sparse.values.name, sparse.indices.name = 'sparse-values', 'sparse-indices'
        attr = model.graph.node.add().attribute.add()
        attr.type = onnx.AttributeProto.SPARSE_TENSOR
        attr.sparse_tensor.values.name = 'attr-values'
        attr.sparse_tensor.indices.name = 'attr-indices'
        repeated = model.graph.node[0].attribute.add()
        repeated.type = onnx.AttributeProto.TENSORS
        repeated.tensors.add().name = 'repeated-0'
        repeated.tensors.add().name = 'repeated-1'
        names = {value.name for value in _walk_tensor_protos(model)}
        self.assertEqual(
            names, {
                'sparse-values', 'sparse-indices', 'attr-values', 'attr-indices',
                'repeated-0', 'repeated-1'
            })
        # Metadata-only tests use genuine TensorProto messages, without opening
        # any tensor storage. Numerical external-data tests are above.
        for value in _walk_tensor_protos(model):
            value.data_type = onnx.TensorProto.FLOAT
            value.dims.append(1)
            value.data_location = onnx.TensorProto.EXTERNAL
            entry = value.external_data.add()
            entry.key, entry.value = 'location', 'never-opened.bin'
        converted = onnx.ModelProto()
        converted.CopyFrom(model)
        for value in _walk_tensor_protos(converted):
            value.ClearField('external_data')
            value.ClearField('data_location')
        _restore_external_references(onnx, model, converted)
        self.assertEqual(converted, model)

    def test_reference_restore_rejects_missing_colliding_or_changed_tensor(self):
        from fluxhermes.exporters.onnx import _restore_external_references
        source = onnx.ModelProto()
        tensor = source.graph.initializer.add()
        tensor.name = 'external'
        tensor.dims.append(2)
        tensor.data_type = onnx.TensorProto.FLOAT
        tensor.data_location = onnx.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key, entry.value = 'location', 'never-opened.bin'
        for change, expected in (
            ('missing', '0 matches'),
            ('duplicate', '2 matches'),
            ('shape', 'changed external tensor'),
            ('dtype', 'changed external tensor'),
            ('payload', 'changed external tensor'),
        ):
            with self.subTest(change=change):
                converted = onnx.ModelProto()
                converted.CopyFrom(source)
                current = converted.graph.initializer[0]
                current.ClearField('external_data')
                current.ClearField('data_location')
                if change == 'missing':
                    converted.graph.ClearField('initializer')
                elif change == 'duplicate':
                    converted.graph.initializer.add().CopyFrom(current)
                elif change == 'shape':
                    current.dims[0] = 3
                elif change == 'dtype':
                    current.data_type = onnx.TensorProto.DOUBLE
                else:
                    current.raw_data = b'changed'
                before = converted.SerializeToString()
                with self.assertRaisesRegex(ArtifactError, expected):
                    _restore_external_references(onnx, source, converted)
                self.assertEqual(converted.SerializeToString(), before)
        for change in ('empty-name', 'duplicate-source'):
            original = onnx.ModelProto()
            original.CopyFrom(source)
            if change == 'empty-name':
                original.graph.initializer[0].name = ''
            else:
                original.graph.initializer.add().CopyFrom(tensor)
            with self.assertRaisesRegex(ArtifactError, 'empty or ambiguous'):
                _restore_external_references(onnx, original, source)


if __name__ == '__main__':
    unittest.main()
