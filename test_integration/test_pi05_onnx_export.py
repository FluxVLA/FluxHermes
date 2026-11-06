"""Exercise the actual Hermes BF16 exporter with the verified SDK toolchain."""

import copy
import json
import unittest

import onnx
import test_export_package_adapter as fixtures
import torch

from fluxhermes.adapters.fluxvla_export import FluxVLAExportAdapter
from fluxhermes.backend.onnxruntime.wrapper import require_cpu_graph
from fluxhermes.errors import ConfigurationError
from fluxhermes.exporters import export_onnx
from fluxhermes.exporters.artifacts import stage_graph
from fluxhermes.multigraph import _export_model
from fluxhermes.pipeline import CompilePipeline


class PI05OnnxExportTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fixtures.ExportPackageAdapterTest.setUpClass.__func__(cls)

    def test_bfloat16_uses_real_dynamo_and_packages_exact_graph_dtypes(self):
        document = copy.deepcopy(self.document)
        document['precision']['autocast_dtype'] = 'bfloat16'
        document['precision']['flow_fp32'] = True
        document['precision']['residual_fp32'] = False
        document['precision']['backbone_fp32_layers'] = [0]
        path = self.root / 'bf16.json'
        path.write_text(json.dumps(document))
        config = fixtures.ExportPackageAdapterTest.deployment(self)
        source = dict(config.source, deployment_config=str(path))
        prepared = FluxVLAExportAdapter(source, config).prepare('cpu')
        types = {
            torch.bool: onnx.TensorProto.BOOL,
            torch.int64: onnx.TensorProto.INT64,
            torch.float32: onnx.TensorProto.FLOAT,
            torch.bfloat16: onnx.TensorProto.BFLOAT16
        }
        for name, graph in prepared.graphs.items():
            with self.subTest(graph=name):
                self.assertEqual(graph.export_options['opset'], 22)
                self.assertIs(graph.export_options['dynamo'], True)
                self.assertIs(graph.export_options['optimize'], True)
                output = export_onnx(
                    _export_model(graph), self.root / f'{name}.onnx',
                    graph.export_options)
                loaded = onnx.load(str(output)).graph
                initializers = {value.name for value in loaded.initializer}
                reductions = [
                    node for node in loaded.node if node.op_type.startswith('Reduce')
                ]
                self.assertTrue(reductions)
                for node in reductions:
                    self.assertIn(node.input[1], initializers, node.name)
                self.assertEqual(
                    tuple(value.name for value in loaded.input), tuple(graph.inputs))
                self.assertEqual(
                    tuple(value.type.tensor_type.elem_type for value in loaded.input),
                    tuple(types[value.dtype] for value in graph.inputs.values()))
                with torch.no_grad():
                    expected = graph.module(*graph.inputs.values())
                if isinstance(expected, torch.Tensor):
                    expected = (expected, )
                self.assertEqual(
                    tuple(value.type.tensor_type.elem_type for value in loaded.output),
                    tuple(types[value.dtype] for value in expected))
                staged, _ = stage_graph(output, self.root / f'staged-{name}')
                onnx.checker.check_model(str(staged[0]), full_check=True)
                with self.assertRaisesRegex(ConfigurationError, 'BF16'):
                    require_cpu_graph(staged[0])

    def test_source_export_profile_cannot_be_silently_overridden(self):
        config = fixtures.ExportPackageAdapterTest.deployment(
            self,
            backend_config={
                'type': 'onnxruntime',
                'device': 'cpu',
                'precision': 'fp32'
            },
            ir_config={
                'type': 'onnx',
                'opset': 22
            })
        with self.assertRaisesRegex(ConfigurationError, 'requires ir_config.opset=18'):
            CompilePipeline(config).execute()


if __name__ == '__main__':
    unittest.main()
