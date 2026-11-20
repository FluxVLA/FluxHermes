"""Exercise the tensor visitor against real Protobuf descriptors, without Torch."""
import unittest

import onnx
from onnx import TensorProto, helper

from fluxhermes.exporters.onnx import _walk_tensor_protos


class TensorWalkerTest(unittest.TestCase):

    def test_repeated_singular_nested_and_sparse_fields(self):
        def tensor(name):
            return helper.make_tensor(name, TensorProto.FLOAT, [1], [1.0])

        nested = helper.make_graph([], 'nested', [], [], [tensor('nested_initializer')])
        node = helper.make_node('Custom', [], ['output'])
        node.attribute.extend([
            helper.make_attribute('tensor', tensor('attribute_tensor')),
            helper.make_attribute('tensors', [tensor('list_a'), tensor('list_b')]),
            helper.make_attribute('body', nested),
        ])
        graph = helper.make_graph([node], 'root', [], [], [tensor('initializer')])
        graph.sparse_initializer.extend([
            helper.make_sparse_tensor(
                tensor('sparse_values'),
                helper.make_tensor('sparse_indices', TensorProto.INT64, [1], [0]),
                [1])
        ])
        model = helper.make_model(graph)
        # Serialization/reload proves traversal uses the actual installed
        # protobuf implementation, not a hand-built descriptor substitute.
        model = onnx.load_model_from_string(model.SerializeToString())
        names = [item.name for item in _walk_tensor_protos(model)]
        self.assertCountEqual(names, [
            'initializer', 'nested_initializer', 'attribute_tensor',
            'list_a', 'list_b', 'sparse_values', 'sparse_indices'
        ])

    def test_tensor_root_and_empty_model(self):
        tensor = helper.make_tensor('root', TensorProto.FLOAT, [1], [1.0])
        self.assertEqual(list(_walk_tensor_protos(tensor)), [tensor])
        self.assertEqual(list(_walk_tensor_protos(onnx.ModelProto())), [])


if __name__ == '__main__':
    unittest.main()
