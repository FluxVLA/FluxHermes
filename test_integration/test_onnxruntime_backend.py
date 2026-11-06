"""Real optional ONNX Runtime CPU execution and portable external-data tests."""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from fluxhermes.backend import get_backend_manager
from fluxhermes.backend.onnxruntime import ONNXRuntimeBackendManager
from fluxhermes.bundle import create_bundle, verify_bundle
from fluxhermes.contracts import BundleManifest, DeploymentConfig
from fluxhermes.errors import ArtifactError, ConfigurationError
from fluxhermes.exporters import artifacts as onnx_artifacts
from fluxhermes.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def config(root, **backend):
    return DeploymentConfig.from_dict({
        'name':
        'real-ort-test',
        'work_dir':
        str(root),
        'codebase_config': {
            'type': 'python',
            'task': 'custom',
            'device': 'cpu',
            'source': {
                'type': 'python',
                'factory': 'torch.nn:Identity'
            }
        },
        'backend_config': {
            'type': 'onnxruntime',
            'precision': 'fp32',
            **backend
        },
        'ir_config': {
            'type': 'onnx'
        },
        'inputs': [
            {
                'name': 'x',
                'dtype': 'float32',
                'shape': [2, 2]
            },
            {
                'name': 'mask',
                'dtype': 'bool',
                'shape': [2, 2]
            },
            {
                'name': 'steps',
                'dtype': 'int64',
                'shape': [2, 2]
            },
        ],
        'outputs': [
            {
                'name': 'y',
                'dtype': 'float32',
                'shape': [2, 2]
            },
            {
                'name': 'mask_out',
                'dtype': 'bool',
                'shape': [2, 2]
            },
            {
                'name': 'steps_out',
                'dtype': 'int64',
                'shape': [2, 2]
            },
        ],
    })


def graph(root, external=False):
    root.mkdir(parents=True, exist_ok=True)
    weights = [
        numpy_helper.from_array(np.array(value, dtype=np.float32), name)
        for name, value in (('weight_a', [0.25, -0.75]), ('weight_b', [1.0, 2.0]))
    ]
    if external:
        for tensor, directory in zip(weights, ('first', 'second')):
            relative = f'{directory}/weights.bin'
            path = root / relative
            path.parent.mkdir()
            prefix = b'weight-header'
            path.write_bytes(prefix + tensor.raw_data + b'weight-tail')
            onnx.external_data_helper.set_external_data(
                tensor,
                location=relative,
                offset=len(prefix),
                length=len(tensor.raw_data),
                checksum=hashlib.sha1(path.read_bytes()).hexdigest())
            tensor.ClearField('raw_data')
    inputs = [
        helper.make_tensor_value_info(name, dtype, ['batch', 2])
        for name, dtype in (('x', TensorProto.FLOAT), ('mask', TensorProto.BOOL),
                            ('steps', TensorProto.INT64))
    ]
    outputs = [
        helper.make_tensor_value_info(name, dtype, ['batch', 2])
        for name, dtype in (('y', TensorProto.FLOAT), ('mask_out', TensorProto.BOOL),
                            ('steps_out', TensorProto.INT64))
    ]
    nodes = [
        helper.make_node('Add', ['x', 'weight_a'], ['shifted']),
        helper.make_node('Cast', ['steps'], ['step_values'], to=TensorProto.FLOAT),
        helper.make_node('Add', ['shifted', 'step_values'], ['summed']),
        helper.make_node('Where', ['mask', 'summed', 'weight_b'], ['y']),
        helper.make_node('Identity', ['mask'], ['mask_out']),
        helper.make_node('Identity', ['steps'], ['steps_out'])
    ]
    model = helper.make_model(
        helper.make_graph(nodes, 'real-graph', inputs, outputs, weights),
        opset_imports=[helper.make_opsetid('', 17)],
        ir_version=10)
    path = root / 'source.onnx'
    path.write_bytes(model.SerializeToString())
    return path


def inputs(batch=2):
    return {
        'x': torch.arange(batch * 2, dtype=torch.float32).reshape(batch, 2) / 3,
        'mask': (torch.arange(batch * 2).reshape(batch, 2) % 3 != 0),
        'steps': torch.arange(batch * 2, dtype=torch.int64).reshape(batch, 2)
    }


def expected(values):
    return {
        'y':
        torch.where(values['mask'],
                    values['x'] + torch.tensor([0.25, -0.75]) + values['steps'].float(),
                    torch.tensor([1.0, 2.0])),
        'mask_out':
        values['mask'],
        'steps_out':
        values['steps']
    }


def bf16_graph(root, constant_attribute=False):
    """Valid opset-22 graph with actual BF16 external tensor bytes."""
    root.mkdir(parents=True, exist_ok=True)
    weights = torch.tensor([0.25, -0.75], dtype=torch.bfloat16)
    raw = weights.view(torch.uint16).numpy().tobytes()
    tensor = TensorProto(
        name='bf16_weight', data_type=TensorProto.BFLOAT16, dims=[2], raw_data=raw)
    data = root / 'weights/bf16.bin'
    data.parent.mkdir()
    prefix = b'bf16-header'
    data.write_bytes(prefix + raw + b'bf16-tail')
    onnx.external_data_helper.set_external_data(
        tensor,
        location='weights/bf16.bin',
        offset=len(prefix),
        length=len(raw),
        checksum=hashlib.sha1(data.read_bytes()).hexdigest())
    tensor.ClearField('raw_data')
    nodes = [
        helper.make_node(
            'Cast', ['bf16_weight'], ['float_weight'], to=TensorProto.FLOAT),
        helper.make_node('Add', ['x', 'float_weight'], ['y'])
    ]
    initializers = [tensor]
    if constant_attribute:
        nodes.insert(0, helper.make_node('Constant', [], ['bf16_weight'], value=tensor))
        initializers = []
    model = helper.make_model(
        helper.make_graph(
            nodes, 'actual-bf16',
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])],
            initializers),
        opset_imports=[helper.make_opsetid('', 22)],
        ir_version=10)
    path = root / 'bf16.onnx'
    path.write_bytes(model.SerializeToString())
    return path, raw


class ONNXRuntimeBackendTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='hermes-real-ort-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def build(self, external=False):
        source = graph(self.root / 'source', external=external)
        result = ONNXRuntimeBackendManager.to_backend([source], self.root / 'build',
                                                      config(self.root))
        return source, result

    def wrapper(self, result, **kwargs):
        return ONNXRuntimeBackendManager.build_wrapper(
            result.files,
            device='cpu',
            target_metadata=result.metadata,
            require_target_match=True,
            **kwargs)

    def assert_results(self, actual, values):
        self.assertEqual(set(actual), {'y', 'mask_out', 'steps_out'})
        for name, reference in expected(values).items():
            self.assertIsInstance(actual[name], torch.Tensor)
            self.assertEqual(actual[name].device.type, 'cpu')
            torch.testing.assert_close(actual[name], reference, rtol=0, atol=0)

    def test_registry_import_is_lazy_and_preserves_tensorrt(self):
        command = '''import sys
from fluxhermes.backend import get_backend_manager
assert get_backend_manager('onnxruntime').backend_name == 'onnxruntime'
assert get_backend_manager('tensorrt').backend_name == 'tensorrt'
assert not set(('onnxruntime', 'onnx', 'torch', 'tensorrt')).intersection(sys.modules)
'''
        completed = subprocess.run([sys.executable, '-c', command],
                                   cwd=ROOT,
                                   text=True,
                                   capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIs(get_backend_manager('onnxruntime'), ONNXRuntimeBackendManager)
        self.assertTrue(ONNXRuntimeBackendManager.is_available())
        self.assertTrue(ONNXRuntimeBackendManager.get_version())

    def test_missing_optional_dependencies_fail_explicitly_without_site_packages(self):
        # Load only the actual package under test, not its site-packages parent.
        # This preserves missing optional dependencies under both wheel and
        # checkout execution without depending on the subprocess cwd.
        package_init = Path(sys.modules['fluxhermes'].__file__).resolve()
        command = '''import importlib.util
import sys
spec = importlib.util.spec_from_file_location('fluxhermes', sys.argv[1])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from fluxhermes.backend import get_backend_manager
from fluxhermes.errors import DependencyError
manager = get_backend_manager('onnxruntime')
assert not manager.is_available()
try:
    manager.get_version()
except DependencyError as error:
    assert 'optional dependency onnxruntime' in str(error)
else:
    raise AssertionError('Missing ONNX Runtime must not silently succeed')
'''
        completed = subprocess.run(
            [sys.executable, '-S', '-c', command, str(package_init)],
            cwd=self.root,
            text=True,
            capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_external_constant_attribute_is_copied_and_executed(self):
        source = graph(self.root / 'source', external=True)
        model = onnx.load_model(str(source), load_external_data=False)
        tensor = copy.deepcopy(model.graph.initializer[0])
        model.graph.initializer.remove(model.graph.initializer[0])
        model.graph.node.insert(
            0, helper.make_node('Constant', [], ['weight_a'], value=tensor))
        source.write_bytes(model.SerializeToString())
        result = ONNXRuntimeBackendManager.to_backend([source], self.root / 'build',
                                                      config(self.root))
        self.assertEqual(len(result.files), 3)
        self.assert_results(self.wrapper(result)(inputs()), inputs())

    def test_real_named_mixed_dtype_cpu_inference_and_dynamic_batch(self):
        _, result = self.build()
        wrapper = self.wrapper(result)
        self.assertEqual(wrapper.session.get_providers(), ['CPUExecutionProvider'])
        self.assertEqual(wrapper.input_names, ['x', 'mask', 'steps'])
        self.assertEqual(wrapper.output_names, ['y', 'mask_out', 'steps_out'])
        self.assertEqual(result.metadata['backend'], 'onnxruntime')
        self.assertEqual(result.metadata['execution_provider'], 'CPUExecutionProvider')
        for batch in (2, 3):
            values = inputs(batch)
            before = {name: tensor.clone() for name, tensor in values.items()}
            rng = torch.random.get_rng_state().clone()
            self.assert_results(wrapper.infer(values, synchronize=True), values)
            self.assert_results(wrapper(values), values)
            for name in values:
                torch.testing.assert_close(values[name], before[name], rtol=0, atol=0)
            torch.testing.assert_close(
                torch.random.get_rng_state(), rng, rtol=0, atol=0)

    def test_real_noncontiguous_and_requires_grad_inputs(self):
        _, result = self.build()
        values = inputs()
        values['x'] = values['x'].T.requires_grad_()
        self.assertFalse(values['x'].is_contiguous())
        self.assert_results(self.wrapper(result)(values), values)
        self.assertIsNone(values['x'].grad)

    def test_rejects_missing_extra_names_and_non_tensor_values(self):
        _, result = self.build()
        wrapper = self.wrapper(result)
        with self.assertRaisesRegex(ConfigurationError, 'input names differ'):
            wrapper({'x': inputs()['x']})
        with self.assertRaisesRegex(ConfigurationError, 'input names differ'):
            wrapper({**inputs(), 'extra': torch.ones(2, 2)})
        with self.assertRaisesRegex(TypeError, 'Torch CPU Tensor'):
            wrapper({**inputs(), 'x': np.ones((2, 2), dtype=np.float32)})

    def test_rejects_wrong_dtype_rank_fixed_and_symbolic_dimensions(self):
        _, result = self.build()
        wrapper = self.wrapper(result)
        for replacement, message in (
            ({
                'steps': torch.ones(2, 2, dtype=torch.int32)
            }, 'expected dtype'),
            ({
                'mask': torch.ones(2, 2)
            }, 'expected dtype'),
            ({
                'x': torch.ones(2, 2, dtype=torch.bfloat16)
            }, 'BF16'),
            ({
                'x': torch.ones(4)
            }, 'expected rank'),
            ({
                'x': torch.ones(2, 3)
            }, 'axis 1'),
            ({
                'mask': torch.ones(3, 2, dtype=torch.bool)
            }, 'symbolic dimension'),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                    ConfigurationError, message):
                wrapper({**inputs(), **replacement})

    def test_requires_explicit_cpu_and_rejects_profiles_and_provider_options(self):
        _, result = self.build()
        for device in ('cuda:0', 'mps', 'cpu:0', '', None):
            with self.subTest(device=device), self.assertRaisesRegex(
                    ConfigurationError, 'explicit device'):
                ONNXRuntimeBackendManager.build_wrapper(result.files, device=device)
        with self.assertRaisesRegex(ConfigurationError, 'profile_index'):
            self.wrapper(result, profile_index=1)
        with self.assertRaisesRegex(ConfigurationError, 'Unsupported.*options'):
            self.wrapper(result, providers=['CUDAExecutionProvider'])
        with self.assertRaisesRegex(ConfigurationError, 'metadata is required'):
            ONNXRuntimeBackendManager.build_wrapper(
                result.files, device='cpu', require_target_match=True)

    def test_external_weights_are_flat_distinct_exact_and_portable_in_bundle(self):
        source, result = self.build(external=True)
        original = source.read_bytes()
        self.assertEqual([path.name for path in result.files],
                         ['model.onnx', 'external_0000.data', 'external_0001.data'])
        self.assertNotEqual(sha256_file(result.files[1]), sha256_file(result.files[2]))
        for record in result.metadata['external_data']:
            self.assertEqual(
                sha256_file(source.parent / record['source_location']),
                record['sha256'])
            self.assertEqual(
                sha256_file(result.files[0].parent / record['file']), record['sha256'])
        self.assertEqual(source.read_bytes(), original)
        deployed = onnx.load_model(str(result.files[0]), load_external_data=False)
        self.assertEqual(
            {
                dict((entry.key, entry.value)
                     for entry in tensor.external_data)['location']
                for tensor in deployed.graph.initializer
            }, {'external_0000.data', 'external_0001.data'})
        cfg = config(self.root)
        manifest = BundleManifest(
            schema_version=2,
            name='real-ort',
            created_at=datetime.now(timezone.utc).isoformat(),
            backend='onnxruntime',
            precision='fp32',
            backend_files=tuple(f'backend/prefix/{path.name}' for path in result.files),
            backend_metadata=result.metadata,
            source={},
            inputs=cfg.inputs,
            outputs=cfg.outputs,
            runtime={'action_output': 'y'},
            validation={'passed': True},
            files={})
        bundle = create_bundle(
            self.root / 'bundle', manifest,
            {f'backend/prefix/{path.name}': path
             for path in result.files})
        verified = verify_bundle(bundle)
        # Relocation uses only the actual checksummed bundle, not source paths.
        with tempfile.TemporaryDirectory(prefix='hermes-relocated-ort-') as directory:
            relocated = Path(directory) / 'bundle'
            shutil.copytree(bundle, relocated)
            wrapper = ONNXRuntimeBackendManager.build_wrapper(
                [relocated / name for name in verified.backend_files],
                device='cpu',
                target_metadata=verified.backend_metadata,
                require_target_match=True)
            self.assert_results(wrapper(inputs()), inputs())

    def test_detects_external_weight_tamper_and_missing_artifact_coverage(self):
        _, result = self.build(external=True)
        with self.assertRaisesRegex(ArtifactError, 'coverage differs'):
            ONNXRuntimeBackendManager.build_wrapper(result.files[:-1], device='cpu')
        data = result.files[1].read_bytes()
        # A valid-length modification must not evade byte-level integrity checks.
        result.files[1].write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        with self.assertRaisesRegex(ArtifactError,
                                    'checksum mismatch|integrity mismatch'):
            self.wrapper(result)

    def test_rejects_missing_external_data_before_copy(self):
        source = graph(self.root / 'source', external=True)
        copied_graph = self.root / 'graph-only.onnx'
        copied_graph.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ArtifactError, 'external data is missing'):
            ONNXRuntimeBackendManager.to_backend([copied_graph], self.root / 'build',
                                                 config(self.root))
        self.assertFalse((self.root / 'build/onnxruntime').exists())

    def test_rejects_external_path_traversal_absolute_windows_and_noncanonical_paths(
            self):
        source = graph(self.root / 'source', external=True)
        original = onnx.load_model(str(source), load_external_data=False)
        for location in ('../outside.bin', '/tmp/outside.bin', 'C:\\outside.bin',
                         'first/../first/weights.bin', './first/weights.bin', '',
                         'file:///tmp/a'):
            model = copy.deepcopy(original)
            model.graph.initializer[0].external_data[0].value = location
            source.write_bytes(model.SerializeToString())
            with self.subTest(location=location), self.assertRaisesRegex(
                    ArtifactError, 'Unsafe ONNX'):
                ONNXRuntimeBackendManager.to_backend([source], self.root / 'build',
                                                     config(self.root))

    def test_rejects_external_symlink_escape_and_out_of_bounds_ranges(self):
        source = graph(self.root / 'source', external=True)
        original = onnx.load_model(str(source), load_external_data=False)
        outside = self.root / 'outside.bin'
        outside.write_bytes((source.parent / 'first/weights.bin').read_bytes())
        (source.parent / 'escape.bin').symlink_to(outside)
        model = copy.deepcopy(original)
        model.graph.initializer[0].external_data[0].value = 'escape.bin'
        source.write_bytes(model.SerializeToString())
        with self.assertRaisesRegex(ArtifactError, 'escapes the graph directory'):
            ONNXRuntimeBackendManager.to_backend([source], self.root / 'build',
                                                 config(self.root))
        model = copy.deepcopy(original)
        for entry in model.graph.initializer[0].external_data:
            if entry.key == 'length':
                entry.value = '100000'
        source.write_bytes(model.SerializeToString())
        with self.assertRaisesRegex(ArtifactError, 'external-data range'):
            ONNXRuntimeBackendManager.to_backend([source], self.root / 'build',
                                                 config(self.root))

    def test_rejects_bad_metadata_and_preexisting_build_outputs(self):
        source, result = self.build()
        bad = {**result.metadata, 'execution_provider': 'CUDAExecutionProvider'}
        with self.assertRaisesRegex(ArtifactError, 'CPU backend'):
            ONNXRuntimeBackendManager.build_wrapper(
                result.files, device='cpu', target_metadata=bad)
        bad = {**result.metadata, 'artifacts': {}}
        with self.assertRaisesRegex(ArtifactError, 'cover every backend file'):
            ONNXRuntimeBackendManager.build_wrapper(
                result.files, device='cpu', target_metadata=bad)
        with self.assertRaisesRegex(ArtifactError, 'already exists'):
            ONNXRuntimeBackendManager.to_backend([source], self.root / 'build',
                                                 config(self.root))
        self.assert_results(self.wrapper(result)(inputs()), inputs())

    def test_rejects_multiple_graphs_and_unrequested_backend_or_precision(self):
        source = graph(self.root / 'source')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            ONNXRuntimeBackendManager.to_backend([source, source], self.root / 'build',
                                                 config(self.root))
        for options, message in (({
                'type': 'tensorrt'
        }, 'backend_config.type'), ({
                'precision': 'bf16'
        }, 'precision'), ({
                'providers': ['CUDAExecutionProvider']
        }, 'only providers'), ({
                'device': 'cuda:0'
        }, 'explicit device')):
            with self.subTest(options=options), self.assertRaisesRegex(
                    ConfigurationError, message):
                ONNXRuntimeBackendManager.to_backend([source], self.root / 'build',
                                                     config(self.root, **options))

    def test_generic_bf16_opset22_staging_and_relocation_without_ort_import(self):
        source, raw = bf16_graph(self.root / 'bf16-source')
        original_graph = source.read_bytes()
        original_weights = (source.parent / 'weights/bf16.bin').read_bytes()
        command = (
            'import json, sys\n'
            'from pathlib import Path\n'
            'from fluxhermes.exporters.artifacts import checked_model, '
            'stage_graph, validate_backend_files\n'
            "assert 'onnxruntime' not in sys.modules\n"
            'files, metadata = stage_graph(Path(sys.argv[1]), Path(sys.argv[2]))\n'
            'validate_backend_files(files, metadata)\n'
            "assert 'onnxruntime' not in sys.modules\n"
            "assert 'torch' not in sys.modules\n"
            'print(json.dumps(metadata))\n'
        )
        staged = self.root / 'bf16-staged'
        completed = subprocess.run(
            [sys.executable, '-c', command,
             str(source), str(staged)],
            cwd=ROOT,
            text=True,
            capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        metadata = json.loads(completed.stdout)
        self.assertEqual(source.read_bytes(), original_graph)
        self.assertEqual((source.parent / 'weights/bf16.bin').read_bytes(),
                         original_weights)
        self.assertEqual((staged / 'external_0000.data').read_bytes(), original_weights)
        # Generic metadata validates artifact integrity, not backend/provider policy.
        metadata.update(backend='tensorrt', device='cuda:0', precision='bf16')
        relocated = self.root / 'bf16-relocated'
        shutil.copytree(staged, relocated)
        files = [relocated / 'model.onnx', relocated / 'external_0000.data']
        graph_path = onnx_artifacts.validate_backend_files(files, metadata)
        model, external = onnx_artifacts.checked_model(graph_path)
        self.assertEqual(model.opset_import[0].version, 22)
        self.assertEqual(model.graph.initializer[0].data_type, TensorProto.BFLOAT16)
        self.assertEqual(set(external), {'external_0000.data'})
        loaded = onnx.load_model(str(graph_path), load_external_data=True)
        self.assertEqual(loaded.graph.initializer[0].raw_data, raw)
        self.assertEqual(
            sha256_file(files[1]), metadata['artifacts'][files[1].name]['sha256'])
        with self.assertRaisesRegex(ConfigurationError, 'BF16 is unsupported'):
            ONNXRuntimeBackendManager.build_wrapper(files, device='cpu')

    def test_cpu_manager_rejects_bf16_initializers_and_nested_constants_before_copy(
            self):
        for nested in (False, True):
            with self.subTest(nested=nested):
                source, raw = bf16_graph(self.root / f'bf16-source-{nested}', nested)
                files, metadata = onnx_artifacts.stage_graph(
                    source, self.root / f'bf16-generic-{nested}')
                onnx_artifacts.validate_backend_files(files, metadata)
                model = onnx.load_model(str(files[0]), load_external_data=True)
                tensors = [
                    value for value in onnx_artifacts.iter_model_messages(model)
                    if isinstance(value, TensorProto)
                ]
                self.assertEqual(len(tensors), 1)
                self.assertEqual(tensors[0].raw_data, raw)
                destination = self.root / f'bf16-cpu-{nested}'
                with self.assertRaisesRegex(ConfigurationError, 'BF16 is unsupported'):
                    ONNXRuntimeBackendManager.to_backend([source], destination,
                                                         config(self.root))
                self.assertFalse((destination / 'onnxruntime').exists())
                with self.assertRaisesRegex(ConfigurationError, 'BF16 is unsupported'):
                    ONNXRuntimeBackendManager.build_wrapper(files, device='cpu')

    def test_generic_artifact_aliases_share_one_safety_implementation(self):
        from fluxhermes.backend.onnxruntime import artifacts as compatibility
        for name in ('checked_model', 'stage_graph', 'validate_backend_files'):
            self.assertIs(getattr(compatibility, name), getattr(onnx_artifacts, name))

    def test_generic_checker_runs_full_type_inference(self):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node('Add', ['x', 'steps'], ['y'])], 'invalid-mixed-types',
                [
                    helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2]),
                    helper.make_tensor_value_info('steps', TensorProto.INT64, [2, 2])
                ], [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])]),
            opset_imports=[helper.make_opsetid('', 22)],
            ir_version=10)
        path = self.root / 'type-error.onnx'
        path.write_bytes(model.SerializeToString())
        onnx.checker.check_model(str(path), full_check=False)
        with self.assertRaisesRegex(ArtifactError, 'Invalid ONNX graph'):
            onnx_artifacts.checked_model(path)

    def test_rejects_internal_bf16_cast_without_conversion(self):
        source = graph(self.root / 'source')
        model = onnx.load_model(str(source), load_external_data=False)
        model.graph.node.insert(
            0,
            helper.make_node('Cast', ['x'], ['hidden_bf16'], to=TensorProto.BFLOAT16))
        source.write_bytes(model.SerializeToString())
        with self.assertRaisesRegex(ConfigurationError, 'BF16 is unsupported'):
            ONNXRuntimeBackendManager.to_backend([source], self.root / 'build',
                                                 config(self.root))

    def test_rejects_bf16_graph_without_conversion(self):
        input_info = helper.make_tensor_value_info('x', TensorProto.BFLOAT16, [2, 2])
        output_info = helper.make_tensor_value_info('y', TensorProto.BFLOAT16, [2, 2])
        model = helper.make_model(
            helper.make_graph([helper.make_node('Identity', ['x'], ['y'])], 'bf16',
                              [input_info], [output_info]),
            opset_imports=[helper.make_opsetid('', 17)],
            ir_version=10)
        path = self.root / 'bf16.onnx'
        path.write_bytes(model.SerializeToString())
        with self.assertRaisesRegex(ConfigurationError, 'BF16 is unsupported'):
            ONNXRuntimeBackendManager.to_backend([path], self.root / 'build',
                                                 config(self.root))


if __name__ == '__main__':
    unittest.main()
