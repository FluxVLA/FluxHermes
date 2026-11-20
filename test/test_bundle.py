import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fluxhermes.bundle import create_bundle, describe_bundle, verify_bundle
from fluxhermes.cli.inspect_bundle import main as inspect_main
from fluxhermes.contracts import BundleManifest, TensorSpec
from fluxhermes.errors import ArtifactError, DependencyError
from fluxhermes.utils import read_json, sha256_file, write_json


def manifest():
    return BundleManifest(
        schema_version=2,
        name='demo',
        created_at=datetime.now(timezone.utc).isoformat(),
        backend='tensorrt',
        precision='fp16',
        backend_files=('backend/model.plan',),
        backend_metadata={'tensorrt_version': '10.0'},
        source={},
        inputs=(TensorSpec(name='x', dtype='float32', shape=(1, 2)),),
        outputs=(TensorSpec(name='y', dtype='float32', shape=(1, 2)),),
        runtime={'action_output': 'y'},
        validation={'passed': True},
        files={},
    )


class BundleTest(unittest.TestCase):

    def test_create_verify_and_detect_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / 'source.plan'
            engine.write_bytes(b'engine')
            bundle = create_bundle(
                root / 'bundle',
                manifest(),
                {'backend/model.plan': engine},
            )
            loaded = verify_bundle(bundle)
            self.assertTrue(loaded.validation['passed'])
            (bundle / 'backend/model.plan').write_bytes(b'changed')
            with self.assertRaises(ArtifactError):
                verify_bundle(bundle)

    def test_hardlinks_backend_file_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / 'source.plan'
            engine.write_bytes(b'engine')
            with patch.dict(
                    'os.environ',
                    {'FLUXHERMES_BUNDLE_FILE_MODE': 'hardlink'}):
                bundle = create_bundle(
                    root / 'bundle',
                    manifest(),
                    {'backend/model.plan': engine},
                )
            target = bundle / 'backend/model.plan'
            self.assertEqual(engine.stat().st_ino, target.stat().st_ino)
            verify_bundle(bundle)

    def test_rejects_unknown_bundle_file_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / 'source.plan'
            engine.write_bytes(b'engine')
            with patch.dict(
                    'os.environ',
                    {'FLUXHERMES_BUNDLE_FILE_MODE': 'invalid'}):
                with self.assertRaisesRegex(ArtifactError, 'expected'):
                    create_bundle(
                        root / 'bundle',
                        manifest(),
                        {'backend/model.plan': engine},
                    )

    def test_copies_asset_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / 'source.plan'
            engine.write_bytes(b'engine')
            assets = root / 'processor'
            assets.mkdir()
            (assets / 'stats.json').write_text('{}', encoding='utf-8')
            bundle = create_bundle(
                root / 'bundle',
                manifest(),
                {'backend/model.plan': engine},
                assets=[assets],
            )
            verify_bundle(bundle)
            self.assertTrue((bundle / 'assets/processor/stats.json').is_file())

    def test_rejects_unsafe_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / 'source.plan'
            engine.write_bytes(b'engine')
            with self.assertRaises(ArtifactError):
                create_bundle(
                    root / 'bundle',
                    manifest(),
                    {'../outside.plan': engine},
                )

    def test_reads_legacy_single_engine_manifest(self):
        value = manifest().to_dict()
        value['schema_version'] = 1
        value['engine'] = {
            'path': 'backend/model.plan',
            'tensorrt_version': '10.0',
        }
        value.pop('backend_files')
        value.pop('backend_metadata')
        loaded = BundleManifest.from_dict(value)
        self.assertEqual(loaded.backend_files, ('backend/model.plan',))

    def test_rejects_backend_file_without_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            (root / 'backend').mkdir()
            (root / 'backend/model.plan').write_bytes(b'engine')
            write_json(root / 'manifest.json', manifest().to_dict())
            with self.assertRaisesRegex(ArtifactError, 'not covered'):
                verify_bundle(root)


@unittest.skipUnless(importlib.util.find_spec('onnx'), 'requires ONNX')
class ExternalDataBundleTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.engine = self.source / 'model.plan'
        self.engine.write_bytes(b'engine')

    def graph(self, location='weights.bin', width=2):
        import struct

        import onnx
        from onnx import TensorProto, helper

        tensor = helper.make_tensor(
            'weight', TensorProto.FLOAT, [1, width],
            struct.pack(f'<{width}f', *range(1, width + 1)), raw=True)
        graph = helper.make_graph(
            [helper.make_node('Add', ['x', 'weight'], ['y'])],
            'external',
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, width])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, width])],
            initializer=[tensor])
        model = helper.make_model(graph)
        path = self.source / 'source.onnx'
        (self.source / location).parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            location=location,
            size_threshold=0)
        return path

    def package(self, graph, **extra):
        return create_bundle(self.root / 'bundle', manifest(), {
            'backend/model.plan': self.engine,
            'model/model.onnx': graph,
            **extra,
        })

    def test_discovers_arbitrary_external_names_and_preserves_subdirectories(self):
        import onnx

        for mode in ('copy', 'hardlink'):
            for location in ('weights.bin', 'weights', 'nested/part-000.bin'):
                with self.subTest(mode=mode, location=location):
                    graph = self.graph(location)
                    with patch.dict('os.environ',
                                    {'FLUXHERMES_BUNDLE_FILE_MODE': mode}):
                        bundle = self.package(graph)
                    relative = 'model/' + location
                    loaded = verify_bundle(bundle)
                    self.assertIn(relative, loaded.files)
                    target = bundle / relative
                    self.assertEqual(target.read_bytes(),
                                     (self.source / location).read_bytes())
                    if mode == 'hardlink':
                        self.assertEqual(target.stat().st_ino,
                                         (self.source / location).stat().st_ino)
                    # Loading the actual relocated graph must resolve the weights.
                    model = onnx.load(bundle / 'model/model.onnx')
                    self.assertTrue(model.graph.initializer[0].raw_data)

    def test_external_data_with_onnx_suffix_is_not_treated_as_a_graph(self):
        import onnx

        graph = self.graph('aaa.onnx')
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                files = {'backend/model.plan': self.engine}
                if explicit:
                    # Exercise a sidecar encountered before the referencing graph.
                    files['model/aaa.onnx'] = self.source / 'aaa.onnx'
                files['model/model.onnx'] = graph
                bundle = create_bundle(self.root / 'bundle', manifest(), files)
                self.assertIn('model/aaa.onnx', verify_bundle(bundle).files)
                model = onnx.load(bundle / 'model/model.onnx')
                self.assertTrue(model.graph.initializer[0].raw_data)

    def test_skips_loading_larger_onnx_sidecar_before_parsing(self):
        import onnx

        graph = self.graph('large.onnx', width=16384)
        sidecar = self.source / 'large.onnx'
        self.assertGreater(sidecar.stat().st_size, graph.stat().st_size)
        files = {
            'model/large.onnx': sidecar,
            'model/model.onnx': graph,
            'backend/model.plan': self.engine,
        }
        with patch('onnx.load', wraps=onnx.load) as load:
            bundle = create_bundle(self.root / 'bundle', manifest(), files)
            self.assertIn('model/large.onnx', verify_bundle(bundle).files)
            describe_bundle(bundle)
        self.assertGreater(load.call_count, 0)
        self.assertNotIn('large.onnx',
                         [Path(call.args[0]).name for call in load.call_args_list])
        self.assertTrue(all(call.kwargs.get('load_external_data') is False
                            for call in load.call_args_list))

    def test_unreferenced_malformed_graph_still_fails_next_to_valid_graph(self):
        graph = self.graph('aaa.onnx')
        broken = self.source / 'broken.onnx'
        broken.write_bytes(b'not an ONNX protobuf')
        with self.assertRaisesRegex(ArtifactError, 'ONNX.*broken.onnx'):
            self.package(graph, **{'model/broken.onnx': broken})

    def test_load_onnx_package_transform_uses_references_not_globs(self):
        from fluxhermes.backend.base import BackendBuildResult
        from fluxhermes.contracts import DeploymentConfig
        from fluxhermes.core import PipelineManager
        from fluxhermes.transforms.stages import LoadONNX, PackageBundle

        graph = self.graph('nested/weights.bin')
        (self.source / 'source.onnx.unrelated').write_bytes(b'not a dependency')
        config = DeploymentConfig.from_dict(
            dict(
                name='external',
                work_dir=str(self.root),
                inputs=[dict(name='x', dtype='float32', shape=(1, 2))],
                outputs=[dict(name='y', dtype='float32', shape=(1, 2))],
                package=dict(include_onnx=True)))
        write_json(self.root / 'resolved_config.json', config.to_dict())
        data = LoadONNX(str(graph))({'config': config})
        data.update(
            backend_result=BackendBuildResult((self.engine, ), {}),
            manager=PipelineManager())
        bundle = PackageBundle()(data)['bundle']
        files = verify_bundle(bundle).files
        self.assertIn('model/nested/weights.bin', files)
        self.assertNotIn('model/source.onnx.unrelated', files)

    def test_collects_nested_attribute_and_sparse_tensor_references(self):
        import struct

        import onnx
        from onnx import TensorProto, helper
        from onnx.external_data_helper import set_external_data

        graph = self.graph()
        model = onnx.load(graph, load_external_data=False)

        def external(name, location, dtype=TensorProto.FLOAT):
            raw = struct.pack('<q', 0) if dtype == TensorProto.INT64 else struct.pack(
                '<f', 1)
            tensor = helper.make_tensor(name, dtype, [1], raw, raw=True)
            set_external_data(tensor, location=location)
            tensor.ClearField('raw_data')
            path = self.source / location
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            return tensor

        value_info = helper.make_tensor_value_info('nested_value', TensorProto.FLOAT,
                                                   [1])
        branch = helper.make_graph(
            [helper.make_node('Identity', ['nested_weight'], ['nested_value'])],
            'branch', [], [value_info],
            initializer=[external('nested_weight', 'subgraph/weights.bin')])
        model.graph.input.append(
            helper.make_tensor_value_info('condition', TensorProto.BOOL, []))
        model.graph.node.append(
            helper.make_node(
                'If', ['condition'], ['branch_output'],
                then_branch=branch,
                else_branch=branch))
        model.graph.node.append(
            helper.make_node(
                'Constant', [], ['constant'],
                value=external('constant', 'attributes/value')))
        model.graph.sparse_initializer.append(
            helper.make_sparse_tensor(
                external('sparse', 'sparse/values.bin'),
                external('indices', 'sparse/indices.bin', TensorProto.INT64), [1]))
        onnx.save_model(model, graph)
        bundle = self.package(graph)
        files = verify_bundle(bundle).files
        for location in ('weights.bin', 'subgraph/weights.bin', 'attributes/value',
                         'sparse/values.bin', 'sparse/indices.bin'):
            self.assertIn('model/' + location, files)

    def test_rejects_missing_source_external_data(self):
        graph = self.graph()
        missing = self.source / 'weights.bin'
        self.assertEqual(missing.resolve(), self.source.resolve() / 'weights.bin')
        missing.unlink()
        with self.assertRaisesRegex(ArtifactError, 'weights.bin'):
            self.package(graph)

    def test_rejects_conflicting_explicit_external_destination(self):
        graph = self.graph()
        other = self.source / 'other.bin'
        other.write_bytes(b'wrong weights')
        with self.assertRaisesRegex(ArtifactError, 'conflict.*model/weights.bin'):
            self.package(graph, **{'model/weights.bin': other})

    def test_accepts_already_listed_external_data_once(self):
        graph = self.graph()
        bundle = self.package(graph,
                              **{'model/weights.bin': self.source / 'weights.bin'})
        self.assertEqual(len(verify_bundle(bundle).files), 3)

    def test_verify_and_inspect_reject_untracked_external_reference(self):
        graph = self.graph()
        bundle = self.package(graph,
                              **{'model/weights.bin': self.source / 'weights.bin'})
        value = read_json(bundle / 'manifest.json')
        value['files'].pop('model/weights.bin')
        write_json(bundle / 'manifest.json', value)
        for action in (verify_bundle, describe_bundle):
            with self.subTest(action=action.__name__):
                with self.assertRaisesRegex(ArtifactError, 'not covered.*weights.bin'):
                    action(bundle)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(inspect_main([str(bundle)]), 2)
        self.assertIn('weights.bin', stderr.getvalue())

    def test_verify_rejects_missing_or_tampered_external_data(self):
        graph = self.graph()
        for tamper in (False, True):
            with self.subTest(tamper=tamper):
                bundle = self.package(graph)
                target = bundle / 'model/weights.bin'
                self.assertIn('model/weights.bin', verify_bundle(bundle).files)
                if tamper:
                    target.write_bytes(b'changed')
                else:
                    self.assertEqual(target.resolve(), bundle / 'model/weights.bin')
                    target.unlink()
                with self.assertRaisesRegex(ArtifactError, 'weights.bin'):
                    verify_bundle(bundle)

    def test_rejects_unsafe_or_missing_external_locations(self):
        import onnx

        graph = self.graph()
        for location in ('../outside.bin', '/absolute.bin', ''):
            with self.subTest(location=location):
                model = onnx.load(graph, load_external_data=False)
                tensor = model.graph.initializer[0]
                del tensor.external_data[:]
                if location:
                    tensor.external_data.add(key='location', value=location)
                onnx.save_model(model, graph)
                with self.assertRaisesRegex(ArtifactError, 'external data.*location'):
                    self.package(graph)

    def test_verify_rejects_unsafe_reference_even_with_matching_graph_checksum(self):
        import onnx

        graph = self.graph()
        bundle = self.package(graph,
                              **{'model/weights.bin': self.source / 'weights.bin'})
        target = bundle / 'model/model.onnx'
        model = onnx.load(target, load_external_data=False)
        for entry in model.graph.initializer[0].external_data:
            if entry.key == 'location':
                entry.value = '../outside.bin'
        onnx.save_model(model, target)
        value = read_json(bundle / 'manifest.json')
        value['files']['model/model.onnx'] = sha256_file(target)
        write_json(bundle / 'manifest.json', value)
        with self.assertRaisesRegex(ArtifactError, 'external data.*location'):
            verify_bundle(bundle)

    def test_onnx_dependency_failure_is_explicit(self):
        graph = self.graph()
        with patch.dict('sys.modules', {'onnx': None}):
            with self.assertRaisesRegex(DependencyError, 'onnx'):
                self.package(graph)

    def test_rejects_malformed_onnx_instead_of_publishing_bundle(self):
        graph = self.source / 'broken.onnx'
        graph.write_bytes(b'not an ONNX protobuf')
        with self.assertRaisesRegex(ArtifactError, 'ONNX.*broken.onnx'):
            self.package(graph)


if __name__ == '__main__':
    unittest.main()
