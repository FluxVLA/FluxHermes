import unittest

from fluxhermes.backend import get_backend_manager
from fluxhermes.codebase import get_codebase
from fluxhermes.contracts import DeploymentConfig
from fluxhermes.core import FunctionRewriter, PipelineManager, Registry


def sample_function(value):
    return value + 1


class RegistryTest(unittest.TestCase):

    def test_register_and_build(self):
        registry = Registry('demo')

        @registry.register_module('item')
        class Item:
            def __init__(self, value):
                self.value = value

        built = registry.build({'type': 'item', 'value': 3})
        self.assertEqual(built.value, 3)


class PipelineManagerTest(unittest.TestCase):

    def test_hooks_and_trace(self):
        manager = PipelineManager()
        events = []
        manager.add_hook(
            'double',
            lambda **kwargs: events.append(kwargs['result']),
            when='after',
        )
        result = manager.run('double', lambda value: value * 2, 4)
        self.assertEqual(result, 8)
        self.assertEqual(events, [8])
        self.assertEqual(manager.report()[0]['status'], 'completed')


class FunctionRewriterTest(unittest.TestCase):

    def test_context_rewrites_and_restores_function(self):
        rewriter = FunctionRewriter()

        @rewriter.register_rewriter(
            'test_core_components.sample_function',
            backend='tensorrt',
            ir='onnx',
        )
        def rewrite(context, value):
            return context.origin_func(value) * 10

        self.assertEqual(sample_function(1), 2)
        with rewriter.context(
            codebase='fluxvla',
            backend='tensorrt',
            ir='onnx',
        ):
            self.assertEqual(sample_function(1), 20)
        self.assertEqual(sample_function(1), 2)


class ComponentRegistryTest(unittest.TestCase):

    def test_builtin_codebase_and_backend_are_discoverable(self):
        codebase = get_codebase('fluxvla')
        backend = get_backend_manager('tensorrt')
        self.assertIn('predict_action', codebase.task_registry.module_dict)
        self.assertEqual(backend.backend_name, 'tensorrt')

    def test_codebase_builds_task_processor_without_loading_model(self):
        config = DeploymentConfig.from_dict({
            'name': 'demo',
            'work_dir': '/tmp/demo',
            'codebase_config': {
                'type': 'python',
                'task': 'custom',
                'source': {'type': 'python', 'factory': 'missing:factory'},
            },
            'backend_config': {'type': 'tensorrt', 'precision': 'fp16'},
            'ir_config': {'type': 'onnx'},
            'inputs': [
                {'name': 'x', 'dtype': 'float32', 'shape': (1, 2)},
            ],
            'outputs': [
                {'name': 'y', 'dtype': 'float32', 'shape': (1, 2)},
            ],
        })
        processor = get_codebase('python').build_task_processor(
            config, device='cuda:0')
        self.assertEqual(processor.device, 'cuda:0')


if __name__ == '__main__':
    unittest.main()
