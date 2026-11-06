"""Real TensorRT logger regression; requires the actual target CUDA stack.

Run: python -m unittest discover -s test_gpu -p test_tensorrt_logging.py -v
No mocked TensorRT objects, skips, stream redirection or output stripping.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _probe() -> None:
    import onnx
    import tensorrt as trt
    import torch
    from onnx import TensorProto, helper

    from fluxhermes.builders.tensorrt import build_tensorrt_engine
    from fluxhermes.contracts import TensorSpec
    from fluxhermes.runtime.tensorrt import TensorRTEngine
    from fluxhermes.runtime.tensorrt_logger import get_tensorrt_logger

    device = os.environ.get('FLUXHERMES_CUDA_DEVICE', 'cuda:0')
    logger = get_tensorrt_logger(trt.Logger.WARNING)
    assert isinstance(logger, trt.ILogger)
    identity = id(logger)
    logger.log(trt.Logger.WARNING, 'fluxhermes-warning-visible')
    logger.log(trt.Logger.ERROR, 'fluxhermes-error-visible')
    logger.log(trt.Logger.INFO, 'fluxhermes-info-below-warning-threshold')
    assert get_tensorrt_logger(trt.Logger.INFO) is logger
    logger.log(trt.Logger.INFO, 'fluxhermes-info-requested-visible')
    assert get_tensorrt_logger(trt.Logger.WARNING) is logger
    del logger
    gc.collect()
    assert id(get_tensorrt_logger()) == identity

    with tempfile.TemporaryDirectory(prefix='hermes-trt-log-') as directory:
        root = Path(directory)
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node('Add', ['x', 'x'], ['y'])], 'double',
                [helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 4])],
                [helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 4])]),
            opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
        onnx.checker.check_model(model)
        graph = root / 'double.onnx'
        onnx.save(model, graph)
        engines = []
        for index in range(2):
            path = root / f'graph-{index}.plan'
            details = build_tensorrt_engine(
                graph, path, (TensorSpec('x', 'float32', (2, 4)), ), {
                    'device': device, 'precision': 'fp32',
                    'tf32': False, 'workspace_size_gib': 1,
                    'log_level': 'warning',
                })
            engine = TensorRTEngine(
                path, device=device, target_metadata=details,
                require_target_match=True)
            assert id(engine.logger) == identity
            engines.append(engine)
        assert engines[0].logger is engines[1].logger
        assert engines[0].logger is get_tensorrt_logger()
        for engine in engines:
            value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
            actual = engine.infer({'x': value}, synchronize=True)['y'].cpu()
            torch.testing.assert_close(actual, value + value, rtol=0, atol=0)
        print(json.dumps({'passed': True, 'engines': len(engines),
                          'shared_logger': True}))


class TensorRTLoggingTest(unittest.TestCase):

    def test_logger_import_remains_lazy_without_site_packages(self):
        import fluxhermes

        # Execute only the real package under test, not its site-packages parent.
        # This works for an installed wheel without depending on checkout cwd.
        package_init = Path(fluxhermes.__file__).resolve()
        command = (
            'import importlib.util\n'
            'import sys\n'
            "spec = importlib.util.spec_from_file_location('fluxhermes', sys.argv[1])\n"
            'package = importlib.util.module_from_spec(spec)\n'
            'sys.modules[spec.name] = package\n'
            'spec.loader.exec_module(package)\n'
            'from fluxhermes.runtime.tensorrt_logger import get_tensorrt_logger\n'
            'from fluxhermes.builders.tensorrt import build_tensorrt_engine\n'
            "assert 'tensorrt' not in sys.modules\n"
            "assert 'torch' not in sys.modules\n")
        with tempfile.TemporaryDirectory(prefix='hermes-logger-import-') as directory:
            completed = subprocess.run(
                [sys.executable, '-S', '-c', command, str(package_init)],
                cwd=directory, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)

    def test_two_real_engines_keep_json_stdout_and_shared_stderr_logger(self):
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--probe'],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)
        # Parse the original full stdout; never filter TensorRT diagnostics.
        self.assertEqual(json.loads(completed.stdout), {
            'passed': True, 'engines': 2, 'shared_logger': True})
        self.assertIn('[WARNING] fluxhermes-warning-visible', completed.stderr)
        self.assertIn('[ERROR] fluxhermes-error-visible', completed.stderr)
        self.assertIn('[INFO] fluxhermes-info-requested-visible', completed.stderr)
        self.assertNotIn('fluxhermes-info-below-warning-threshold', completed.stderr)
        self.assertNotIn('the logger passed into', completed.stderr.lower())
        self.assertNotIn('different logger', completed.stderr.lower())


if __name__ == '__main__':
    if sys.argv[1:] == ['--probe']:
        _probe()
    else:
        unittest.main()
