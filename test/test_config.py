import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fluxhermes.config import load_config
from fluxhermes.errors import ConfigurationError


class ConfigTest(unittest.TestCase):

    def test_loads_and_expands_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'deploy.py'
            path.write_text(
                "deployment = dict(\n"
                " name='demo', work_dir='$DEMO_ROOT/work',\n"
                " source=dict(type='python', factory='a:b'),\n"
                " inputs=[dict(name='x', dtype='float32', shape=(1, 2))],\n"
                " outputs=[dict(name='y', dtype='float32', shape=(1, 2))],\n"
                ")\n",
                encoding='utf-8',
            )
            with mock.patch.dict(os.environ, {'DEMO_ROOT': directory}):
                config = load_config(path)
            self.assertEqual(config.name, 'demo')
            self.assertEqual(config.work_dir, Path(directory) / 'work')

    def test_requires_deployment_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.py'
            path.write_text('value = 1\n', encoding='utf-8')
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_repository_templates_load(self):
        root = Path(__file__).resolve().parents[1]
        fp16 = load_config(root / 'configs/pi05/pi05_aloha_fp16.py')
        int8 = load_config(root / 'configs/pi05/pi05_aloha_int8.py')
        gr00t = load_config(root / 'configs/gr00t/gr00t_aloha_fp16.py')
        self.assertEqual(fp16.precision, 'fp16')
        self.assertEqual(int8.precision, 'int8')
        self.assertEqual(gr00t.codebase['type'], 'python')

    def test_loads_mmdeploy_style_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'deploy.py'
            path.write_text(
                "deployment = dict(\n"
                " name='demo', work_dir='work',\n"
                " codebase_config=dict(type='python', task='custom', "
                "source=dict(type='python', factory='a:b')),\n"
                " backend_config=dict(type='tensorrt', precision='fp16'),\n"
                " ir_config=dict(type='onnx', opset=18),\n"
                " inputs=[dict(name='x', dtype='float32', shape=(1, 2))],\n"
                " outputs=[dict(name='y', dtype='float32', shape=(1, 2))],\n"
                ")\n",
                encoding='utf-8',
            )
            config = load_config(path)
            self.assertEqual(config.codebase['task'], 'custom')
            self.assertEqual(config.backend['type'], 'tensorrt')
            self.assertEqual(config.ir['type'], 'onnx')


if __name__ == '__main__':
    unittest.main()
