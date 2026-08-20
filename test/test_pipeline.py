import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from fluxhermes.cli.main import main
from fluxhermes.config import load_config
from fluxhermes.pipeline import CompilePipeline


class PipelinePlanTest(unittest.TestCase):

    def test_dry_run_does_not_import_gpu_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / 'configs/pi05/pi05_aloha_fp16.py')
        plan = CompilePipeline(config).plan_dict()
        self.assertEqual(plan['precision'], 'fp16')
        self.assertEqual(plan['stages'][0]['name'], 'prepare_model')
        self.assertEqual(plan['codebase'], 'fluxvla')
        self.assertEqual(plan['backend'], 'tensorrt')
        self.assertTrue(plan['stages'][2]['enabled'])

    def test_cli_compile_defaults_to_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'deploy.py'
            path.write_text(
                "deployment = dict(\n"
                " name='demo', work_dir='work',\n"
                " source=dict(type='python', factory='missing:factory'),\n"
                " inputs=[dict(name='x', dtype='float32', shape=(1, 2))],\n"
                " outputs=[dict(name='y', dtype='float32', shape=(1, 2))],\n"
                ")\n",
                encoding='utf-8',
            )
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['compile', str(path)])
            self.assertEqual(code, 0)
            self.assertIn('Dry run only', output.getvalue())


if __name__ == '__main__':
    unittest.main()
