import tempfile
import unittest
from pathlib import Path

from fluxhermes.post_training import (PostTrainingPipeline,
                                      _checkpoint_state_for_model,
                                      _representative_config,
                                      _resolve_checkpoint, _tensor_specs)


class _Tensor:

    def __init__(self, shape, dtype='torch.float32'):
        self.shape = shape
        self.dtype = dtype


class _Model:

    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class PostTrainingContractTest(unittest.TestCase):

    def test_requires_exact_training_checkpoint(self):
        expected = {'weight': _Tensor((2, 3))}

        state = _checkpoint_state_for_model(_Model(expected), expected)

        self.assertEqual(state, expected)

    def test_rejects_checkpoint_shape_mismatch(self):
        model = _Model({'weight': _Tensor((2, 3))})
        checkpoint = {'weight': _Tensor((3, 2))}

        with self.assertRaisesRegex(RuntimeError, 'shape_mismatch'):
            _checkpoint_state_for_model(model, checkpoint)

    def test_does_not_silently_replace_pt_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath('model.safetensors').touch()

            with self.assertRaises(FileNotFoundError):
                _resolve_checkpoint(root / 'model.pt')

    def test_builds_policy_tensor_specs(self):
        specs = _tensor_specs({
            'images': _Tensor((1, 6, 224, 224)),
            'tokens': _Tensor((1, 200), 'torch.int64'),
        })

        self.assertEqual(specs[0].dtype, 'float32')
        self.assertEqual(specs[0].shape, (1, 6, 224, 224))
        self.assertEqual(specs[1].dtype, 'int64')

    def test_rejects_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'train.py'
            checkpoint = root / 'model.safetensors'
            output = root / 'tensorrt'
            config.touch()
            checkpoint.touch()
            output.mkdir()
            output.joinpath('existing.plan').touch()

            pipeline = PostTrainingPipeline(config, checkpoint, output)

            with self.assertRaisesRegex(FileExistsError, 'must be empty'):
                pipeline.execute()

    def test_prefers_eval_dataset_for_representative_inputs(self):
        config = {
            'train_dataloader': {
                'dataset': {'name': 'train'}
            },
            'eval': {
                'dataset': {'name': 'eval'}
            },
        }

        selected = _representative_config(config)

        self.assertEqual(selected['name'], 'eval')


if __name__ == '__main__':
    unittest.main()
