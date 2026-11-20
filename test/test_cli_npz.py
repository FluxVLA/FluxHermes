"""NPZ output serialization uses real tensors, including BF16."""
import numpy as np
import torch

from fluxhermes.cli.common import save_npz


def test_save_bfloat16_outputs_as_lossless_float32(tmp_path):
    tensor = torch.tensor(
        [1.0078125, -3.140625, 0.0,
         float('inf'), float('nan')],
        dtype=torch.bfloat16,
        requires_grad=True)
    path = tmp_path / 'nested' / 'outputs.npz'
    save_npz(str(path), {'actions': tensor})
    with np.load(path, allow_pickle=False) as archive:
        assert archive.files == ['actions']
        assert archive['actions'].dtype == np.float32
        np.testing.assert_array_equal(archive['actions'],
                                      tensor.detach().float().numpy())
    assert tensor.dtype == torch.bfloat16
    assert tensor.requires_grad


def test_save_npz_preserves_other_dtypes_and_names(tmp_path):
    outputs = {
        'half': torch.tensor([1.5], dtype=torch.float16),
        'single': torch.tensor([1.5], requires_grad=True),
        'double': torch.tensor([1.5], dtype=torch.float64),
        'tokens': torch.tensor([2**60 + 1], dtype=torch.int64),
        'mask': torch.tensor([True, False]),
        'array': np.array([7], dtype=np.int32),
    }
    path = tmp_path / 'outputs.npz'
    save_npz(str(path), outputs)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(outputs)
        for name, value in outputs.items():
            expected = value.detach().numpy() if isinstance(value,
                                                            torch.Tensor) else value
            assert archive[name].dtype == expected.dtype
            np.testing.assert_array_equal(archive[name], expected)
