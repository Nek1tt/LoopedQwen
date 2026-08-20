import numpy as np
import pytest
import torch

from looped_lm.data import BinaryTokenDataset


def test_sequential_batches_are_non_overlapping_and_resumable(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(64, dtype=np.uint16).tofile(path)
    dataset = BinaryTokenDataset(path, sequence_length=8)

    first = dataset.get_sequential_batch(batch_size=2, batch_index=0, device=torch.device("cpu"))
    second = dataset.get_sequential_batch(batch_size=2, batch_index=1, device=torch.device("cpu"))
    resumed = dataset.get_sequential_batch(batch_size=2, batch_index=1, device=torch.device("cpu"))

    torch.testing.assert_close(first.flatten(), torch.arange(16))
    torch.testing.assert_close(second.flatten(), torch.arange(16, 32))
    torch.testing.assert_close(second, resumed)


def test_sequential_batch_refuses_to_wrap(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(32, dtype=np.uint16).tofile(path)
    dataset = BinaryTokenDataset(path, sequence_length=8)
    with pytest.raises(IndexError):
        dataset.get_sequential_batch(batch_size=2, batch_index=2, device=torch.device("cpu"))

