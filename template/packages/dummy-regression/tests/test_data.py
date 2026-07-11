"""Tests for the synthetic regression dataset."""

import torch

from dummy_regression import RegressionBatch
from dummy_regression import SyntheticRegression


def _data(n_train: int = 32) -> SyntheticRegression:
    """A small three-split dataset for shape and presence checks."""
    return SyntheticRegression(
        n_features=8,
        n_targets=1,
        n_train=n_train,
        n_val=16,
        n_test=16,
        noise_std=0.1,
        dataset_seed=0,
        batch_size=8,
        dataloader_seed=0,
    )


def test_train_dataloader_yields_typed_batches() -> None:
    """Each training batch is a RegressionBatch with the configured shapes."""
    batch = next(iter(_data().train_dataloader()))
    assert isinstance(batch, RegressionBatch)
    assert batch.input.shape == (8, 8)
    assert batch.target.shape == (8, 1)
    assert batch.get("prediction", None) is None


def test_every_split_returns_a_dataloader() -> None:
    """Each split exposes an iterable loader under its documented name."""
    data = _data()
    assert next(iter(data.train_dataloader())) is not None
    assert set(data.eval_dataloaders()) == {"validation", "test"}


def test_dataset_is_reproducible_from_the_seed() -> None:
    """Two instances with the same seed produce identical first batches."""
    first = next(iter(_data().train_dataloader()))
    second = next(iter(_data().train_dataloader()))
    torch.testing.assert_close(first.input, second.input)
    torch.testing.assert_close(first.target, second.target)


def test_eval_splits_do_not_depend_on_the_train_size() -> None:
    """Changing n_train leaves the validation and test samples untouched."""
    small = _data(n_train=8).eval_dataloaders()
    large = _data(n_train=64).eval_dataloaders()
    for name in small:
        first = next(iter(small[name]))
        second = next(iter(large[name]))
        torch.testing.assert_close(first.input, second.input)
        torch.testing.assert_close(first.target, second.target)
