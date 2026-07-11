"""Tests for the MLP model."""

import pytest
import tensordict
import torch

from dummy_regression import MLP


@pytest.fixture
def model() -> MLP:
    """A small seeded MLP matching the ``small_batch`` fixture's width."""
    return MLP(
        in_features=8,
        hidden_features=16,
        out_features=1,
        generator=torch.Generator().manual_seed(0),
    )


@pytest.fixture
def small_batch() -> tensordict.TensorDict:
    """A fixed 4-sample batch carrying an 8-wide ``input``."""
    generator = torch.Generator().manual_seed(0)
    return tensordict.TensorDict(
        {"input": torch.randn(4, 8, generator=generator)}, batch_size=[4]
    )


def test_mlp_writes_prediction_with_output_shape(
    model: MLP, small_batch: tensordict.TensorDict
) -> None:
    """`MLP` adds a `prediction` entry shaped `(batch, out_features)`."""
    out = model(small_batch)
    assert out["prediction"].shape == (4, 1)


def test_mlp_returns_the_same_batch(
    model: MLP, small_batch: tensordict.TensorDict
) -> None:
    """The model enriches and returns the batch it was given, not a new one."""
    assert model(small_batch) is small_batch


def test_mlp_init_is_reproducible_from_the_generator() -> None:
    """Two models built with identically seeded generators share every weight."""
    models = [
        MLP(
            in_features=8,
            hidden_features=16,
            out_features=1,
            generator=torch.Generator().manual_seed(0),
        )
        for _ in range(2)
    ]
    for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
        torch.testing.assert_close(left, right)
