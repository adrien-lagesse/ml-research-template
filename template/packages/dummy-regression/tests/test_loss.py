"""Tests for the MSE loss."""

import pytest
import tensordict
import torch

from dummy_regression import MSELoss


def test_mse_loss_is_zero_for_a_perfect_prediction(make_batch) -> None:
    """A prediction equal to the target scores zero, scalar and term alike."""
    target = torch.tensor([[1.0], [2.0]])
    loss = MSELoss()
    scalar = loss.update_and_loss(make_batch(target.clone(), target))
    torch.testing.assert_close(scalar, torch.tensor(0.0))
    torch.testing.assert_close(loss.compute()["loss"], torch.tensor(0.0))


def test_mse_scalar_matches_manual_mean_squared_error(make_batch) -> None:
    """The returned scalar is the mean squared error over all elements."""
    prediction = torch.tensor([[0.0], [0.0]])
    target = torch.tensor([[1.0], [3.0]])
    scalar = MSELoss().update_and_loss(make_batch(prediction, target))
    torch.testing.assert_close(scalar, torch.tensor(5.0))


def test_mse_compute_weights_batches_by_element_count(make_batch) -> None:
    """The accumulated value is the mean over every element, not over batches."""
    loss = MSELoss()
    loss.update_and_loss(make_batch(torch.zeros(2, 1), torch.tensor([[1.0], [3.0]])))
    loss.update_and_loss(make_batch(torch.zeros(1, 1), torch.tensor([[2.0]])))
    torch.testing.assert_close(loss.compute()["loss"], torch.tensor(14.0 / 3.0))


def test_mse_reset_clears_the_accumulation(make_batch) -> None:
    """After a reset, only later batches count."""
    loss = MSELoss()
    loss.update_and_loss(make_batch(torch.zeros(2, 1), torch.tensor([[1.0], [3.0]])))
    loss.reset()
    loss.update_and_loss(make_batch(torch.zeros(1, 1), torch.tensor([[2.0]])))
    torch.testing.assert_close(loss.compute()["loss"], torch.tensor(4.0))


def test_mse_loss_rejects_mismatched_shapes() -> None:
    """Shapes that would silently broadcast fail loudly instead."""
    batch = tensordict.TensorDict(
        {"prediction": torch.zeros(2, 1), "target": torch.zeros(2, 3)}, batch_size=[2]
    )
    with pytest.raises(AssertionError, match="prediction"):
        MSELoss().update_and_loss(batch)
