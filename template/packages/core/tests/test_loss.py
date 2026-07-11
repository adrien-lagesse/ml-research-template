"""Tests for the Loss base class."""

import tensordict
import torch

from core import Loss
from core.metrics import MetricCollection


class _Zero(Loss):
    """Loss that scores every batch zero and counts its updates."""

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.n_updates = 0

    def update_and_loss(self, batch: tensordict.TensorDictBase) -> torch.Tensor:
        assert batch.batch_dims == 1
        self.n_updates += 1
        return torch.zeros(())

    def compute(self) -> dict[str, torch.Tensor]:
        return {"loss": torch.tensor(float(self.n_updates))}


def _batch() -> tensordict.TensorDict:
    """A minimal one-key batch."""
    return tensordict.TensorDict({"x": torch.zeros(2)}, batch_size=[2])


def test_loss_is_an_nn_module() -> None:
    """A Loss subclass carries the nn.Module machinery."""
    assert isinstance(_Zero(), torch.nn.Module)


def test_update_and_loss_returns_a_scalar() -> None:
    """The training entry point yields a 0-d tensor to backpropagate."""
    scalar = _Zero().update_and_loss(_batch())
    assert scalar.shape == ()
    assert scalar.item() == 0.0


def test_update_delegates_to_update_and_loss() -> None:
    """The Metric-style `update` accumulates exactly like `update_and_loss`."""
    loss = _Zero()
    loss.update(_batch())
    loss.update_and_loss(_batch())
    torch.testing.assert_close(loss.compute()["loss"], torch.tensor(2.0))


def test_loss_slots_into_a_metric_collection() -> None:
    """A Loss is a Metric, so collections drive it like any other."""
    collection = MetricCollection([_Zero()])
    collection.update(_batch())
    torch.testing.assert_close(collection.compute()["loss"], torch.tensor(1.0))
    collection.reset()
    collection.update(_batch())
    torch.testing.assert_close(collection.compute()["loss"], torch.tensor(1.0))
