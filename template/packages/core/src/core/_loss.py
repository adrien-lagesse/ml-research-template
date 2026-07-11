"""The loss contract: a metric whose batches also yield a scalar to backpropagate."""

from abc import ABC
from abc import abstractmethod

import tensordict
import torch

from core._contracts import Scalar
from core.metrics import Metric


class Loss(Metric, torch.nn.Module, ABC):
    """A metric that additionally produces the scalar to backpropagate.

    ``update_and_loss`` plays the role of ``Metric.update``: it folds one
    batch into the accumulators and returns that batch's 0-d loss for
    ``.backward()``. ``update`` delegates to it and discards the scalar, so a
    loss drops into any ``MetricCollection``; ``compute`` then reports the
    accumulated loss terms like any other metric. Being a
    ``torch.nn.Module``, a loss may carry learnable parameters or buffers.
    """

    @abstractmethod
    def update_and_loss(self, batch: tensordict.TensorDictBase) -> Scalar:
        """Fold one batch into the accumulators and return its loss.

        Args:
            batch: Named tensors the loss reads, typically model outputs.

        Returns:
            The 0-d loss for this batch, ready for ``.backward()``.
        """

    def update(self, batch: tensordict.TensorDictBase) -> None:
        """Fold one batch into the accumulators, discarding the loss scalar."""
        self.update_and_loss(batch)
