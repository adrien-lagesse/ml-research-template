"""The metric contract and the collection that drives several at once."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence

import tensordict

from core._contracts import Scalar


class Metric(ABC):
    """Base class for metrics accumulated batch by batch.

    The lifecycle is ``reset`` to clear state, ``update`` once per batch, and
    ``compute`` to reduce the accumulated state to named scalars. A subclass
    whose ``reset`` initializes state must end its own ``__init__`` with
    ``self.reset()``.
    """

    @abstractmethod
    def reset(self) -> None:
        """Create or restore the accumulator state to its empty value."""

    @abstractmethod
    def update(self, batch: tensordict.TensorDictBase) -> None:
        """Fold one batch into the accumulator.

        Args:
            batch: Named tensors carrying the values this metric reads.
        """

    @abstractmethod
    def compute(self) -> dict[str, Scalar]:
        """Reduce the accumulated state to named 0-d tensors.

        Returns:
            One entry per value this metric produces, keyed by name.
        """


class MetricCollection(Metric):
    """Several metrics driven over the same stream of batches.

    ``update`` fans out to every child; ``compute`` merges the children's
    outputs into one dict, so the children must produce distinct keys.
    Collections nest: a collection is itself a ``Metric``.
    """

    def __init__(self, metrics: Sequence[Metric]) -> None:
        """Wrap the given metrics.

        The sequence is copied, so later mutation of the argument doesn't
        reach the collection.
        """
        self._metrics = list(metrics)

    def reset(self) -> None:
        """Reset every metric in the collection."""
        for metric in self._metrics:
            metric.reset()

    def update(self, batch: tensordict.TensorDictBase) -> None:
        """Feed the same batch to every metric in the collection."""
        for metric in self._metrics:
            metric.update(batch)

    def compute(self) -> dict[str, Scalar]:
        """Merge every child's computed values into one dict.

        Asserts that no two children emit the same key.
        """
        out: dict[str, Scalar] = {}
        for metric in self._metrics:
            computed = metric.compute()
            colliding = computed.keys() & out.keys()
            assert not colliding, f"duplicate metric keys: {colliding}"
            out.update(computed)
        return out
