"""The streaming element-weighted mean composed by metrics and losses."""

from jaxtyping import Float
from torch import Tensor

from core._contracts import Scalar


class RunningMean:
    """A detached running sum and element count, reduced to a mean on demand.

    ``add`` folds a tensor's elements in; ``add_sum`` folds a precomputed
    sum in when the caller already reduced. Every element counts equally, so
    values from batches of different sizes are weighted by element count.
    The running sum lives on the device of the values it accumulates.
    """

    def __init__(self) -> None:
        """Start with an empty accumulator."""
        self.reset()

    def reset(self) -> None:
        """Clear the running sum and element count."""
        self.total: Scalar | None = None
        self.count: int = 0

    def add(self, values: Float[Tensor, "..."]) -> None:
        """Fold every element of ``values`` in, detaching first."""
        self.add_sum(values.detach().sum(), values.numel())

    def add_sum(self, total: Scalar, count: int) -> None:
        """Fold a precomputed sum over ``count`` elements in.

        Args:
            total: The 0-d sum of the elements, already detached.
            count: How many elements ``total`` sums over.
        """
        self.total = total if self.total is None else self.total + total
        self.count += count

    def mean(self) -> Scalar:
        """Reduce the accumulator to the element-weighted mean, moved to CPU.

        Divides the running sum by the total element count, so every element
        carries equal weight regardless of which batch it arrived in. Asserts
        that at least one element has been accumulated; an empty accumulator
        (never updated, or fed only zero-element tensors) fails instead of
        returning ``nan`` from ``0 / 0``.

        Returns:
            The 0-d mean over all elements folded in since the last ``reset``,
            on CPU.
        """
        assert self.total is not None, "mean() called before any add"
        assert self.count > 0, f"mean() over {self.count} accumulated elements"
        return (self.total / self.count).cpu()
