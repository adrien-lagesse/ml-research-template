"""Mean-squared-error loss over ``prediction`` and ``target`` batch keys."""

import einops
import tensordict

from core import Loss
from core import Scalar
from core.metrics import RunningMean


class MSELoss(Loss):
    """Mean squared error between a batch's ``prediction`` and ``target``.

    ``update_and_loss`` returns one batch's element-mean squared error and
    folds the batch into the accumulator; ``compute`` reports the
    element-weighted mean over every batch seen since the last reset, under
    the ``"loss"`` key.
    """

    def __init__(self) -> None:
        """Set up the module with an empty accumulator."""
        super().__init__()
        self._running = RunningMean()

    def reset(self) -> None:
        """Clear the accumulated squared errors."""
        self._running.reset()

    def update_and_loss(self, batch: tensordict.TensorDictBase) -> Scalar:
        """Score one batch and fold its squared errors into the accumulator.

        The accumulated sum is detached and lives on the batch's device; the
        returned scalar keeps its graph.

        Args:
            batch: Must hold ``prediction`` and ``target``, two float tensors
                of matching shape ``(batch, dim)``.

        Returns:
            The batch's squared error averaged over every element, 0-d,
            ready for ``.backward()``.
        """
        prediction = batch.get("prediction")
        target = batch.get("target")
        assert prediction.shape == target.shape, (
            f"prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}"
        )
        squared_error = (prediction - target) ** 2
        batch_loss = einops.reduce(squared_error, "batch dim ->", "mean")
        self._running.add_sum(
            batch_loss.detach() * squared_error.numel(), squared_error.numel()
        )
        return batch_loss

    def compute(self) -> dict[str, Scalar]:
        """Return the element-weighted MSE since the last reset, on CPU.

        Asserts that at least one ``update_and_loss`` has happened.
        """
        return {"loss": self._running.mean()}
