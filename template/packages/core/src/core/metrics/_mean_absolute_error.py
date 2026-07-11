"""Streaming mean absolute error between a batch's prediction and target entries."""

import tensordict

from core._contracts import Scalar
from core.metrics._metric import Metric
from core.metrics._running_mean import RunningMean


class MeanAbsoluteError(Metric):
    """Streaming mean absolute error between two batch entries.

    Every element counts equally, so batches of different sizes are weighted
    by element count. ``compute`` emits the result under ``"mae"``.
    """

    def __init__(
        self, prediction_key: str = "prediction", target_key: str = "target"
    ) -> None:
        """Set up an MAE between the entries named by the two keys."""
        self.prediction_key = prediction_key
        self.target_key = target_key
        self._running = RunningMean()

    def reset(self) -> None:
        """Clear the accumulated absolute errors."""
        self._running.reset()

    def update(self, batch: tensordict.TensorDictBase) -> None:
        """Add one batch's absolute errors to the running sum.

        Values are detached before summing, and the running sum lives on
        their device.

        Args:
            batch: Must carry tensors of matching shape under
                ``prediction_key`` and ``target_key``.
        """
        prediction = batch.get(self.prediction_key)
        target = batch.get(self.target_key)
        assert prediction.shape == target.shape, (
            f"prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}"
        )
        self._running.add((prediction - target).abs())

    def compute(self) -> dict[str, Scalar]:
        """Return the mean absolute error since the last reset, on CPU.

        The mean is over every element seen. Asserts that at least one
        ``update`` has happened.
        """
        return {"mae": self._running.mean()}
