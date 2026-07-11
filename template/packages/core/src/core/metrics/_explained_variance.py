"""Streaming explained variance over a batch's prediction and target entries."""

import einops
from jaxtyping import Float
import tensordict
import torch
from torch import Tensor

from core._contracts import Scalar
from core.metrics._metric import Metric


class ExplainedVariance(Metric):
    """Streaming explained variance, uniformly averaged over target outputs.

    Per output ``d``, explained variance is
    ``1 - Var(target_d - prediction_d) / Var(target_d)`` with population
    variances over every sample seen; the emitted value, under
    ``"explained_variance"``, is the mean across outputs. A perfect predictor
    scores 1 and predicting the target mean scores 0. The per-output mean and
    sum of squared deviations are merged batch by batch with Chan's parallel
    algorithm, so the score is stable for large-magnitude targets and
    independent of how the samples were batched.
    """

    def __init__(
        self, prediction_key: str = "prediction", target_key: str = "target"
    ) -> None:
        """Set up an explained variance between the entries named by the keys."""
        self.prediction_key = prediction_key
        self.target_key = target_key
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated mean, squared-deviation sum, and sample count."""
        self.mean: Float[Tensor, "2 dim"] | None = None
        self.sum_squares: Float[Tensor, "2 dim"] | None = None
        self.count: int = 0

    def update(self, batch: tensordict.TensorDictBase) -> None:
        """Merge one batch's per-output moments into the accumulators.

        Row 0 of each accumulator tracks the residual ``target - prediction``
        and row 1 tracks the target. Values are detached first, and the
        accumulators live on their device.

        Args:
            batch: Must carry float tensors of matching shape
                ``(batch, dim)`` under ``prediction_key`` and ``target_key``.
        """
        prediction = batch.get(self.prediction_key).detach()
        target = batch.get(self.target_key).detach()
        assert prediction.ndim == 2, f"expected 2-D, got shape {prediction.shape}"
        assert prediction.shape == target.shape, (
            f"prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}"
        )
        values: Float[Tensor, "2 batch dim"] = torch.stack(
            [target - prediction, target]
        )
        batch_count = target.shape[0]
        batch_mean = einops.reduce(values, "row batch dim -> row dim", "mean")
        deviations = values - einops.rearrange(batch_mean, "row dim -> row 1 dim")
        batch_sum_squares = einops.reduce(
            deviations**2, "row batch dim -> row dim", "sum"
        )
        if self.mean is None or self.sum_squares is None:
            self.mean = batch_mean
            self.sum_squares = batch_sum_squares
            self.count = batch_count
            return
        # Chan's parallel algorithm: merge the running (count, mean, sum of
        # squared deviations) with this batch's, never forming E[x^2]-E[x]^2.
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total_count)
        self.sum_squares = (
            self.sum_squares
            + batch_sum_squares
            + delta**2 * (self.count * batch_count / total_count)
        )
        self.count = total_count

    def compute(self) -> dict[str, Scalar]:
        """Return the output-averaged explained variance since the last reset, on CPU.

        Asserts that at least one ``update`` has happened and that every
        target output has nonzero variance, without which the score is
        undefined.
        """
        assert self.sum_squares is not None, "compute() called before any update"
        error_var, target_var = self.sum_squares / self.count
        assert (target_var > 0).all(), "explained variance needs varying targets"
        explained = 1.0 - error_var / target_var
        return {"explained_variance": explained.mean().cpu()}
