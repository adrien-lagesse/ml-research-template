"""Tests for the streaming regression metrics."""

import einops
import pytest
import torch

from core.metrics import ExplainedVariance
from core.metrics import MeanAbsoluteError
from core.metrics import RunningMean


def _random_pair(n: int, dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A random prediction/target pair of shape (n, dim)."""
    generator = torch.Generator().manual_seed(seed)
    prediction = torch.randn(n, dim, generator=generator)
    target = torch.randn(n, dim, generator=generator)
    return prediction, target


def test_running_mean_rejects_an_empty_accumulator() -> None:
    """Folding in only zero-element tensors asserts instead of yielding nan."""
    running = RunningMean()
    running.add(torch.empty(0))
    with pytest.raises(AssertionError, match="0 accumulated elements"):
        running.mean()


def test_mae_is_zero_for_a_perfect_prediction(make_batch) -> None:
    """Identical prediction and target give an MAE of exactly 0."""
    _, target = _random_pair(16, 3, seed=0)
    metric = MeanAbsoluteError()
    metric.update(make_batch(target.clone(), target))
    torch.testing.assert_close(metric.compute()["mae"], torch.tensor(0.0))


def test_mae_matches_the_direct_mean(make_batch) -> None:
    """The streamed MAE equals the plain mean of absolute errors."""
    prediction, target = _random_pair(32, 4, seed=1)
    metric = MeanAbsoluteError()
    metric.update(make_batch(prediction, target))
    expected = (prediction - target).abs().mean()
    torch.testing.assert_close(metric.compute()["mae"], expected)


def test_mae_is_independent_of_batching(make_batch) -> None:
    """Streaming over chunks equals one update over the whole set."""
    prediction, target = _random_pair(48, 2, seed=2)
    streamed = MeanAbsoluteError()
    for chunk in range(3):
        rows = slice(16 * chunk, 16 * (chunk + 1))
        streamed.update(make_batch(prediction[rows], target[rows]))
    whole = MeanAbsoluteError()
    whole.update(make_batch(prediction, target))
    torch.testing.assert_close(streamed.compute()["mae"], whole.compute()["mae"])


def test_explained_variance_is_one_for_a_perfect_prediction(make_batch) -> None:
    """Zero residuals explain all of the target variance."""
    _, target = _random_pair(16, 3, seed=3)
    metric = ExplainedVariance()
    metric.update(make_batch(target.clone(), target))
    torch.testing.assert_close(
        metric.compute()["explained_variance"], torch.tensor(1.0)
    )


def test_explained_variance_is_zero_for_the_mean_prediction(make_batch) -> None:
    """Predicting each output's mean explains none of the variance."""
    _, target = _random_pair(32, 3, seed=4)
    target_mean = einops.reduce(target, "n dim -> dim", "mean")
    prediction = einops.repeat(target_mean, "dim -> n dim", n=len(target))
    metric = ExplainedVariance()
    metric.update(make_batch(prediction, target))
    torch.testing.assert_close(
        metric.compute()["explained_variance"], torch.tensor(0.0)
    )


def test_explained_variance_matches_the_direct_formula(make_batch) -> None:
    """The streamed score equals 1 - Var(error)/Var(target), output-averaged."""
    prediction, target = _random_pair(64, 5, seed=5)
    metric = ExplainedVariance()
    metric.update(make_batch(prediction, target))
    error = target - prediction
    expected = (
        1.0 - error.var(dim=0, correction=0) / target.var(dim=0, correction=0)
    ).mean()
    torch.testing.assert_close(metric.compute()["explained_variance"], expected)


def test_explained_variance_is_independent_of_batching(make_batch) -> None:
    """Streaming over chunks equals one update over the whole set."""
    prediction, target = _random_pair(48, 2, seed=6)
    streamed = ExplainedVariance()
    for chunk in range(3):
        rows = slice(16 * chunk, 16 * (chunk + 1))
        streamed.update(make_batch(prediction[rows], target[rows]))
    whole = ExplainedVariance()
    whole.update(make_batch(prediction, target))
    torch.testing.assert_close(
        streamed.compute()["explained_variance"],
        whole.compute()["explained_variance"],
    )


def test_explained_variance_averages_uniformly_over_outputs(make_batch) -> None:
    """One perfect and one mean-predicted output average to one half."""
    _, target = _random_pair(32, 2, seed=7)
    prediction = target.clone()
    prediction[:, 1] = target[:, 1].mean()
    metric = ExplainedVariance()
    metric.update(make_batch(prediction, target))
    torch.testing.assert_close(
        metric.compute()["explained_variance"], torch.tensor(0.5)
    )
