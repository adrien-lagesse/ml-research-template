"""Tests for the MetricCollection primitive."""

import pytest
import torch

from core.metrics import ExplainedVariance
from core.metrics import MeanAbsoluteError
from core.metrics import MetricCollection


def test_metric_collection_merges_child_outputs(make_batch) -> None:
    """The collection's output carries every child's keys."""
    collection = MetricCollection([MeanAbsoluteError(), ExplainedVariance()])
    generator = torch.Generator().manual_seed(0)
    prediction = torch.randn(8, 2, generator=generator)
    target = torch.randn(8, 2, generator=generator)
    collection.update(make_batch(prediction, target))
    assert set(collection.compute().keys()) == {"mae", "explained_variance"}


def test_metric_collection_is_itself_a_metric(make_batch) -> None:
    """Collections nest, so harness code can treat them uniformly."""
    nested = MetricCollection([MetricCollection([MeanAbsoluteError()])])
    prediction = torch.tensor([[0.0], [0.0]])
    target = torch.tensor([[4.0], [6.0]])
    nested.update(make_batch(prediction, target))
    torch.testing.assert_close(nested.compute()["mae"], torch.tensor(5.0))


def test_metric_collection_rejects_duplicate_output_keys(make_batch) -> None:
    """Two children emitting the same key fail loudly instead of clobbering."""
    collection = MetricCollection([MeanAbsoluteError(), MeanAbsoluteError()])
    collection.update(make_batch(torch.zeros(1, 1), torch.ones(1, 1)))
    with pytest.raises(AssertionError):
        collection.compute()


def test_metric_collection_reset_clears_children(make_batch) -> None:
    """Resetting the collection resets the metrics it holds."""
    collection = MetricCollection([MeanAbsoluteError()])
    collection.update(make_batch(torch.zeros(2, 1), torch.full((2, 1), 4.0)))
    collection.reset()
    collection.update(make_batch(torch.zeros(1, 1), torch.ones(1, 1)))
    torch.testing.assert_close(collection.compute()["mae"], torch.tensor(1.0))
