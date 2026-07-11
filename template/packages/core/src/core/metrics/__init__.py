"""Streaming metrics: the base contract, a running mean, and regression metrics.

`Metric` accumulates statistics over batches and reduces them to named scalars;
`MetricCollection` drives several over one batch stream and merges their results.
`RunningMean` is the element-weighted accumulator that `MeanAbsoluteError` and
custom losses compose. `MeanAbsoluteError` and `ExplainedVariance` compare a
batch's prediction and target entries.
"""

from core.metrics._explained_variance import ExplainedVariance
from core.metrics._mean_absolute_error import MeanAbsoluteError
from core.metrics._metric import Metric
from core.metrics._metric import MetricCollection
from core.metrics._running_mean import RunningMean

__all__ = [
    "ExplainedVariance",
    "MeanAbsoluteError",
    "Metric",
    "MetricCollection",
    "RunningMean",
]
