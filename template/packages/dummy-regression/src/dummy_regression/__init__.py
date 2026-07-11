"""A synthetic linear-regression task implementing the `core` contracts.

`SyntheticRegression` is a `core.Data` whose datasets yield raw
``(input, target)`` tuples drawn from one noisy linear map and whose collate
assembles them into `RegressionBatch`, the task's typed batch. `MLP` is a
`core.Model` that reads a batch's ``input`` and writes ``prediction``;
`MSELoss` is a `core.Loss` that scores ``prediction`` against ``target``.
"""

from dummy_regression._data import RegressionBatch
from dummy_regression._data import SyntheticRegression
from dummy_regression._loss import MSELoss
from dummy_regression._model import MLP

__all__ = [
    "MLP",
    "MSELoss",
    "RegressionBatch",
    "SyntheticRegression",
]
