"""The training loop, its eval pass, and the run context threaded through both."""

from core.train._context import TrainContext
from core.train._loop import EvalSplit
from core.train._loop import evaluate
from core.train._loop import fit

__all__ = ["EvalSplit", "TrainContext", "evaluate", "fit"]
