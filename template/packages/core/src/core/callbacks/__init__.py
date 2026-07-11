"""Callbacks hooked into `fit`'s loop, and the three shipped with the package.

`Callback` observes training through four no-op hooks, each receiving the run's
`core.train.TrainContext`. `ModelSummary` records the model's parameter counts
once at the start of training, computed by `model_summary`. `ConfigSummary`
records the run's resolved config the same way. `LRMonitor` logs the optimizer's
learning rate on a step cadence. `Checkpoint` saves the training state on a step
or wall-clock cadence, or when a monitored eval metric improves.
"""

from core.callbacks._callback import Callback
from core.callbacks._checkpoint import Checkpoint
from core.callbacks._config_summary import ConfigSummary
from core.callbacks._lr_monitor import LRMonitor
from core.callbacks._model_summary import ModelSummary
from core.callbacks._model_summary import model_summary

__all__ = [
    "Callback",
    "Checkpoint",
    "ConfigSummary",
    "LRMonitor",
    "ModelSummary",
    "model_summary",
]
