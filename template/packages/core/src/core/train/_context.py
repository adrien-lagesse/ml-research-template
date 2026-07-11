"""The run context threaded through the training loop and every callback hook."""

from collections.abc import Mapping
import dataclasses

import torch
import torch.optim
import torch.optim.lr_scheduler

from core._contracts import Model
from core._contracts import Scalar
from core._loss import Loss
from core.logger import Logger


@dataclasses.dataclass(frozen=True)
class TrainContext:
    """A read-only view of the run passed to ``fit`` and every callback hook.

    Built once at the entry point and handed to ``fit``, which threads it to
    each hook with the evolving fields (``current_step``, ``time_since_start``)
    refreshed to the moment of the call. A callback reads the run's objects and
    its progress through them without reaching into ``fit``.

    Attributes:
        model: The model being trained.
        loss: The training loss.
        optimizer: The optimizer.
        scheduler: The learning-rate scheduler.
        logger: The run's logger.
        device: The device batches are moved to.
        log_every_steps: The train-loss logging cadence, in global steps.
        total_steps: The run's length in global steps.
        seed: The seed the run's generators were built from.
        is_resumed: Whether the run continues from a saved checkpoint.
        current_step: The number of optimizer steps taken so far.
        time_since_start: Seconds of wall-clock time since the first step.
        last_loss: The most recent training step's loss, or ``None`` before the
            first step.
        last_metrics: The latest logged value of each eval metric, keyed by
            ``split/metric``; empty before the first eval pass.
        _resolved_config: The run's fully resolved config, carried as opaque
            provenance for the ``ConfigSummary`` callback to persist and for
            checkpointing. Private: a callback may record it wholesale, but none
            may branch on its contents.
    """

    model: Model
    loss: Loss
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    logger: Logger
    device: torch.device
    log_every_steps: int
    total_steps: int
    seed: int = 0
    is_resumed: bool = False
    current_step: int = 0
    time_since_start: float = 0.0
    last_loss: Scalar | None = None
    last_metrics: dict[str, Scalar] = dataclasses.field(default_factory=dict)
    _resolved_config: Mapping[str, object] = dataclasses.field(default_factory=dict)
