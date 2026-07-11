"""A callback that saves the training state on a cadence or a metric improvement."""

import time

from core._checkpoint import CheckpointState
from core._contracts import Scalar
from core.callbacks._callback import Callback
from core.train import TrainContext


class Checkpoint(Callback):
    """Saves the full training state on a cadence or when a metric improves.

    A checkpoint is one of two kinds. A *cadence* checkpoint sets ``every_steps``
    and/or ``every_seconds`` and saves from ``on_step_end``, so its timing is
    exact and independent of evaluation, the fit for a periodic "latest" snapshot
    to resume from. A *metric* checkpoint sets ``monitor`` and saves from
    ``on_eval_end`` whenever that eval metric improves, the fit for a "best"
    snapshot. The two kinds are mutually exclusive: a periodic save would
    overwrite the best, so each purpose gets its own instance and its own
    ``name``. Every save hands the model, loss, optimizer, and scheduler state to
    the logger under ``name``, overwriting the previous one.
    """

    def __init__(
        self,
        *,
        every_steps: int | None = None,
        every_seconds: float | None = None,
        monitor: str | None = None,
        mode: str = "min",
        name: str = "last",
    ) -> None:
        """Configure the trigger and the name the checkpoint is saved under.

        Set either a cadence (``every_steps`` and/or ``every_seconds``) or a
        ``monitor``, not both.

        Args:
            every_steps: Save every this many global steps.
            every_seconds: Save every this much wall-clock time, measured from
                the start of training and each time-triggered save.
            monitor: Prefixed eval-metric key, e.g. ``"validation/loss"``; save
                when its value improves. Only the eval split whose metrics carry
                the key is considered.
            mode: ``"min"`` to treat lower ``monitor`` values as better,
                ``"max"`` for higher.
            name: Filename stem the checkpoint is stored under; each save
                overwrites the previous one.

        Raises:
            ValueError: If no trigger is set, a cadence is combined with
                ``monitor``, a cadence is not positive, or ``mode`` is not
                ``"min"`` or ``"max"``.
        """
        has_cadence = every_steps is not None or every_seconds is not None
        if not has_cadence and monitor is None:
            raise ValueError("set a cadence (every_steps/every_seconds) or monitor")
        if has_cadence and monitor is not None:
            raise ValueError("set a cadence or monitor, not both; use two callbacks")
        if every_steps is not None and every_steps <= 0:
            raise ValueError(f"every_steps must be positive: {every_steps}")
        if every_seconds is not None and every_seconds <= 0:
            raise ValueError(f"every_seconds must be positive: {every_seconds}")
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max': {mode!r}")
        self._every_steps = every_steps
        self._every_seconds = every_seconds
        self._monitor = monitor
        self._mode = mode
        self._name = name
        self._last_time: float | None = None
        self._best: float | None = None

    def on_train_start(self, ctx: TrainContext) -> None:  # noqa: ARG002
        """Anchor the wall-clock cadence at the start of training."""
        if self._every_seconds is not None:
            self._last_time = time.monotonic()

    def on_step_end(self, ctx: TrainContext) -> None:
        """Save when a step or wall-clock cadence is reached."""
        fired = (
            self._every_steps is not None and ctx.current_step % self._every_steps == 0
        )
        if self._every_seconds is not None:
            assert self._last_time is not None
            now = time.monotonic()
            if now - self._last_time >= self._every_seconds:
                self._last_time = now
                fired = True
        if fired:
            self._save(ctx)

    def on_eval_end(
        self, ctx: TrainContext, *, name: str, metrics: dict[str, Scalar]
    ) -> None:
        """Save when the monitored eval metric improves."""
        if self._monitor is None:
            return
        prefixed = {f"{name}/{key}": value for key, value in metrics.items()}
        if self._monitor not in prefixed:
            return
        value = float(prefixed[self._monitor].item())
        if self._best is None or self._improved(value):
            self._best = value
            self._save(ctx)

    def _improved(self, value: float) -> bool:
        """Whether ``value`` beats the best seen so far, per ``mode``."""
        assert self._best is not None
        return value < self._best if self._mode == "min" else value > self._best

    def _save(self, ctx: TrainContext) -> None:
        """Hand the current training state to the logger under ``name``."""
        state = CheckpointState(
            model=ctx.model.state_dict(),
            loss=ctx.loss.state_dict(),
            optimizer=ctx.optimizer.state_dict(),
            scheduler=ctx.scheduler.state_dict(),
            global_step=ctx.current_step,
        )
        ctx.logger.save_checkpoint(state, name=self._name)
