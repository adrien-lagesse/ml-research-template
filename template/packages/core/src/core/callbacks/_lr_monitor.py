"""A callback that logs the optimizer's learning rate on a step cadence."""

import torch

from core._contracts import Scalar
from core.callbacks._callback import Callback
from core.train import TrainContext


class LRMonitor(Callback):
    """Logs the optimizer's learning rate every fixed number of steps.

    The rate is read from the optimizer's parameter groups, so it reflects what
    the last scheduler step actually applied. With one parameter group the value
    is logged under ``lr``; with several, each is logged under ``<prefix>/group{i}``.
    Logging on the train cadence keeps the rate on the same step timeline as the
    training loss.
    """

    def __init__(self, *, every_steps: int, prefix: str = "lr") -> None:
        """Set the logging cadence and the key the rate is logged under.

        Args:
            every_steps: Log the rate every this many global steps; pass the
                train logging cadence to align it with the loss.
            prefix: Key prefix used only when there is more than one parameter
                group, giving ``<prefix>/group0``, ``<prefix>/group1``, ….

        Raises:
            ValueError: If ``every_steps`` is not positive.
        """
        if every_steps <= 0:
            raise ValueError(f"every_steps must be positive: {every_steps}")
        self._every_steps = every_steps
        self._prefix = prefix

    def on_train_start(self, ctx: TrainContext) -> None:
        """Log the initial rate, before the first step."""
        self._log_learning_rates(ctx)

    def on_step_end(self, ctx: TrainContext) -> None:
        """Log the rate every ``every_steps`` steps."""
        if ctx.current_step % self._every_steps == 0:
            self._log_learning_rates(ctx)

    def _log_learning_rates(self, ctx: TrainContext) -> None:
        """Log the current learning rate of every parameter group.

        Each rate is a configured float, so it is wrapped as a double-precision
        0-d tensor to log the exact value rather than its float32 rounding.
        """
        groups = ctx.optimizer.param_groups
        if len(groups) == 1:
            rate = torch.tensor(groups[0]["lr"], dtype=torch.float64)
            ctx.logger.log_dict({"lr": rate}, step=ctx.current_step)
            return
        values: dict[str, Scalar] = {
            f"group{index}": torch.tensor(group["lr"], dtype=torch.float64)
            for index, group in enumerate(groups)
        }
        ctx.logger.log_dict(values, step=ctx.current_step, prefix=self._prefix)
