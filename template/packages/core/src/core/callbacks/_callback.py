"""The callback interface: the four hooks and the context they receive."""

from core._contracts import Scalar
from core.train import TrainContext


class Callback:
    """Observer hooked into ``fit``'s loop; every hook is a no-op by default.

    Subclass and override the hooks you need. The base leaves all four empty so
    a callback stays focused on its concern, the same way ``Logger`` leaves
    ``close`` a no-op for backends that hold nothing. It is a concrete base, not
    an abstract one: a bare ``Callback`` is a valid do-nothing observer.
    """

    # No-op defaults: a callback overrides only the hooks it uses.
    def on_train_start(self, ctx: TrainContext) -> None:
        """Called once before the first step, after ``model.train()``."""

    def on_step_end(self, ctx: TrainContext) -> None:
        """Called after each optimizer step, once ``current_step`` is incremented."""

    def on_eval_end(
        self, ctx: TrainContext, *, name: str, metrics: dict[str, Scalar]
    ) -> None:
        """Called after an eval pass, with the split's name and its metrics.

        Args:
            ctx: The current training context.
            name: The eval split that just ran.
            metrics: That pass's metrics, keyed as ``evaluate`` returns them
                (without the split-name prefix).
        """

    def on_train_end(self, ctx: TrainContext) -> None:
        """Called once after the last step and the final evaluations."""
