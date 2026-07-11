"""A callback that records the run's resolved config at the start of training."""

from core.callbacks._callback import Callback
from core.train import TrainContext


class ConfigSummary(Callback):
    """Records the run's fully resolved config once, at the start of training.

    On ``on_train_start`` it hands ``TrainContext._resolved_config`` to the
    logger under ``name``, so a ``CSVLogger`` writes it as ``<name>.json`` and a
    terminal logger prints its top-level scalar settings. It persists the config
    wholesale as run provenance, the one sanctioned use of that private field: it
    stores the config, never reads its values.
    """

    def __init__(self, *, name: str = "config") -> None:
        """Set the document name the config is stored under.

        Args:
            name: Name passed to ``log_summary``; a ``CSVLogger`` writes it as
                ``<name>.json`` under the run directory.
        """
        self._name = name

    def on_train_start(self, ctx: TrainContext) -> None:
        """Hand the resolved config to the logger as run provenance."""
        ctx.logger.log_summary(ctx._resolved_config, name=self._name)  # noqa: SLF001
