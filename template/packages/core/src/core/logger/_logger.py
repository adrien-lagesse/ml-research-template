"""The logger base class: prefixing, tensor-to-float conversion, and no-op edges."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping

from core._checkpoint import CheckpointState
from core._contracts import Scalar


def _scalar_entries(summary: Mapping[str, object]) -> dict[str, int | float | str]:
    """Keep a summary's flat scalar entries and drop its nested ones.

    A summary mixes flat scalars (a seed, a device, a parameter count) with
    nested breakdowns (per-module, per-dtype). Backends that store a flat table,
    the terminal line and MLflow's params, take the scalars and skip the rest. A
    ``bool`` counts as a scalar: it is an ``int`` subclass and passes through.
    """
    return {
        key: value
        for key, value in summary.items()
        if isinstance(value, (int, float, str))
    }


class Logger(ABC):
    """Records named scalar metrics against a global step.

    ``log_dict`` is the single entry point: it applies an optional key prefix,
    converts each 0-d tensor to a plain float, and hands the result to the
    backend-specific ``_log``. Subclasses implement only ``_log`` (and
    ``close`` when they hold resources), so prefixing and conversion behave
    identically across backends.
    """

    def log_dict(
        self,
        metrics: Mapping[str, Scalar],
        *,
        step: int,
        prefix: str | None = None,
    ) -> None:
        """Log one dict of metrics at a global step.

        Args:
            metrics: Named 0-d tensors, as returned by ``Metric.compute``.
            step: Global step the values belong to.
            prefix: When given, each key is logged as ``"{prefix}/{key}"``.
        """
        values = {
            f"{prefix}/{name}" if prefix else name: float(value.item())
            for name, value in metrics.items()
        }
        self._log(values, step=step)

    @abstractmethod
    def _log(self, values: dict[str, float], *, step: int) -> None:
        """Write already-prefixed, already-converted values for one step.

        Args:
            values: Metric values keyed by their final logged name.
            step: Global step the values belong to.
        """

    # An intentional no-op default: only backends holding resources override.
    def close(self) -> None:  # noqa: B027
        """Release any resources the logger holds. No-op by default."""

    # An intentional no-op default: only backends that persist state override.
    def save_checkpoint(self, state: CheckpointState, *, name: str) -> None:  # noqa: B027
        """Persist one checkpoint under ``name``. No-op by default.

        Backends that persist checkpoints (local files, a tracking server)
        override this; the rest ignore the call, so a checkpoint callback can
        hand every logger a snapshot and let each decide whether to store it.

        Args:
            state: The snapshot to persist.
            name: Filename stem or artifact name the checkpoint is stored under;
                an existing checkpoint of the same name is overwritten.
        """

    # An intentional no-op default: only backends that store metadata override.
    def log_summary(self, summary: Mapping[str, object], *, name: str) -> None:  # noqa: B027
        """Persist a one-off run-summary document under ``name``. No-op by default.

        A summary is run metadata computed once, not a per-step metric: a model's
        parameter counts, a config dump, environment facts. ``summary`` is a
        JSON-serializable mapping whose values are scalars or nested mappings,
        the shape every backend can store in its idiomatic metadata slot. A
        ``LocalLogger`` writes ``<name>.json`` under its run directory; a future
        ``MLflowLogger`` would call ``mlflow.log_dict(dict(summary), f"{name}.json")``
        (optionally flattening the scalar top level into ``mlflow.log_params`` for
        the searchable table); a future ``WandbLogger`` would set
        ``run.summary[name] = dict(summary)`` or ``run.config.update``.

        Args:
            summary: JSON-serializable metadata; scalar or nested-mapping values.
            name: Document name the summary is stored under; an existing summary
                of the same name is overwritten.
        """
