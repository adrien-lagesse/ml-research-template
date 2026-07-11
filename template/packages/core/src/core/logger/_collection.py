"""A logger that fans every call out to a sequence of child loggers."""

from collections.abc import Mapping
from collections.abc import Sequence

from core._checkpoint import CheckpointState
from core.logger._logger import Logger


class LoggerCollection(Logger):
    """Logger that fans every call out to a sequence of child loggers."""

    def __init__(self, loggers: Sequence[Logger]) -> None:
        """Wrap child loggers behind the single-logger interface.

        Args:
            loggers: Backends that each receive every logged step.
        """
        self._loggers = list(loggers)

    def _log(self, values: dict[str, float], *, step: int) -> None:
        for logger in self._loggers:
            logger._log(values, step=step)  # noqa: SLF001

    def close(self) -> None:
        """Close every child logger."""
        for logger in self._loggers:
            logger.close()

    def save_checkpoint(self, state: CheckpointState, *, name: str) -> None:
        """Offer the snapshot to every child; each stores it or ignores it."""
        for logger in self._loggers:
            logger.save_checkpoint(state, name=name)

    def log_summary(self, summary: Mapping[str, object], *, name: str) -> None:
        """Offer the summary to every child; each records it or ignores it."""
        for logger in self._loggers:
            logger.log_summary(summary, name=name)
