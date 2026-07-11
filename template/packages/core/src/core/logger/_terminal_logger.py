"""Logger that renders each step and summary as one line via the stdlib logging."""

from collections.abc import Mapping
import logging

from core.logger._logger import Logger

_terminal_log = logging.getLogger(__name__)


class TerminalLogger(Logger):
    """Logger that renders each step as one line via the stdlib ``logging``."""

    def _log(self, values: dict[str, float], *, step: int) -> None:
        rendered = " ".join(f"{name}={value:.4f}" for name, value in values.items())
        _terminal_log.info("step %d: %s", step, rendered)

    def log_summary(self, summary: Mapping[str, object], *, name: str) -> None:
        """Render the summary's top-level scalar entries as one line.

        Nested entries (per-module or per-dtype breakdowns) are left for the
        persisting backends; the terminal shows only the flat scalars, so the
        line stays readable regardless of which summary it is.
        """
        scalars = {
            key: value
            for key, value in summary.items()
            if isinstance(value, (int, float, str))
        }
        rendered = " ".join(f"{key}={value}" for key, value in scalars.items())
        _terminal_log.info("summary %s: %s", name, rendered)
