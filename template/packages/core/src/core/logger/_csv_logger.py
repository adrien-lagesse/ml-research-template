"""Logger that writes a wide ``metrics.csv`` and JSON summaries under a run dir."""

from collections.abc import Mapping
import csv
from datetime import datetime
import json
from pathlib import Path

from core._checkpoint import CheckpointState
from core._checkpoint import save_checkpoint_files
from core.logger._logger import Logger


class CSVLogger(Logger):
    """Logger that writes a wide ``metrics.csv`` under a per-run directory.

    The file lives at ``<root>/<experiment_name>/<timestamp>-<run_name>/metrics.csv``;
    the resolved directory is exposed as ``run_dir``. Each ``_log`` call
    appends one row keyed by ``step``. A call that introduces metric names not
    yet in the header reads all rows back and rewrites the file under the
    widened header, leaving cells empty where a row lacks a column. The file
    is valid CSV after every call, so ``close`` has nothing to flush; between
    calls only the header names are held in memory, never the rows.

    When ``checkpointing`` is set, ``save_checkpoint`` writes the snapshot under
    ``run_dir/checkpoints``, next to ``metrics.csv``; otherwise it is a no-op.
    """

    def __init__(
        self,
        *,
        experiment_name: str,
        run_name: str,
        root: str | Path = ".logs",
        checkpointing: bool = False,
    ) -> None:
        """Create the run directory and ``metrics.csv`` containing only the header.

        The directory is named ``<timestamp>-<run_name>``, with the timestamp
        taken at construction at one-second resolution. When that name is
        already taken (runs started within the same second, e.g. a sweep), a
        ``-2``, ``-3``, … suffix is appended until the directory can be
        created, so two runs never share a file. The initial header holds only
        the ``step`` column; metric columns appear as they are first logged.

        Args:
            experiment_name: Groups runs under ``<root>/<experiment_name>``.
            run_name: Human-readable part of the run directory name, after the
                timestamp.
            root: Directory that holds all experiments. Created if missing.
            checkpointing: When true, ``save_checkpoint`` persists snapshots
                under ``run_dir/checkpoints``; when false, it is a no-op.
        """
        self.checkpointing = checkpointing
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        parent = Path(root) / experiment_name
        base_name = f"{stamp}-{run_name}"
        self.run_dir = parent / base_name
        suffix = 2
        while True:
            try:
                self.run_dir.mkdir(parents=True)
                break
            except FileExistsError:
                self.run_dir = parent / f"{base_name}-{suffix}"
                suffix += 1
        self._path = self.run_dir / "metrics.csv"
        self._fieldnames: list[str] = ["step"]
        with self._path.open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=self._fieldnames).writeheader()

    def _log(self, values: dict[str, float], *, step: int) -> None:
        row: dict[str, float] = {"step": step, **values}
        new_names = [name for name in row if name not in self._fieldnames]
        if new_names:
            self._fieldnames.extend(new_names)
            self._rewrite_widened(row)
        else:
            with self._path.open("a", newline="") as file:
                csv.DictWriter(file, fieldnames=self._fieldnames).writerow(row)

    def _rewrite_widened(self, row: dict[str, float]) -> None:
        """Rewrite the file under the widened header, with ``row`` appended."""
        with self._path.open(newline="") as file:
            rows: list[dict[str, str] | dict[str, float]] = list(csv.DictReader(file))
        rows.append(row)
        with self._path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self._fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_checkpoint(self, state: CheckpointState, *, name: str) -> None:
        """Write ``state`` under ``run_dir/checkpoints`` when checkpointing is on.

        Args:
            state: The snapshot to persist.
            name: Filename stem for the checkpoint's two files; an existing
                checkpoint of the same name is overwritten.
        """
        if self.checkpointing:
            save_checkpoint_files(state, self.run_dir / "checkpoints", name=name)

    def log_summary(self, summary: Mapping[str, object], *, name: str) -> None:
        """Write ``summary`` as ``run_dir/<name>.json``, pretty-printed.

        Args:
            summary: JSON-serializable metadata to record with the run.
            name: Filename stem for the ``.json`` file; an existing file of the
                same name is overwritten.
        """
        path = self.run_dir / f"{name}.json"
        path.write_text(json.dumps(dict(summary), indent=2) + "\n")
