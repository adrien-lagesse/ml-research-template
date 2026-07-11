"""Tests for the Logger interface, LocalLogger, TerminalLogger, and LoggerCollection."""

import csv
from datetime import datetime
import logging
from pathlib import Path

import pytest
import torch

from core.logger import LocalLogger
from core.logger import Logger
from core.logger import LoggerCollection
from core.logger import TerminalLogger


class _Recorder(Logger):
    """Logger that records every (values, step) pair `_log` receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], int]] = []
        self.closed = False

    def _log(self, values: dict[str, float], *, step: int) -> None:
        self.calls.append((values, step))

    def close(self) -> None:
        self.closed = True


def test_log_dict_converts_scalars_to_floats() -> None:
    """`log_dict` hands `_log` plain floats, not tensors."""
    recorder = _Recorder()
    recorder.log_dict({"loss": torch.tensor(2.5)}, step=3)
    assert recorder.calls == [({"loss": 2.5}, 3)]
    assert isinstance(recorder.calls[0][0]["loss"], float)


def test_log_dict_applies_prefix() -> None:
    """A prefix lands on every key as `prefix/key`."""
    recorder = _Recorder()
    recorder.log_dict(
        {"loss": torch.tensor(1.0), "mae": torch.tensor(0.5)},
        step=10,
        prefix="validation",
    )
    assert recorder.calls == [({"validation/loss": 1.0, "validation/mae": 0.5}, 10)]


def test_log_dict_without_prefix_keeps_keys() -> None:
    """Keys pass through unchanged when no prefix is given."""
    recorder = _Recorder()
    recorder.log_dict({"loss": torch.tensor(1.0)}, step=1)
    assert list(recorder.calls[0][0]) == ["loss"]


def test_local_logger_creates_run_directory(tmp_path: Path) -> None:
    """The metrics file sits under `<root>/<experiment>/<stamp>-<run>/`."""
    logger = LocalLogger(experiment_name="exp", run_name="baseline", root=tmp_path)
    assert logger.run_dir.parent == tmp_path / "exp"
    assert logger.run_dir.name.endswith("-baseline")
    logger.log_dict({"loss": torch.tensor(1.0)}, step=1)
    assert (logger.run_dir / "metrics.csv").exists()


def test_local_logger_suffixes_colliding_run_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs starting in the same second get distinct `-2`, `-3`, … directories."""

    class _FrozenDatetime:
        @staticmethod
        def now() -> datetime:
            return datetime(2026, 1, 1, 12, 0, 0)

    monkeypatch.setattr("core.logger._local_logger.datetime", _FrozenDatetime)
    loggers = [
        LocalLogger(experiment_name="exp", run_name="run", root=tmp_path)
        for _ in range(3)
    ]
    names = [logger.run_dir.name for logger in loggers]
    assert names == [
        "2026-01-01-12-00-00-run",
        "2026-01-01-12-00-00-run-2",
        "2026-01-01-12-00-00-run-3",
    ]
    assert all((logger.run_dir / "metrics.csv").exists() for logger in loggers)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def test_local_logger_widens_header_for_new_keys(tmp_path: Path) -> None:
    """Rows with disjoint keys share one header, with empty cells elsewhere."""
    logger = LocalLogger(experiment_name="exp", run_name="run", root=tmp_path)
    logger.log_dict({"loss": torch.tensor(2.0)}, step=1, prefix="train")
    logger.log_dict({"mae": torch.tensor(0.5)}, step=2, prefix="validation")
    rows = _read_rows(logger.run_dir / "metrics.csv")
    assert rows == [
        {"step": "1", "train/loss": "2.0", "validation/mae": ""},
        {"step": "2", "train/loss": "", "validation/mae": "0.5"},
    ]


def test_local_logger_appends_rows_with_known_keys(tmp_path: Path) -> None:
    """Repeated keys accumulate as rows without losing earlier ones."""
    logger = LocalLogger(experiment_name="exp", run_name="run", root=tmp_path)
    for step in (1, 2, 3):
        logger.log_dict({"loss": torch.tensor(float(step))}, step=step)
    rows = _read_rows(logger.run_dir / "metrics.csv")
    assert [row["step"] for row in rows] == ["1", "2", "3"]
    assert [row["loss"] for row in rows] == ["1.0", "2.0", "3.0"]


def test_terminal_logger_renders_one_line(caplog: pytest.LogCaptureFixture) -> None:
    """The terminal logger emits `step N: key=value` at 4 decimals."""
    logger = TerminalLogger()
    with caplog.at_level(logging.INFO):
        logger.log_dict({"loss": torch.tensor(2.5)}, step=7, prefix="train")
    assert "step 7: train/loss=2.5000" in caplog.text


def test_logger_collection_fans_out_to_every_child() -> None:
    """One `log_dict` call reaches each child with identical values."""
    children = [_Recorder(), _Recorder()]
    collection = LoggerCollection(children)
    collection.log_dict({"loss": torch.tensor(1.0)}, step=5, prefix="train")
    for child in children:
        assert child.calls == [({"train/loss": 1.0}, 5)]


def test_logger_collection_closes_every_child() -> None:
    """Closing the collection closes each child."""
    children = [_Recorder(), _Recorder()]
    LoggerCollection(children).close()
    assert all(child.closed for child in children)
