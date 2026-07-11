"""Tests for the Checkpoint callback, its triggers, and the on-disk format."""

import pathlib
import time

import pytest
import safetensors.torch
import tensordict
import torch

from core import CheckpointState
from core import Loss
from core import Model
from core import Scalar
from core.callbacks import Checkpoint
from core.logger import LocalLogger
from core.logger import Logger
from core.logger import LoggerCollection
from core.metrics import RunningMean
from core.train import TrainContext


class _Scale(Model):
    """Model that multiplies ``input`` by one learnable scalar into ``prediction``."""

    def __init__(self) -> None:
        super().__init__(generator=torch.Generator())
        self.scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: tensordict.TensorDictBase) -> tensordict.TensorDictBase:
        batch["prediction"] = batch.get("input") * self.scale
        return batch


class _SquaredLoss(Loss):
    """Squared-error loss accumulating its terms under ``loss``."""

    def __init__(self) -> None:
        super().__init__()
        self._running = RunningMean()

    def reset(self) -> None:
        self._running.reset()

    def update_and_loss(self, batch: tensordict.TensorDictBase) -> Scalar:
        squared = (batch.get("prediction") - batch.get("target")) ** 2
        self._running.add(squared)
        return squared.mean()

    def compute(self) -> dict[str, Scalar]:
        return {"loss": self._running.mean()}


class _RecordingLogger(Logger):
    """Logger that records the (name, step) of every checkpoint offered to it."""

    def __init__(self) -> None:
        self.saves: list[tuple[str, int]] = []

    def _log(self, values: dict[str, float], *, step: int) -> None:
        pass

    def save_checkpoint(self, state: CheckpointState, *, name: str) -> None:
        self.saves.append((name, state.global_step))


class _Clock:
    """A hand-advanced monotonic clock; patch over `time.monotonic` in tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _context(logger: Logger, *, current_step: int) -> TrainContext:
    """A context wrapping fresh training objects at a given step."""
    model = _Scale()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=0
    )
    return TrainContext(
        model=model,
        loss=_SquaredLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        device=torch.device("cpu"),
        log_every_steps=100,
        total_steps=100,
        current_step=current_step,
    )


def _sample_state(*, global_step: int = 7, scale: float = 1.5) -> CheckpointState:
    """A full checkpoint state with the model's scalar set to ``scale``."""
    model = _Scale()
    with torch.no_grad():
        model.scale.copy_(torch.tensor(scale))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=0
    )
    return CheckpointState(
        model=model.state_dict(),
        loss={},
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        global_step=global_step,
    )


def test_every_steps_saves_on_the_step_cadence() -> None:
    """A `every_steps` checkpoint saves on the exact step multiples, per step."""
    logger = _RecordingLogger()
    checkpoint = Checkpoint(every_steps=5, name="last")
    checkpoint.on_train_start(_context(logger, current_step=0))
    for step in range(1, 13):
        checkpoint.on_step_end(_context(logger, current_step=step))
    assert [step for _, step in logger.saves] == [5, 10]


def test_monitor_saves_only_when_the_metric_improves() -> None:
    """A `min` monitor saves the first eval and every strict improvement."""
    logger = _RecordingLogger()
    checkpoint = Checkpoint(monitor="validation/loss", mode="min", name="best")
    checkpoint.on_train_start(_context(logger, current_step=0))
    for step, value in enumerate([1.0, 0.8, 0.9, 0.5, 0.6], start=1):
        checkpoint.on_eval_end(
            _context(logger, current_step=step),
            name="validation",
            metrics={"loss": torch.tensor(value)},
        )
    assert [step for _, step in logger.saves] == [1, 2, 4]


def test_monitor_max_mode_saves_on_increase() -> None:
    """A `max` monitor treats higher values as better."""
    logger = _RecordingLogger()
    checkpoint = Checkpoint(monitor="validation/acc", mode="max")
    checkpoint.on_train_start(_context(logger, current_step=0))
    for step, value in enumerate([0.5, 0.7, 0.6, 0.9], start=1):
        checkpoint.on_eval_end(
            _context(logger, current_step=step),
            name="validation",
            metrics={"acc": torch.tensor(value)},
        )
    assert [step for _, step in logger.saves] == [1, 2, 4]


def test_monitor_ignores_splits_without_the_key() -> None:
    """A monitor keyed on one split does not react to another split's eval."""
    logger = _RecordingLogger()
    checkpoint = Checkpoint(monitor="validation/loss", mode="min")
    checkpoint.on_train_start(_context(logger, current_step=0))
    checkpoint.on_eval_end(
        _context(logger, current_step=1),
        name="test",
        metrics={"loss": torch.tensor(0.1)},
    )
    assert logger.saves == []


def test_every_seconds_saves_on_the_wall_clock_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `every_seconds` checkpoint saves once the interval has elapsed."""
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    logger = _RecordingLogger()
    checkpoint = Checkpoint(every_seconds=30, name="last")
    checkpoint.on_train_start(_context(logger, current_step=0))
    for step, now in [(1, 20.0), (2, 35.0), (3, 50.0), (4, 70.0)]:
        clock.now = now
        checkpoint.on_step_end(_context(logger, current_step=step))
    assert [step for _, step in logger.saves] == [2, 4]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "cadence"),
        ({"monitor": "validation/loss", "every_steps": 100}, "not both"),
        ({"every_steps": 0}, "positive"),
        ({"every_seconds": -1.0}, "positive"),
        ({"monitor": "validation/loss", "mode": "lowest"}, "mode"),
    ],
    ids=[
        "no-trigger",
        "cadence-and-monitor",
        "zero-steps",
        "negative-seconds",
        "bad-mode",
    ],
)
def test_invalid_configuration_raises(kwargs: dict[str, object], match: str) -> None:
    """A checkpoint that could never fire or is misconfigured raises."""
    with pytest.raises(ValueError, match=match):
        Checkpoint(**kwargs)  # ty: ignore[invalid-argument-type]


def test_local_logger_writes_both_checkpoint_files(tmp_path: pathlib.Path) -> None:
    """An enabled LocalLogger writes the weights and the state sidecar."""
    logger = LocalLogger(
        experiment_name="exp", run_name="run", root=tmp_path, checkpointing=True
    )
    logger.save_checkpoint(_sample_state(global_step=7, scale=1.5), name="best")
    checkpoints = logger.run_dir / "checkpoints"
    assert (checkpoints / "best.safetensors").is_file()
    assert (checkpoints / "best.state.pt").is_file()
    weights = safetensors.torch.load_file(checkpoints / "best.safetensors")
    torch.testing.assert_close(weights["model.scale"], torch.tensor(1.5))
    sidecar = torch.load(checkpoints / "best.state.pt", weights_only=False)
    assert sidecar["global_step"] == 7


def test_local_logger_skips_checkpoint_when_disabled(tmp_path: pathlib.Path) -> None:
    """A LocalLogger with checkpointing off writes nothing."""
    logger = LocalLogger(
        experiment_name="exp", run_name="run", root=tmp_path, checkpointing=False
    )
    logger.save_checkpoint(_sample_state(), name="best")
    assert not (logger.run_dir / "checkpoints").exists()


def test_logger_collection_saves_to_enabled_children_only(
    tmp_path: pathlib.Path,
) -> None:
    """A collection offers the checkpoint to every child; only enabled ones write."""
    enabled = LocalLogger(
        experiment_name="exp", run_name="on", root=tmp_path, checkpointing=True
    )
    disabled = LocalLogger(
        experiment_name="exp", run_name="off", root=tmp_path, checkpointing=False
    )
    LoggerCollection([enabled, disabled]).save_checkpoint(_sample_state(), name="last")
    assert (enabled.run_dir / "checkpoints" / "last.safetensors").is_file()
    assert not (disabled.run_dir / "checkpoints").exists()
