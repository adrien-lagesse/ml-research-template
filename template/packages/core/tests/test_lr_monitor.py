"""Tests for the LRMonitor callback."""

import pytest
import tensordict
import torch

from core import Loss
from core import Model
from core import Scalar
from core import TensorDictDataset
from core.callbacks import LRMonitor
from core.logger import Logger
from core.metrics import RunningMean
from core.train import TrainContext
from core.train import fit


class _Scale(Model):
    """Model that multiplies ``input`` by one learnable scalar into ``prediction``."""

    def __init__(self) -> None:
        super().__init__(generator=torch.Generator())
        self.scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: tensordict.TensorDictBase) -> tensordict.TensorDictBase:
        batch["prediction"] = batch.get("input") * self.scale
        return batch


class _TwoScale(Model):
    """Model with two learnable scalars, for a two-parameter-group optimizer."""

    def __init__(self) -> None:
        super().__init__(generator=torch.Generator())
        self.a = torch.nn.Parameter(torch.tensor(0.0))
        self.b = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch: tensordict.TensorDictBase) -> tensordict.TensorDictBase:
        batch["prediction"] = batch.get("input") * (self.a + self.b)
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


class _Recorder(Logger):
    """Logger that records every (values, step) pair `_log` receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], int]] = []

    def _log(self, values: dict[str, float], *, step: int) -> None:
        self.calls.append((values, step))

    def series(self, key: str) -> list[tuple[int, float]]:
        return [(step, values[key]) for values, step in self.calls if key in values]


def _loader(
    n: int, batch_size: int, seed: int
) -> torch.utils.data.DataLoader[tensordict.TensorDictBase]:
    """A loader over ``target = 2 * input`` pairs of shape (n, 1)."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n, 1, generator=generator)
    data = tensordict.TensorDict(
        {"input": inputs, "target": 2.0 * inputs}, batch_size=[n]
    )
    return torch.utils.data.DataLoader(
        TensorDictDataset(data), batch_size=batch_size, collate_fn=lambda batch: batch
    )


def test_lr_monitor_logs_the_scheduled_learning_rate() -> None:
    """The logged rate tracks the per-step StepLR decay at the schedule steps."""
    recorder = _Recorder()
    model = _Scale()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = TrainContext(
        model=model,
        loss=_SquaredLoss(),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5),
        logger=recorder,
        device=torch.device("cpu"),
        log_every_steps=100,
        total_steps=6,
    )
    fit(
        context,
        train_loader=_loader(2, batch_size=1, seed=0),
        evals={},
        callbacks=[LRMonitor(every_steps=2)],
    )
    # Initial rate at step 0, then the post-step rate at each schedule multiple:
    # the scheduler steps every training step, so StepLR halves the rate every
    # two steps.
    series = recorder.series("lr")
    steps = [step for step, _ in series]
    values = [value for _, value in series]
    assert steps == [0, 2, 4, 6]
    torch.testing.assert_close(values, [0.1, 0.05, 0.025, 0.0125])


def test_lr_monitor_names_multiple_groups() -> None:
    """With several parameter groups each rate is logged under lr/group{i}."""
    recorder = _Recorder()
    model = _TwoScale()
    optimizer = torch.optim.SGD(
        [
            {"params": [model.a], "lr": 0.1},
            {"params": [model.b], "lr": 0.2},
        ]
    )
    context = TrainContext(
        model=model,
        loss=_SquaredLoss(),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        ),
        logger=recorder,
        device=torch.device("cpu"),
        log_every_steps=100,
        total_steps=2,
    )
    fit(
        context,
        train_loader=_loader(2, batch_size=1, seed=0),
        evals={},
        callbacks=[LRMonitor(every_steps=2)],
    )
    assert recorder.series("lr/group0")[0] == (0, pytest.approx(0.1))
    assert recorder.series("lr/group1")[0] == (0, pytest.approx(0.2))
