"""Tests for the callback interface and how fit fires its hooks."""

import tensordict
import torch

from core import EvalSchedule
from core import Loss
from core import Model
from core import Scalar
from core import ScheduleTracker
from core import TensorDictDataset
from core.callbacks import Callback
from core.logger import Logger
from core.metrics import MeanAbsoluteError
from core.metrics import RunningMean
from core.train import EvalSplit
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


class _NullLogger(Logger):
    """Logger that discards every call."""

    def _log(self, values: dict[str, float], *, step: int) -> None:
        pass


class _RecordingCallback(Callback):
    """Callback that records every hook it receives and the step it saw."""

    def __init__(self) -> None:
        self.hooks: list[tuple[str, int]] = []
        self.evals: list[tuple[str, frozenset[str], int]] = []
        self.total_steps: list[int] = []

    def on_train_start(self, ctx: TrainContext) -> None:
        self.hooks.append(("train_start", ctx.current_step))
        self.total_steps.append(ctx.total_steps)

    def on_step_end(self, ctx: TrainContext) -> None:
        self.hooks.append(("step_end", ctx.current_step))

    def on_eval_end(
        self, ctx: TrainContext, *, name: str, metrics: dict[str, Scalar]
    ) -> None:
        self.hooks.append(("eval_end", ctx.current_step))
        self.evals.append((name, frozenset(metrics), ctx.current_step))

    def on_train_end(self, ctx: TrainContext) -> None:
        self.hooks.append(("train_end", ctx.current_step))


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


def _fit(callback: Callback, evals: dict[str, EvalSplit]) -> None:
    """Run `fit` for 10 single-sample steps with one callback."""
    model = _Scale()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = TrainContext(
        model=model,
        loss=_SquaredLoss(),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        ),
        logger=_NullLogger(),
        device=torch.device("cpu"),
        log_every_steps=4,
        total_steps=10,
    )
    fit(
        context,
        train_loader=_loader(10, batch_size=1, seed=0),
        evals=evals,
        callbacks=[callback],
    )


def test_train_start_and_train_end_fire_once_first_and_last() -> None:
    """`on_train_start` opens and `on_train_end` closes the hook sequence."""
    callback = _RecordingCallback()
    _fit(callback, evals={})
    names = [name for name, _ in callback.hooks]
    assert names[0] == "train_start"
    assert names[-1] == "train_end"
    assert names.count("train_start") == 1
    assert names.count("train_end") == 1


def test_step_end_fires_once_per_step_with_monotonic_steps() -> None:
    """`on_step_end` fires every step, seeing 1..total_steps in order."""
    callback = _RecordingCallback()
    _fit(callback, evals={})
    step_ends = [step for name, step in callback.hooks if name == "step_end"]
    assert step_ends == list(range(1, 11))


def test_context_reports_the_full_run_length() -> None:
    """`total_steps` reports the configured run length in global steps."""
    callback = _RecordingCallback()
    _fit(callback, evals={})
    assert callback.total_steps == [10]


def test_eval_end_fires_per_due_eval_with_its_name_and_metrics() -> None:
    """`on_eval_end` fires for each due eval, carrying the split name and keys."""
    callback = _RecordingCallback()
    evals = {
        "validation": EvalSplit(
            loader=_loader(6, batch_size=3, seed=1),
            metrics=MeanAbsoluteError(),
            tracker=ScheduleTracker(EvalSchedule(steps=5), total_steps=10),
        ),
        "test": EvalSplit(
            loader=_loader(6, batch_size=3, seed=2),
            metrics=MeanAbsoluteError(),
            tracker=ScheduleTracker(EvalSchedule(at_end=True), total_steps=10),
        ),
    }
    _fit(callback, evals)
    assert callback.evals == [
        ("validation", frozenset({"mae"}), 5),
        ("validation", frozenset({"mae"}), 10),
        ("test", frozenset({"mae"}), 10),
    ]


def test_fit_runs_without_callbacks() -> None:
    """The default empty callbacks sequence leaves the loop unchanged."""
    model = _Scale()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = TrainContext(
        model=model,
        loss=_SquaredLoss(),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        ),
        logger=_NullLogger(),
        device=torch.device("cpu"),
        log_every_steps=4,
        total_steps=10,
    )
    fit(context, train_loader=_loader(10, batch_size=1, seed=0), evals={})


def test_context_carries_last_loss_and_latest_eval_metrics() -> None:
    """`last_loss` fills from the first step; `last_metrics` from the first eval."""
    seen: list[tuple[bool, frozenset[str]]] = []

    class _Probe(Callback):
        def on_step_end(self, ctx: TrainContext) -> None:
            seen.append((ctx.last_loss is not None, frozenset(ctx.last_metrics)))

    evals = {
        "validation": EvalSplit(
            loader=_loader(6, batch_size=3, seed=1),
            metrics=MeanAbsoluteError(),
            tracker=ScheduleTracker(EvalSchedule(steps=5), total_steps=10),
        ),
    }
    _fit(_Probe(), evals)
    # A step's loss is on the context from step 1 onward.
    assert all(has_loss for has_loss, _ in seen)
    # The step-5 validation eval seeds last_metrics; step 1 is still empty.
    assert seen[0][1] == frozenset()
    assert seen[-1][1] == frozenset({"validation/mae"})
