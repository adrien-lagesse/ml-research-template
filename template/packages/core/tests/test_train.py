"""Tests for the training loop and the evaluation pass."""

import tensordict
import torch

from core import EvalSchedule
from core import Loss
from core import Model
from core import Scalar
from core import ScheduleTracker
from core import TensorDictDataset
from core.logger import Logger
from core.metrics import MeanAbsoluteError
from core.metrics import RunningMean
from core.train import EvalSplit
from core.train import TrainContext
from core.train import evaluate
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


class _Recorder(Logger):
    """Logger that records every (values, step) pair `_log` receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], int]] = []

    def _log(self, values: dict[str, float], *, step: int) -> None:
        self.calls.append((values, step))

    def steps_of(self, key: str) -> list[int]:
        return [step for values, step in self.calls if key in values]


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


def _fit(model: Model, logger: Logger, evals: dict[str, EvalSplit]) -> None:
    """Run `fit` for 10 single-sample steps, logging every 4."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = TrainContext(
        model=model,
        loss=_SquaredLoss(),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        ),
        logger=logger,
        device=torch.device("cpu"),
        log_every_steps=4,
        total_steps=10,
    )
    fit(context, train_loader=_loader(10, batch_size=1, seed=0), evals=evals)


def test_fit_logs_train_loss_on_cadence_and_flushes_the_remainder() -> None:
    """Train loss lands every `log_every_steps` steps, plus once at the end."""
    recorder = _Recorder()
    _fit(_Scale(), recorder, evals={})
    assert recorder.steps_of("train/loss") == [4, 8, 10]


def test_fit_evaluates_each_split_on_its_own_schedule() -> None:
    """A `steps` schedule fires during training; `at_end` fires once after."""
    recorder = _Recorder()
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
    _fit(_Scale(), recorder, evals)
    assert recorder.steps_of("validation/mae") == [5, 10]
    assert recorder.steps_of("test/mae") == [10]


def test_fit_reduces_the_training_loss() -> None:
    """SGD on the scale parameter drives the logged loss down."""
    recorder = _Recorder()
    _fit(_Scale(), recorder, evals={})
    losses = [values["train/loss"] for values, _ in recorder.calls]
    assert losses[-1] < losses[0]


def test_evaluate_computes_over_the_full_loader() -> None:
    """The returned MAE covers every batch of the loader."""
    model = _Scale()
    loader = _loader(8, batch_size=3, seed=3)
    result = evaluate(model, loader, MeanAbsoluteError(), torch.device("cpu"))
    inputs = torch.cat([batch.get("input") for batch in loader])
    torch.testing.assert_close(result["mae"], (2.0 * inputs).abs().mean())


def test_evaluate_returns_the_model_to_train_mode() -> None:
    """An eval pass leaves the model ready to keep training."""
    model = _Scale()
    model.eval()
    evaluate(
        model,
        _loader(4, batch_size=2, seed=4),
        MeanAbsoluteError(),
        torch.device("cpu"),
    )
    assert model.training
