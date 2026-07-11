"""Tests for model_summary, the ModelSummary callback, and summary persistence."""

from collections.abc import Mapping
import json
import pathlib

import tensordict
import torch

from core import Loss
from core import Model
from core import Scalar
from core import TensorDictDataset
from core.callbacks import ConfigSummary
from core.callbacks import ModelSummary
from core.callbacks import model_summary
from core.logger import CSVLogger
from core.logger import Logger
from core.logger import LoggerCollection
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
    """Logger that records every (name, summary) pair `log_summary` receives."""

    def __init__(self) -> None:
        self.summaries: list[tuple[str, dict[str, object]]] = []

    def _log(self, values: dict[str, float], *, step: int) -> None:
        pass

    def log_summary(self, summary: Mapping[str, object], *, name: str) -> None:
        self.summaries.append((name, dict(summary)))


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


def test_model_summary_counts_parameters() -> None:
    """A linear layer's totals, bytes, and dtype map match its weight and bias."""
    summary = model_summary(torch.nn.Linear(8, 4))
    assert summary["total_parameters"] == 36  # 8*4 weight + 4 bias
    assert summary["trainable_parameters"] == 36
    assert summary["non_trainable_parameters"] == 0
    assert summary["parameter_bytes"] == 36 * 4  # float32 is 4 bytes each
    assert summary["dtypes"] == {"torch.float32": 36}


def test_model_summary_splits_frozen_parameters() -> None:
    """A frozen parameter counts as non-trainable, not trainable."""
    module = torch.nn.Linear(8, 4)
    module.bias.requires_grad_(False)
    summary = model_summary(module)
    assert summary["trainable_parameters"] == 32
    assert summary["non_trainable_parameters"] == 4


def test_model_summary_breaks_down_by_child() -> None:
    """Each direct child contributes its own parameter totals under ``modules``."""
    module = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 2))
    summary = model_summary(module)
    assert summary["total_parameters"] == 46  # 36 + (4*2 + 2)
    assert summary["modules"] == {
        "0": {"total_parameters": 36, "trainable_parameters": 36},
        "1": {"total_parameters": 10, "trainable_parameters": 10},
    }


def test_model_summary_counts_buffers() -> None:
    """Buffers are counted apart from parameters."""
    # BatchNorm1d(4) holds weight and bias as parameters, and running_mean,
    # running_var, and num_batches_tracked as buffers of 4, 4, and 1 elements.
    summary = model_summary(torch.nn.BatchNorm1d(4))
    assert summary["total_parameters"] == 8
    assert summary["buffer_elements"] == 9


def test_model_summary_omits_directly_owned_from_modules() -> None:
    """A parameter the module owns directly is in the total but not in ``modules``."""
    summary = model_summary(_Scale())
    assert summary["total_parameters"] == 1
    assert summary["modules"] == {}


def test_model_summary_callback_logs_once_at_train_start() -> None:
    """The callback hands one named summary to the logger over a whole run."""
    recorder = _Recorder()
    model = _Scale()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
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
        total_steps=6,
    )
    fit(
        context,
        train_loader=_loader(4, batch_size=2, seed=0),
        evals={},
        callbacks=[ModelSummary(name="model")],
    )
    assert len(recorder.summaries) == 1
    name, summary = recorder.summaries[0]
    assert name == "model"
    assert summary["total_parameters"] == 1


def test_config_summary_logs_the_resolved_config_at_train_start() -> None:
    """The callback hands the context's resolved config to the logger verbatim."""
    recorder = _Recorder()
    model = _Scale()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    context = TrainContext(
        model=model,
        loss=_SquaredLoss(),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        ),
        logger=recorder,
        device=torch.device("cpu"),
        log_every_steps=1,
        total_steps=1,
        _resolved_config={"seed": 0, "model": {"width": 32}},
    )
    ConfigSummary(name="config").on_train_start(context)
    assert recorder.summaries == [("config", {"seed": 0, "model": {"width": 32}})]


def test_csv_logger_writes_summary_json(tmp_path: pathlib.Path) -> None:
    """CSVLogger writes the summary as pretty-printed JSON under its run dir."""
    logger = CSVLogger(experiment_name="exp", run_name="run", root=tmp_path)
    logger.log_summary(
        {"total_parameters": 12, "dtypes": {"torch.float32": 12}}, name="model"
    )
    path = logger.run_dir / "model.json"
    assert path.is_file()
    assert json.loads(path.read_text()) == {
        "total_parameters": 12,
        "dtypes": {"torch.float32": 12},
    }


def test_logger_collection_fans_summary_out(tmp_path: pathlib.Path) -> None:
    """A collection offers the summary to every child logger."""
    first = CSVLogger(experiment_name="exp", run_name="a", root=tmp_path)
    second = CSVLogger(experiment_name="exp", run_name="b", root=tmp_path)
    LoggerCollection([first, second]).log_summary({"total_parameters": 3}, name="model")
    assert (first.run_dir / "model.json").is_file()
    assert (second.run_dir / "model.json").is_file()
