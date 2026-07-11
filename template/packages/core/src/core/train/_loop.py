"""The generic training loop and evaluation pass over the core contracts."""

from collections.abc import Sequence
import dataclasses
import time
from typing import TYPE_CHECKING

import tensordict
import torch
import torch.utils.data

from core._contracts import Model
from core._contracts import Scalar
from core._schedule import ScheduleTracker
from core.metrics import Metric
from core.train._context import TrainContext

if TYPE_CHECKING:
    from core.callbacks import Callback


@dataclasses.dataclass(frozen=True)
class EvalSplit:
    """One eval dataset's loader, metric collection, and firing schedule."""

    loader: torch.utils.data.DataLoader[tensordict.TensorDictBase]
    metrics: Metric
    tracker: ScheduleTracker


def evaluate(
    model: Model,
    loader: torch.utils.data.DataLoader[tensordict.TensorDictBase],
    metrics: Metric,
    device: torch.device,
) -> dict[str, Scalar]:
    """Run one full evaluation pass and return the computed metrics.

    ``metrics`` is reset before the pass, so the result covers exactly this
    sweep over ``loader``. Each batch flows through ``model`` in eval mode
    under ``torch.inference_mode()`` before the metrics see it; a ``Loss``
    inside a collection accumulates its terms like any other metric. The
    model is returned to train mode before this returns.

    Args:
        model: Model to evaluate.
        loader: Eval loader to sweep once.
        metrics: Metric to reset, update per batch, and compute.
        device: Device each batch is moved to before the forward pass.

    Returns:
        The computed metrics, one 0-d tensor per name.
    """
    model.eval()
    metrics.reset()
    with torch.inference_mode():
        for sample in loader:
            batch = sample.to(device)
            batch = model(batch)
            metrics.update(batch)
    model.train()
    return metrics.compute()


def fit(
    context: TrainContext,
    *,
    train_loader: torch.utils.data.DataLoader[tensordict.TensorDictBase],
    evals: dict[str, EvalSplit],
    callbacks: Sequence["Callback"] = (),
) -> None:
    """Run the training loop, its scheduled evaluations, and the final evals.

    Drives the loop over ``train_loader`` using the model, loss, optimizer,
    scheduler, logger, device, and step count held in ``context``. Takes exactly
    ``context.total_steps`` optimizer steps, cycling the loader as many times as
    needed, and steps the scheduler after each one. Every
    ``context.log_every_steps`` global steps the loss terms accumulated since the
    last log are logged under ``train/`` and the loss is reset; train logging
    reports nothing else. After each optimizer step, every split whose tracker is
    due gets a full evaluation pass. When the last step is taken, unlogged loss
    terms are flushed and every split whose schedule sets ``at_end`` is evaluated
    once more.

    Callbacks observe the run through four hooks: ``on_train_start`` before the
    first step, ``on_step_end`` after each optimizer step, ``on_eval_end`` after
    each eval pass (with the split's name and metrics), and ``on_train_end``
    after the final evals. Each hook receives ``context`` with ``current_step``,
    ``time_since_start``, ``last_loss`` (the most recent step's training loss),
    and ``last_metrics`` (the latest value of each eval metric) refreshed.

    Args:
        context: The run's objects and settings. Its ``current_step`` must be 0;
            the loss must not be shared with the eval collections, as it
            accumulates the training loss terms between logs.
        train_loader: Loader cycled until ``total_steps`` steps have run.
        evals: Eval splits by name; each is evaluated when its tracker fires.
        callbacks: Callbacks run at the loop's hook points, in order.
    """
    assert context.current_step == 0, (
        f"context must start at step 0, got {context.current_step}"
    )
    assert len(train_loader) > 0, "train_loader is empty"
    assert context.total_steps > 0, (
        f"total_steps must be positive, got {context.total_steps}"
    )

    model = context.model
    loss = context.loss
    optimizer = context.optimizer
    scheduler = context.scheduler
    logger = context.logger
    device = context.device
    total_steps = context.total_steps
    current_step = 0
    start_seconds = time.monotonic()
    last_loss: Scalar | None = None
    last_metrics: dict[str, Scalar] = {}

    def at_step() -> TrainContext:
        """The run context snapshotted at the current step and elapsed time."""
        return dataclasses.replace(
            context,
            current_step=current_step,
            time_since_start=time.monotonic() - start_seconds,
            last_loss=last_loss,
            last_metrics=dict(last_metrics),
        )

    def run_eval(name: str, split: EvalSplit) -> None:
        evaluated = evaluate(model, split.loader, split.metrics, device)
        logger.log_dict(evaluated, step=current_step, prefix=name)
        last_metrics.update(
            {f"{name}/{key}": value for key, value in evaluated.items()}
        )
        for callback in callbacks:
            callback.on_eval_end(at_step(), name=name, metrics=evaluated)

    loss.reset()
    model.train()
    for callback in callbacks:
        callback.on_train_start(at_step())
    while current_step < total_steps:
        for sample in train_loader:
            batch = sample.to(device)
            optimizer.zero_grad()
            batch = model(batch)
            scalar = loss.update_and_loss(batch)
            scalar.backward()
            optimizer.step()
            scheduler.step()
            current_step += 1
            last_loss = scalar.detach()

            if current_step % context.log_every_steps == 0:
                logger.log_dict(loss.compute(), step=current_step, prefix="train")
                loss.reset()
            for name, split in evals.items():
                if split.tracker.due(step=current_step):
                    run_eval(name, split)
            for callback in callbacks:
                callback.on_step_end(at_step())
            if current_step >= total_steps:
                break

    if current_step % context.log_every_steps != 0:
        logger.log_dict(loss.compute(), step=current_step, prefix="train")
    for name, split in evals.items():
        if split.tracker.due_at_end():
            run_eval(name, split)
    for callback in callbacks:
        callback.on_train_end(at_step())
