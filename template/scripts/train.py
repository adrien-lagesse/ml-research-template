"""Hydra glue around ``core.fit`` for the SSL experiments.

Instantiates the ``Data``, ``Model``, ``Loss``, optimizer, scheduler, and
metrics named in the config, assembles one ``core.EvalSplit`` per eval
dataset, and hands everything to ``core.fit``. Train logging reports the loss
terms only, every ``log_every_steps`` global steps; the ``metrics`` config
applies to the eval datasets alone. Each eval loader follows its own
``core.EvalSchedule`` from ``cfg.evals``, keyed by the eval dataset's name.

No default config is set. Add the experiment directory to the search path with
``-cd`` and name its config with ``-cn``; ``just run <experiment>`` wraps this:

    uv run scripts/train.py -cd experiments/dummy_regression -cn config
"""

import hydra
import hydra.utils
import omegaconf
import tensordict
import torch
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data

from core import Data
from core import EvalSchedule
from core import Loss
from core import Model
from core import ScheduleTracker
from core.callbacks import Callback
from core.logger import LoggerCollection
from core.metrics import MetricCollection
from core.train import EvalSplit
from core.train import TrainContext
from core.train import fit


def _eval_splits(
    cfg: omegaconf.DictConfig,
    eval_loaders: dict[str, torch.utils.data.DataLoader[tensordict.TensorDictBase]],
    total_steps: int,
    device: torch.device,
) -> dict[str, EvalSplit]:
    """Assemble one ``EvalSplit`` per eval dataset from the config.

    Each split pairs its loader with a private ``MetricCollection`` (a fresh
    ``cfg.loss`` instance moved to ``device`` plus the ``cfg.metrics``
    collection, so eval passes report loss terms alongside the metrics without
    sharing accumulators) and a ``ScheduleTracker`` built from its ``cfg.evals``
    entry.

    Args:
        cfg: The experiment config; reads ``evals``, ``loss``, and
            ``metrics``. ``cfg.evals`` maps each eval dataset name to
            ``EvalSchedule`` keyword arguments and must cover the eval
            datasets exactly.
        eval_loaders: Eval loaders by split name.
        total_steps: Length of the run in global steps.
        device: Device each split's loss is moved to, matching the batches.

    Returns:
        One split per eval dataset, keyed by name.
    """
    container = omegaconf.OmegaConf.to_container(cfg.evals)
    assert isinstance(container, dict)
    schedule_kwargs = {str(name): kwargs for name, kwargs in container.items()}
    assert schedule_kwargs.keys() == eval_loaders.keys(), (
        f"cfg.evals keys {set(schedule_kwargs)} != eval datasets {set(eval_loaders)}"
    )
    return {
        name: EvalSplit(
            loader=eval_loaders[name],
            metrics=MetricCollection(
                [
                    hydra.utils.instantiate(cfg.loss).to(device),
                    hydra.utils.instantiate(cfg.metrics),
                ]
            ),
            tracker=ScheduleTracker(EvalSchedule(**kwargs), total_steps=total_steps),
        )
        for name, kwargs in schedule_kwargs.items()
    }


@hydra.main(version_base="1.3", config_path="../.global-hydra", config_name=None)
def main(cfg: omegaconf.DictConfig) -> None:
    """Train the configured model and log train and eval metrics.

    Everything comes from ``cfg``: ``data``, ``model``, ``loss``,
    ``optimizer``, ``scheduler``, and ``metrics`` are Hydra-instantiated
    (the optimizer as a partial applied to the model and loss parameters, the
    scheduler as a partial applied to the optimizer), and ``device`` is resolved
    once and threaded through. The model is seeded from a ``torch.Generator``
    set with ``cfg.seed`` and then moved to ``device`` whole; the loss, which
    may carry its own parameters, is moved to ``device`` and its parameters join
    the optimizer.
    The ``metrics`` config applies only to the eval datasets: each eval split
    gets its own ``MetricCollection`` holding a private ``loss`` instance
    plus the ``metrics`` config, so eval passes report loss terms alongside
    the metrics, while train logging reports loss terms only. Any ``callbacks``
    in the config are instantiated and passed through. The assembled pieces are
    collected into a ``TrainContext`` and handed to ``core.fit``, which runs the
    loop.

    Args:
        cfg: Hydra config selected via ``-cn``; see the module docstring.
    """
    device = torch.device(cfg.device)
    model_generator = torch.Generator().manual_seed(cfg.seed)

    data: Data = hydra.utils.instantiate(cfg.data)
    model: Model = hydra.utils.instantiate(cfg.model, generator=model_generator).to(
        device
    )
    loss: Loss = hydra.utils.instantiate(cfg.loss).to(device)
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer)(
        [*model.parameters(), *loss.parameters()]
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler = hydra.utils.instantiate(
        cfg.scheduler
    )(optimizer)

    train_loader = data.train_dataloader()
    total_steps = cfg.total_steps
    evals = _eval_splits(
        cfg, data.eval_dataloaders(), total_steps=total_steps, device=device
    )
    logger = LoggerCollection(hydra.utils.instantiate(cfg.loggers))
    callbacks: list[Callback] = list(hydra.utils.instantiate(cfg.get("callbacks", [])))

    resolved_config = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved_config, dict)
    config = {str(key): value for key, value in resolved_config.items()}
    context = TrainContext(
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        device=device,
        log_every_steps=cfg.log_every_steps,
        total_steps=total_steps,
        seed=cfg.seed,
        _resolved_config=config,
    )

    try:
        fit(context, train_loader=train_loader, evals=evals, callbacks=callbacks)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
