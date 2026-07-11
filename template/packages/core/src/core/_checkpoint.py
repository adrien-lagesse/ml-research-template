"""The on-disk checkpoint format: a safetensors weights file plus a torch sidecar.

A checkpoint splits into two files under one ``name``: ``<name>.safetensors``
holds the model and loss weights (a flat tensor dict, so it stays free of
pickle), and ``<name>.state.pt`` holds the optimizer and scheduler state and the
global step, which carry non-tensor Python objects safetensors cannot store.
``CheckpointState`` is the in-memory record the two files serialize; a matching
loader is not part of this module yet, but the format holds the full training
state so one can be added without changing what is written.
"""

import dataclasses
import pathlib
from typing import Any

import safetensors.torch
import torch
from torch import Tensor


@dataclasses.dataclass(frozen=True)
class CheckpointState:
    """One training snapshot: the weights and the optimizer/scheduler state.

    ``model`` and ``loss`` are the modules' ``state_dict``s (pure tensors);
    ``optimizer`` and ``scheduler`` are their ``state_dict``s, which mix tensors
    with plain Python values. ``global_step`` is the step the snapshot was taken
    at.

    Attributes:
        model: The model's ``state_dict``.
        loss: The loss module's ``state_dict``.
        optimizer: The optimizer's ``state_dict``.
        scheduler: The learning-rate scheduler's ``state_dict``.
        global_step: The global step at which the snapshot was taken.
    """

    model: dict[str, Tensor]
    loss: dict[str, Tensor]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    global_step: int


def save_checkpoint_files(
    state: CheckpointState, directory: pathlib.Path, *, name: str
) -> None:
    """Write ``state`` to ``<directory>/<name>.safetensors`` and ``.state.pt``.

    The weights file merges the model and loss ``state_dict``s into one flat
    tensor dict, keyed ``model.<k>`` and ``loss.<k>``, with every tensor moved to
    CPU. The sidecar holds the optimizer and scheduler state and the global step.
    Each file is written to a temporary path and moved into place with
    ``os.replace``, so a crash mid-write cannot leave a half-written checkpoint
    at the target name.

    Args:
        state: The snapshot to persist.
        directory: Destination directory; created if missing.
        name: Filename stem shared by the two files; an existing checkpoint of
            the same name is overwritten.
    """
    directory.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, Tensor] = {
        f"model.{key}": value.detach().cpu() for key, value in state.model.items()
    }
    tensors.update(
        {f"loss.{key}": value.detach().cpu() for key, value in state.loss.items()}
    )
    weights_tmp = directory / f"{name}.safetensors.tmp"
    safetensors.torch.save_file(
        tensors, weights_tmp, metadata={"global_step": str(state.global_step)}
    )
    weights_tmp.replace(directory / f"{name}.safetensors")

    sidecar_tmp = directory / f"{name}.state.pt.tmp"
    torch.save(
        {
            "optimizer": state.optimizer,
            "scheduler": state.scheduler,
            "global_step": state.global_step,
        },
        sidecar_tmp,
    )
    sidecar_tmp.replace(directory / f"{name}.state.pt")
