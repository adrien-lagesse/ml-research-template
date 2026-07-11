"""A model's parameter summary and the callback that records it at train start."""

import torch

from core.callbacks._callback import Callback
from core.train import TrainContext


def model_summary(module: torch.nn.Module) -> dict[str, object]:
    """Summarize a module's parameters and buffers.

    Counts parameters and buffers over the whole module, splits the parameter
    count by whether a parameter requires gradients and by dtype, and breaks the
    parameter totals down over the module's direct children. Parameters owned by
    the module itself rather than a child are absent from ``modules`` but still
    counted in the totals, so the per-child figures need not sum to the total.
    Every value is a plain ``int``, ``str``, or nested ``dict``, so the result
    is JSON-serializable as returned.

    Args:
        module: The module to inspect. Read only; nothing is moved or mutated.

    Returns:
        A summary with the top-level counts ``total_parameters``,
        ``trainable_parameters``, ``non_trainable_parameters``,
        ``parameter_bytes``, ``buffer_elements``, and ``buffer_bytes``; a
        ``dtypes`` map from each parameter dtype's name to its element count; and
        a ``modules`` map from each direct child's name to its
        ``total_parameters`` and ``trainable_parameters``.
    """
    total = 0
    trainable = 0
    parameter_bytes = 0
    dtypes: dict[str, int] = {}
    for parameter in module.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        parameter_bytes += count * parameter.element_size()
        dtype_name = str(parameter.dtype)
        dtypes[dtype_name] = dtypes.get(dtype_name, 0) + count

    buffer_elements = 0
    buffer_bytes = 0
    for buffer in module.buffers():
        buffer_elements += buffer.numel()
        buffer_bytes += buffer.numel() * buffer.element_size()

    modules: dict[str, object] = {}
    for name, child in module.named_children():
        child_total = sum(parameter.numel() for parameter in child.parameters())
        child_trainable = sum(
            parameter.numel()
            for parameter in child.parameters()
            if parameter.requires_grad
        )
        modules[name] = {
            "total_parameters": child_total,
            "trainable_parameters": child_trainable,
        }

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "parameter_bytes": parameter_bytes,
        "buffer_elements": buffer_elements,
        "buffer_bytes": buffer_bytes,
        "dtypes": dtypes,
        "modules": modules,
    }


class ModelSummary(Callback):
    """Records the model's parameter counts once, at the start of training.

    On ``on_train_start`` it summarizes ``ctx.model`` (total, trainable, and
    non-trainable parameters, byte sizes, a per-dtype and per-child breakdown)
    and hands the summary to the logger under ``name``. It is a one-shot record
    of what is being trained, so it stores through the logger's ``log_summary``
    rather than the per-step metric path.
    """

    def __init__(self, *, name: str = "model") -> None:
        """Set the document name the summary is stored under.

        Args:
            name: Name passed to ``log_summary``; a ``CSVLogger`` writes it as
                ``<name>.json`` under the run directory.
        """
        self._name = name

    def on_train_start(self, ctx: TrainContext) -> None:
        """Summarize the model and hand the summary to the logger."""
        ctx.logger.log_summary(model_summary(ctx.model), name=self._name)
