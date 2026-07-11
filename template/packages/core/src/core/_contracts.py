"""The ``Scalar`` alias and the ``Model`` contract over TensorDict batches."""

from abc import ABC
from abc import abstractmethod

from jaxtyping import Float
import tensordict
import torch
from torch import Tensor

# A 0-d tensor, e.g. a reduced loss ready for ``.backward()``.
type Scalar = Float[Tensor, ""]


class Model(torch.nn.Module, ABC):
    """A module that consumes and produces batches of named tensors.

    Subclasses implement :meth:`forward` over :class:`tensordict.TensorDictBase`
    rather than positional tensors, reading the input keys they need and
    writing their outputs under their own keys. Construction requires a
    keyword-only ``generator``, so any code that builds a model can seed it
    the same way regardless of the concrete class.
    """

    # The base consumes nothing from `generator`; the parameter exists so the
    # constructor contract is uniform across models.
    def __init__(self, *, generator: torch.Generator) -> None:  # noqa: ARG002
        """Initialize the module.

        The base class does not use ``generator``; the parameter exists so
        every :class:`Model` shares one constructor contract. A subclass with
        random parameter initialization draws from it, and one without simply
        inherits this signature.

        Args:
            generator: RNG for the subclass's initial-parameter draws.
        """
        super().__init__()

    @abstractmethod
    def forward(self, batch: tensordict.TensorDictBase) -> tensordict.TensorDictBase:
        """Run the model on a batch.

        Args:
            batch: Named input tensors for one batch.

        Returns:
            A batch carrying the model's outputs under its own keys.
        """
