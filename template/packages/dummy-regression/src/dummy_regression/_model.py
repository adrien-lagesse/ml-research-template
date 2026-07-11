"""A small MLP regressor implementing the ``core.Model`` batch contract."""

from jaxtyping import Float
import tensordict
import torch
from torch import Tensor

from core import Model
from core import seed_parameters


class MLP(Model):
    """A two-layer perceptron with a ReLU in between.

    Reads the ``"input"`` key from a batch and writes its output under
    ``"prediction"``. Parameters are initialized from the generator passed at
    construction, so two models built with the same sizes and generator state
    have identical weights.
    """

    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int,
        out_features: int,
        generator: torch.Generator,
    ) -> None:
        """Build the network and deterministically initialize its parameters.

        Chains to ``core.Model``, then seeds every weight and bias through
        ``core.seed_parameters``, so all initial-parameter draws come from
        ``generator`` and none from PyTorch's global RNG. The network is built
        on CPU and the caller moves the whole model to its device afterward.

        Args:
            in_features: Size of the last axis of the batch's ``"input"``
                tensor.
            hidden_features: Width of the single hidden layer.
            out_features: Size of the last axis of the ``"prediction"`` tensor
                the model writes.
            generator: Source of all initial-parameter draws; two models built
                with the same sizes and generator state get identical weights.
        """
        super().__init__(generator=generator)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features),
        )
        seed_parameters(self.net, generator)

    def forward(self, batch: tensordict.TensorDictBase) -> tensordict.TensorDictBase:
        """Predict on a batch, writing the result back into it.

        Args:
            batch: Must carry ``"input"``, shape ``(batch, in_features)``.

        Returns:
            The same batch, mutated in place with ``"prediction"`` of shape
            ``(batch, out_features)``.
        """
        inputs: Float[Tensor, "batch in_features"] = batch.get("input")
        prediction: Float[Tensor, "batch out_features"] = self.net(inputs)
        batch["prediction"] = prediction
        return batch
