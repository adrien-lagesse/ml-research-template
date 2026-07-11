"""Synthetic linear-regression data; `SyntheticRegression` is the entry point."""

from collections.abc import Mapping
from typing import Any
from typing import NotRequired
from typing import Unpack

import einops
from jaxtyping import Float
import tensordict
import torch
from torch import Tensor
import torch.utils.data

from core import Data
from core import LoaderKwargs

# Exclusive upper bound for the per-split seeds drawn from the master
# generator; any large range works, it only has to be fixed.
_SEED_RANGE = 2**62

# One raw sample as the dataset yields it: an (input, target) pair.
type _RegressionSample = tuple[Float[Tensor, "n_features"], Float[Tensor, "n_targets"]]


class RegressionBatch(tensordict.TypedTensorDict):
    """A batch of regression samples with typed entries.

    ``input`` and ``target`` come from the collate; ``prediction`` is absent
    until a model writes it. Being a ``TensorDictBase``, it flows through the
    ``core`` contracts like any other batch.
    """

    input: Float[Tensor, "batch n_features"]
    target: Float[Tensor, "batch n_targets"]
    prediction: NotRequired[Float[Tensor, "batch n_targets"]]  # ty: ignore[invalid-type-form]


class _RegressionDataset(torch.utils.data.Dataset[_RegressionSample]):
    """In-memory inputs and targets served one ``(input, target)`` tuple at a time."""

    def __init__(
        self,
        inputs: Float[Tensor, "n_samples n_features"],
        targets: Float[Tensor, "n_samples n_targets"],
    ) -> None:
        """Wrap the two aligned sample tensors."""
        assert len(inputs) == len(targets), (
            f"{len(inputs)} inputs, {len(targets)} targets"
        )
        self._inputs = inputs
        self._targets = targets

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._inputs)

    def __getitem__(self, index: int) -> _RegressionSample:
        """Return sample ``index`` as an ``(input, target)`` tuple."""
        return self._inputs[index], self._targets[index]


class SyntheticRegression(Data):
    """Regression data generated from one random linear map.

    Inputs are standard normal. Targets are ``inputs @ weight + bias`` plus
    Gaussian noise with standard deviation ``noise_std``, where ``weight``
    and ``bias`` are drawn once and shared by the train, validation, and
    test splits. Each split is drawn from its own generator, so its samples
    depend only on ``dataset_seed`` and its own size, never on the other
    splits' sizes. All samples are generated eagerly and held in memory.

    The datasets yield raw ``(input, target)`` tuples; ``collate_fn``
    assembles them into `RegressionBatch` batches.
    """

    def __init__(
        self,
        *,
        n_features: int,
        n_targets: int,
        n_train: int,
        n_val: int,
        n_test: int,
        noise_std: float,
        dataset_seed: int,
        **loader_kwargs: Unpack[LoaderKwargs],
    ) -> None:
        """Draw the linear map and generate all three splits.

        Args:
            n_features: Input dimensionality.
            n_targets: Target dimensionality.
            n_train: Number of training samples.
            n_val: Number of validation samples.
            n_test: Number of test samples.
            noise_std: Standard deviation of the additive Gaussian noise on
                the targets.
            dataset_seed: Seed for the generator that draws the linear map
                and the per-split seeds. Separate from the
                ``dataloader_seed`` in ``loader_kwargs``, which only controls
                batch shuffling.
            **loader_kwargs: Dataloader settings forwarded to ``Data``;
                ``batch_size`` is required.
        """
        super().__init__(**loader_kwargs)

        generator = torch.Generator().manual_seed(dataset_seed)
        weight = torch.randn(n_features, n_targets, generator=generator)
        bias = torch.randn(n_targets, generator=generator)
        # One seed per split, drawn in a fixed order, so each split's samples
        # are independent of the other splits' sizes.
        split_seeds = torch.randint(_SEED_RANGE, (3,), generator=generator)

        def _sample(n_samples: int, split_seed: int) -> _RegressionDataset:
            """Draw one split of normal inputs and their noisy linear targets."""
            split_generator = torch.Generator().manual_seed(split_seed)
            inputs = torch.randn(n_samples, n_features, generator=split_generator)
            clean = einops.einsum(inputs, weight, "n f, f t -> n t") + bias
            noise = noise_std * torch.randn(
                n_samples, n_targets, generator=split_generator
            )
            return _RegressionDataset(inputs, clean + noise)

        self._train = _sample(n_train, int(split_seeds[0]))
        self._val = _sample(n_val, int(split_seeds[1]))
        self._test = _sample(n_test, int(split_seeds[2]))

    def train_dataset(self) -> torch.utils.data.Dataset[_RegressionSample]:
        """Return the training split."""
        return self._train

    def eval_datasets(
        self,
    ) -> Mapping[str, torch.utils.data.Dataset[_RegressionSample]]:
        """Return the splits keyed ``"validation"`` and ``"test"``."""
        return {"validation": self._val, "test": self._test}

    def collate_fn(
        self, samples: list[Any] | tensordict.TensorDictBase
    ) -> RegressionBatch:
        """Stack raw ``(input, target)`` tuples into one `RegressionBatch`.

        Args:
            samples: The items of one batch, straight from the dataset. The
                datasets here have no ``__getitems__``, so this is always a
                list of tuples, asserted as such.

        Returns:
            One typed batch with stacked ``input`` and ``target`` entries.
        """
        assert isinstance(samples, list)
        inputs, targets = zip(*samples, strict=True)
        return RegressionBatch(  # ty: ignore[missing-argument]
            input=torch.stack(inputs),
            target=torch.stack(targets),
            batch_size=[len(samples)],  # ty: ignore[unknown-argument]
        )
