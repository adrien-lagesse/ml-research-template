"""Abstract base for turning datasets into configured dataloaders.

``Data`` stores every ``torch.utils.data.DataLoader`` setting once. A subclass
supplies a train dataset and named eval datasets; the base class builds one
loader per dataset, all sharing that configuration. Each loader owns its own
seeded ``torch.Generator``, so iterating one loader never perturbs another.
"""

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any
from typing import Required
from typing import TypedDict
from typing import Unpack

import tensordict
import torch
import torch.utils.data


class _TorchLoaderKwargs(TypedDict, total=False):
    """The ``Data.__init__`` keys forwarded verbatim to every ``DataLoader``.

    Each key means what it means on ``torch.utils.data.DataLoader``. Only
    ``batch_size`` is required.
    """

    batch_size: Required[int]
    num_workers: int
    pin_memory: bool
    timeout: float
    worker_init_fn: Callable[[int], None] | None
    multiprocessing_context: str | None
    prefetch_factor: int | None
    persistent_workers: bool
    pin_memory_device: str
    in_order: bool


class LoaderKwargs(_TorchLoaderKwargs, total=False):
    """Keyword arguments accepted by ``Data.__init__``.

    ``dataloader_seed``, ``shuffle_train``, and ``drop_last`` are interpreted
    by ``Data`` itself; every other key is stored once and forwarded verbatim
    to each ``torch.utils.data.DataLoader``, with the meaning documented
    there. Only ``batch_size`` is required; a missing key falls back to its
    default.
    """

    dataloader_seed: int
    shuffle_train: bool
    drop_last: bool


class TensorDictDataset(torch.utils.data.Dataset[tensordict.TensorDictBase]):
    """A batched ``TensorDict`` exposed as a map-style dataset.

    Integer indexing yields one batchless sample; the ``__getitems__`` fast
    path lets a ``DataLoader`` slice a whole batch out of the underlying
    storage in one advanced-indexing copy, with no per-sample objects.
    """

    def __init__(self, data: tensordict.TensorDictBase) -> None:
        """Wrap ``data``, whose single batch dimension indexes the samples."""
        assert data.batch_dims == 1, f"expected one batch dim, got {data.batch_size}"
        self._data = data

    def __len__(self) -> int:
        """Return the number of samples."""
        return self._data.batch_size[0]

    def __getitem__(self, index: int) -> tensordict.TensorDictBase:
        """Return sample ``index`` as a batchless ``TensorDict``."""
        sample = self._data[index]
        assert isinstance(sample, tensordict.TensorDictBase)
        return sample

    def __getitems__(self, indices: list[int]) -> tensordict.TensorDictBase:
        """Return the samples at ``indices`` as one batched ``TensorDict``."""
        batch = self._data[indices]
        assert isinstance(batch, tensordict.TensorDictBase)
        return batch


class Data(ABC):
    """Datasets plus the dataloader settings needed to iterate them.

    Subclasses provide a training dataset plus named eval datasets. The base
    class turns those into ``DataLoader``s sharing one configuration. Every
    loader gets its own ``torch.Generator`` seeded with ``dataloader_seed``,
    so the training shuffle order is reproducible and independent of when or
    how often other loaders run.

    The train loader shuffles according to ``shuffle_train`` and honors
    ``drop_last``; eval loaders never shuffle and always keep the final
    partial batch.
    """

    def __init__(self, **loader_kwargs: Unpack[LoaderKwargs]) -> None:
        """Store the dataloader configuration shared by all loaders.

        Args:
            **loader_kwargs: Loader settings; see ``LoaderKwargs``.
                ``batch_size`` is required. ``dataloader_seed`` (default 0)
                seeds each loader's private generator; ``shuffle_train``
                (default True) reshuffles the training set each epoch;
                ``drop_last`` (default True) drops the final incomplete
                training batch. The remaining keys forward verbatim to
                ``torch.utils.data.DataLoader``.
        """
        self._dataloader_seed = loader_kwargs.pop("dataloader_seed", 0)
        self._shuffle_train = loader_kwargs.pop("shuffle_train", True)
        self._drop_last = loader_kwargs.pop("drop_last", True)
        self._loader_kwargs: _TorchLoaderKwargs = loader_kwargs

    @abstractmethod
    def train_dataset(self) -> torch.utils.data.Dataset[Any]:
        """Return the dataset to train on."""

    @abstractmethod
    def eval_datasets(self) -> Mapping[str, torch.utils.data.Dataset[Any]]:
        """Return the datasets metrics are computed on, keyed by name.

        The keys become the keys of ``eval_dataloaders``. Return an empty
        mapping when there is nothing to evaluate on.
        """

    def collate_fn(
        self, samples: list[Any] | tensordict.TensorDictBase
    ) -> tensordict.TensorDictBase:
        """Assemble raw dataset items into one batch.

        The default covers datasets that yield ``TensorDict`` samples: a list
        of them is stacked along a new leading batch dimension, and an
        already-batched ``TensorDict`` (from a dataset with ``__getitems__``,
        like ``TensorDictDataset``) passes through unchanged. Override it for
        datasets that yield anything else.

        Args:
            samples: The items of one batch, straight from the dataset.

        Returns:
            The samples as one tensordict batch.
        """
        if isinstance(samples, tensordict.TensorDictBase):
            return samples
        return tensordict.stack(samples, dim=0)

    def train_dataloader(
        self,
    ) -> torch.utils.data.DataLoader[tensordict.TensorDictBase]:
        """Build the loader over ``train_dataset``.

        Shuffling follows ``shuffle_train`` and dropping the last incomplete
        batch follows ``drop_last``, both fixed at construction. Every call
        returns a fresh loader with a freshly seeded generator, so the
        shuffle sequence restarts identically each time.
        """
        return self._dataloader(
            self.train_dataset(), shuffle=self._shuffle_train, drop_last=self._drop_last
        )

    def eval_dataloaders(
        self,
    ) -> dict[str, torch.utils.data.DataLoader[tensordict.TensorDictBase]]:
        """Build one loader per eval dataset, keyed as in ``eval_datasets``.

        Eval loaders never shuffle and keep the final partial batch.
        """
        return {
            name: self._dataloader(dataset, shuffle=False, drop_last=False)
            for name, dataset in self.eval_datasets().items()
        }

    def _dataloader(
        self, dataset: torch.utils.data.Dataset[Any], *, shuffle: bool, drop_last: bool
    ) -> torch.utils.data.DataLoader[tensordict.TensorDictBase]:
        """Build a loader over ``dataset`` with the stored settings.

        The loader receives its own generator seeded with
        ``dataloader_seed``: loaders never share RNG state, so iterating one
        cannot change what another yields.
        """
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
            generator=torch.Generator().manual_seed(self._dataloader_seed),
            **self._loader_kwargs,
        )
