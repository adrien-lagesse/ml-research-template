"""Tests for the Data base class and TensorDictDataset."""

from collections.abc import Mapping
from typing import Any
from typing import Unpack

import tensordict
import torch
import torch.utils.data

from core import Data
from core import LoaderKwargs
from core import TensorDictDataset


def _samples(n: int) -> tensordict.TensorDict:
    """``n`` distinguishable samples under an ``x`` key."""
    generator = torch.Generator().manual_seed(0)
    return tensordict.TensorDict(
        {"x": torch.randn(n, 2, generator=generator)}, batch_size=[n]
    )


class _ToyData(Data):
    """Data over fixed in-memory splits, for exercising the loader plumbing."""

    def __init__(self, **loader_kwargs: Unpack[LoaderKwargs]) -> None:
        super().__init__(**loader_kwargs)
        self._train = TensorDictDataset(_samples(16))
        self._val = TensorDictDataset(_samples(8))
        self._test = TensorDictDataset(_samples(8))

    def train_dataset(self) -> torch.utils.data.Dataset[Any]:
        return self._train

    def eval_datasets(self) -> Mapping[str, torch.utils.data.Dataset[Any]]:
        return {"validation": self._val, "test": self._test}


def test_tensordict_dataset_indexes_single_samples() -> None:
    """Integer indexing yields the matching batchless sample."""
    data = _samples(4)
    dataset = TensorDictDataset(data)
    assert len(dataset) == 4
    torch.testing.assert_close(dataset[2]["x"], data["x"][2])


def test_tensordict_dataset_getitems_returns_one_batch() -> None:
    """The batched fast path slices all requested samples at once."""
    data = _samples(4)
    batch = TensorDictDataset(data).__getitems__([1, 3])
    assert batch.batch_size == torch.Size([2])
    torch.testing.assert_close(batch["x"], data["x"][[1, 3]])


def test_default_collate_yields_tensordict_batches() -> None:
    """A loader over TensorDict samples yields batched TensorDicts."""
    batch = next(iter(_ToyData(batch_size=4).train_dataloader()))
    assert isinstance(batch, tensordict.TensorDictBase)
    assert batch["x"].shape == (4, 2)


def test_train_dataloader_shuffle_is_reproducible() -> None:
    """Two loaders from the same instance replay the same shuffle sequence."""
    data = _ToyData(batch_size=4)
    first = torch.cat([batch["x"] for batch in data.train_dataloader()])
    second = torch.cat([batch["x"] for batch in data.train_dataloader()])
    torch.testing.assert_close(first, second)


def test_eval_passes_do_not_change_the_train_shuffle() -> None:
    """Sweeping eval loaders between epochs leaves the train order alone."""

    def two_epochs(sweep_evals: bool) -> tuple[torch.Tensor, torch.Tensor]:
        data = _ToyData(batch_size=4)
        loader = data.train_dataloader()
        first = torch.cat([batch["x"] for batch in loader])
        if sweep_evals:
            for eval_loader in data.eval_dataloaders().values():
                for _ in eval_loader:
                    pass
        second = torch.cat([batch["x"] for batch in loader])
        return first, second

    quiet_first, quiet_second = two_epochs(sweep_evals=False)
    swept_first, swept_second = two_epochs(sweep_evals=True)
    torch.testing.assert_close(quiet_first, swept_first)
    torch.testing.assert_close(quiet_second, swept_second)
