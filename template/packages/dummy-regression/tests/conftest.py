"""Shared fixtures for the dummy_regression tests."""

from collections.abc import Callable

import pytest
import tensordict
import torch

type MakeBatch = Callable[[torch.Tensor, torch.Tensor], tensordict.TensorDict]


@pytest.fixture
def make_batch() -> MakeBatch:
    """Build a batch carrying the ``prediction`` and ``target`` keys."""

    def _make(prediction: torch.Tensor, target: torch.Tensor) -> tensordict.TensorDict:
        return tensordict.TensorDict(
            {"prediction": prediction, "target": target}, batch_size=[len(target)]
        )

    return _make
