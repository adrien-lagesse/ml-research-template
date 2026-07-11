"""Tests for seeded parameter initialization."""

import pytest
import torch

from core import seed_parameters


def _mlp() -> torch.nn.Sequential:
    """A small Linear-ReLU-Linear stack."""
    return torch.nn.Sequential(
        torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)
    )


def test_seed_parameters_is_reproducible_from_the_generator() -> None:
    """Identical generator states produce identical parameters."""
    first, second = _mlp(), _mlp()
    seed_parameters(first, torch.Generator().manual_seed(0))
    seed_parameters(second, torch.Generator().manual_seed(0))
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right)


def test_seed_parameters_depends_on_the_seed() -> None:
    """Different seeds produce different weights, so seeding actually happens."""
    first, second = _mlp(), _mlp()
    seed_parameters(first, torch.Generator().manual_seed(0))
    seed_parameters(second, torch.Generator().manual_seed(1))
    assert not torch.equal(next(first.parameters()), next(second.parameters()))


def test_seed_parameters_rejects_unsupported_modules() -> None:
    """A parametrized layer without a seeded scheme fails loudly."""
    with pytest.raises(NotImplementedError):
        seed_parameters(torch.nn.Conv2d(1, 1, 3), torch.Generator().manual_seed(0))
