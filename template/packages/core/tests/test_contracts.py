"""Tests for the Model base class."""

import pytest
import tensordict
import torch

from core import Model


class _Identity(Model):
    """Model that returns the batch unchanged."""

    def forward(self, batch: tensordict.TensorDictBase) -> tensordict.TensorDictBase:
        return batch


def test_model_implementation_is_an_nn_module() -> None:
    """A Model subclass carries the nn.Module machinery."""
    assert isinstance(_Identity(generator=torch.Generator()), torch.nn.Module)


def test_model_requires_a_generator_at_construction() -> None:
    """The constructor contract makes the generator keyword mandatory."""
    with pytest.raises(TypeError, match="generator"):
        _Identity()  # ty: ignore[missing-argument]


def test_model_call_routes_through_module_hooks() -> None:
    """Calling a Model dispatches via nn.Module.__call__, so forward hooks fire."""
    model = _Identity(generator=torch.Generator())
    outputs = []
    model.register_forward_hook(lambda _module, _args, output: outputs.append(output))
    batch = tensordict.TensorDict({"x": torch.zeros(2)}, batch_size=[2])
    result = model(batch)
    assert result is batch
    assert len(outputs) == 1
    assert outputs[0] is batch
