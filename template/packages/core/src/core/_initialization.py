"""Deterministic re-initialization of module parameters from a generator."""

import math

import torch
import torch.nn.init


def seed_parameters(module: torch.nn.Module, generator: torch.Generator) -> None:
    """Re-draw every parameter of ``module`` from ``generator``, in place.

    Walks the module tree and reproduces each layer's PyTorch default
    initialization scheme, with every draw taken from ``generator`` instead
    of the global RNG, so the result is a pure function of the generator
    state and the module's shapes.

    Supported layer types: ``torch.nn.Linear`` (Kaiming-uniform weight with
    ``a=sqrt(5)``, bias uniform in ``±1/sqrt(fan_in)``, or zero when
    ``fan_in`` is 0). Any other module that owns parameters directly is
    rejected, so an unsupported layer can never silently keep its unseeded
    default initialization.

    Args:
        module: Module whose parameters are re-drawn.
        generator: Source of every random draw.

    Raises:
        NotImplementedError: If a submodule owns parameters but has no
            seeded scheme implemented here.
    """
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Linear):
            _seed_linear(submodule, generator)
        elif next(submodule.parameters(recurse=False), None) is not None:
            raise NotImplementedError(
                f"no seeded initialization for {type(submodule).__name__}"
            )


def _seed_linear(linear: torch.nn.Linear, generator: torch.Generator) -> None:
    """Apply ``nn.Linear``'s default initialization, drawn from ``generator``.

    Matches ``nn.Linear.reset_parameters``: Kaiming-uniform weight with
    ``a=sqrt(5)``, then a bias uniform in ``±1/sqrt(fan_in)``. When
    ``fan_in`` is 0 the bound is 0, so the bias is set to zero.
    """
    torch.nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=generator)
    if linear.bias is not None:
        fan_in = linear.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        torch.nn.init.uniform_(linear.bias, -bound, bound, generator=generator)
