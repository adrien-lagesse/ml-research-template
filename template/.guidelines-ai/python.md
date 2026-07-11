# Python guidelines

Conventions for Python in this research project. The goal is code that is
readable, testable, and reproducible, where a reader can tell what a value is
and what a function does without running it.

Priorities, in order: clarity > correctness-by-construction > reproducibility >
brevity. This is research code: fail loudly, don't build defensive scaffolding.

## Naming

- **Variables**: `snake_case`, descriptive nouns. The name should hint at the
  type and unit. `n_epochs`, `learning_rate`, `user_ids`, `image_batch`.
- **Plurals for collections**: `samples` is a list/array; `sample` is one item.
- **Booleans** read as predicates: `is_trained`, `has_bias`, `should_cache`.
- **Classes**: `PascalCase` nouns, like `KernelRidge`, `DataLoader`. No `Manager`,
  `Helper`, `Util` grab-bags; name it for what it is.
- **Constants**: `UPPER_SNAKE` at module level, like `DEFAULT_SEED = 0`.
- **Private**: prefix with `_`, like `_scratch`, `_compute_gram`. The `_` is the
  signal that nothing outside the module should touch it.
- Avoid one-letter names except short math scopes (`i`, `x`, `y`) where the
  meaning is local and obvious.

## Typing

Every function is fully typed: parameters and return. No exceptions.

- Use **modern builtin generics**: `list[int]`, `dict[str, float]`,
  `tuple[int, ...]`. Not `List`, `Dict` from `typing`.
- Use `X | None`, not `Optional[X]`. Use `A | B`, not `Union[A, B]`.
- Reach into `typing` only for what builtins can't express: `Protocol`,
  `TypeVar`, `Callable`, `Literal`, `TypeAlias`, `Any`.
- Annotate variables when the type isn't obvious from the right-hand side. A
  reader should never have to guess.

```python
from collections.abc import Callable

def train(
    model: Model,
    data: list[Sample],
    *,
    lr: float = 1e-3,
    n_epochs: int = 100,
    on_epoch: Callable[[int, float], None] | None = None,
) -> TrainResult:
    ...
```

- Type aliases for signatures that repeat or carry meaning. Alias what's
  verbose or domain-specific, not a plain rename of `torch.Tensor`:

```python
from collections.abc import Callable
from typing import TypeAlias

import torch

# A kernel: maps two batches of samples to their pairwise scores.
Kernel: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def gram_matrix(x: torch.Tensor, kernel: Kernel) -> torch.Tensor:
    ...
```

- A value's type should be apparent at its definition site. If a name gets
  reassigned to a different type, use two names instead.

## Docstrings

Every module, package, class, and function has a docstring. **Google style.**
Docstrings are for the reader: what the thing does, its args, what it returns,
what it assumes. They are not a scratchpad: when I ask for a change, don't leave
notes-to-self, changelog lines, or "modified to…" comments in them.

```python
from collections.abc import Callable

from jaxtyping import Float
from torch import Tensor


def gram_matrix(
    x: Float[Tensor, "n_samples n_features"],
    kernel: Callable[[Float[Tensor, "n_features"], Float[Tensor, "n_features"]], float],
) -> Float[Tensor, "n_samples n_samples"]:
    """Compute the pairwise kernel matrix.

    Args:
        x: Samples, shape ``(n_samples, n_features)``.
        kernel: Symmetric positive-definite kernel ``k(a, b) -> float``.

    Returns:
        The Gram matrix ``K`` with ``K[i, j] = kernel(x[i], x[j])``,
        shape ``(n_samples, n_samples)``.

    Raises:
        ValueError: If ``x`` is not 2-D.

    Example:
        >>> import torch
        >>> x = torch.tensor([[0.0], [1.0]])
        >>> gram_matrix(x, lambda a, b: float(a @ b))
        tensor([[0., 0.],
                [0., 1.]])
    """
```

- Module docstring: one line on what the module is for, then any key
  assumptions or entry points.
- Skip the obvious. A one-line `__repr__` doesn't need a five-line docstring.
- **Say what the signature can't, not the signature back.** jaxtyping already
  gives the shape and dtype; the parameter names give the rest. `logits: the
  logits tensor` earns nothing. Spend the prose on what the type can't carry:
  that `logits` are pre-softmax, that `temperature` must be positive, that the
  loss assumes L2-normalized embeddings. In `gram_matrix`, `kernel` being
  symmetric positive-definite and `x` being 2-D are the load-bearing facts.
- **Document only what's there.** Every `Args` entry names a real parameter,
  `Returns` matches what the function returns, every `Raises` names an exception
  the body can actually raise. A documented `temperature` the signature dropped,
  or a `Raises: ValueError` for a check that's gone, is worse than silence: ty
  won't flag it and a reader will trust it.
- **Update the docstring in the same edit as the code.** Change a tensor's
  shape, a default, or the reduction, and its docstring changes in the same
  diff. Never reword a docstring to match a result you suspect is a bug; fix the
  bug. A docstring that promises a scalar over a function that now returns
  per-sample losses is a trap.

- **Write the docstring from the code, not from the conversation that produced
  it.** A docstring describes what the thing is and does. An agent holding the
  whole session in context writes something else: the thinking leaks in, and the
  docstring narrates the path we took rather than the code that landed. "Samples
  come from `target = input @ w + b + noise` … so a model has a real linear
  signal to recover … everything is seeded through explicit generators" reads as
  a design rationale, not an interface. So when you change a function, discard
  its old docstring and regenerate it cold: spawn a fresh sub-agent, hand it only
  that function and the types it references, and have it write the docstring with
  none of this conversation in view. For a module docstring, do this for the
  module docstring alone, from the module; you don't regenerate every docstring
  inside it. The one exception is a docstring I ask you to word a specific way,
  where my instruction is the spec and you follow it as given.

## Comments

Code says how; a comment says why. The reader can already see the how from the
tensors and the einops patterns. Spend a comment on what the code can't show on
its own.

- **Explain intent or a non-obvious choice**, not the mechanics. `# rearrange to
  b (h w) c` over a `rearrange` that already says it is noise. `# subtract the
  row max before exp so the softmax doesn't overflow in fp16` earns its line.
- **Don't narrate the change.** `# was reduce(..., "mean") before` and `#
  switched to einsum to fix the transpose` describe a diff, not the code, and go
  stale the next time the line moves. Git holds the history.
- **No comment on the obvious.** `# move the batch to the device` over
  `.to(device)` restates the call. If the code is clear, delete the comment
  rather than echo it.
- **A comment is a promise to maintain.** When its line changes, the comment
  changes too, or it becomes a lie. A few load-bearing comments (a
  numerical-stability trick, a pointer to the paper's equation) beat blanket
  coverage.

## Functions and side effects

- **Small and single-purpose.** If a function needs a paragraph to explain,
  split it.
- **Prefer pure functions**: output depends only on input, no mutation of
  arguments, no I/O. These are the testable, reproducible core.
- **Isolate side effects** (file/network I/O, logging, RNG, global state) at
  the edges. Keep them out of the computational core so logic stays
  deterministic.

```python
# Pure core: same inputs, same outputs, always.
def split_indices(
    n: int, frac: float, generator: torch.Generator
) -> tuple[Int[Tensor, "n_train"], Int[Tensor, "n_test"]]:
    assert 0.0 <= frac <= 1.0, f"frac out of range: {frac}"
    perm = torch.randperm(n, generator=generator)
    cut = int(n * frac)
    return perm[:cut], perm[cut:]

# Side effects live at the edge, passed in explicitly.
def run(path: Path, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)  # RNG created once, here
    data = load(path)                                 # I/O here
    train_idx, test_idx = split_indices(len(data), 0.8, generator)  # pure
    logging.info("split: %d train, %d test", len(train_idx), len(test_idx))
```

- Pass RNG as an explicit `torch.Generator`. Never rely on the global RNG or
  call `torch.manual_seed` inside library code.
- No mutable default arguments (`def f(x: list = [])`). Use `None` and create
  inside.

## Errors and validation

Fail loudly, fail early. This is research code, so surface broken assumptions
immediately instead of limping along with bad state.

- **`assert` to check our own assumptions**: shapes, invariants, things that
  should be true if the code is correct. Cheap, self-documenting, removable.

```python
assert x.ndim == 2, f"expected 2-D input, got shape {x.shape}"
assert 0.0 <= frac <= 1.0, f"frac out of range: {frac}"
```

- **`raise` for things outside our control**: bad user input, missing files,
  malformed external data. Use a specific builtin (`ValueError`, `FileNotFoundError`).

```python
if not path.exists():
    raise FileNotFoundError(f"dataset not found: {path}")
```

- **No defensive scaffolding.** No retry loops, fallbacks, or custom exception
  hierarchies for a research project. Don't catch what you can't meaningfully
  handle; let it crash with a real traceback.

## Package and module layout

A package's public API is **hand-curated**. Users see only what we choose to
expose, never internal helpers or implementation imports.

- `_<name>.py` holds the real code. The leading `_` marks it private.
- `__init__.py` is the **public facade**: no logic, only re-exports and
  `__all__`. It decides what the package's name means to a user.
- Submodules are nested packages with their own `__init__.py` facade.
- **Never** import an internal/utility module into a public namespace.

```
mypkg/
    __init__.py        # public facade
    _model.py          # implementation
    _kernels.py        # implementation
    io/
        __init__.py    # public facade for mypkg.io
        _loaders.py
```

```python
# mypkg/__init__.py: facade only, no logic
"""Kernel models for the experiments."""

from mypkg._model import KernelRidge, TrainResult
from mypkg._kernels import rbf, linear

__all__ = ["KernelRidge", "TrainResult", "rbf", "linear"]
```

A user writes `from mypkg import KernelRidge`. They never reach into
`mypkg._model`, and `_compute_gram` never appears in the public namespace.

## Code hygiene

- **No dead code**: no commented-out blocks, unused imports, or unreachable
  branches. Git is the history; if you might want it back, it's already in the
  log.
- **Promote stable code out of throwaway.** Don't let logic live in shell
  history or `python -c` one-liners. Once it works and matters, it belongs in a
  module where it can be imported, tested, and reproduced.
- One module, one concern. When a `_file.py` grows two unrelated jobs, split it.
- Use `pathlib.Path` over `os.path`, f-strings over `%`/`.format`, `dataclass`
  (or `frozen=True`) over ad-hoc dicts for structured records.

## Quick checklist before committing

- Every function typed? Every public name documented?
- Could a reader tell each variable's type from its name and annotation?
- Is the compute core pure, with I/O / logging / RNG pushed to the edges?
- Asserts guarding our assumptions; explicit raises for external failures?
- Does `__init__.py` expose only the curated API, with `__all__` set?
- Any commented-out code, unused imports, or logic stranded in a one-liner?
