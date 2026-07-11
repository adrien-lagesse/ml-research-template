# Testing

How we test in this project. Tests exist to pin down the behavior we rely on and
to catch regressions when the code moves under us. Builds on `python.md` (pure
cores, typing, fail-loud) and `ml-pytorch.md` (explicit `torch.Generator`,
jaxtyping shapes). Config lives in `pytest.ini`; run everything through `uv run`.

## What to test

Test the pure compute core. The functions that take values and return values,
with no I/O and no global state, are the ones worth pinning down. They're
deterministic, so a test that passes once keeps passing for a real reason.

Test contracts and invariants, not the implementation. A test should say what
the code promises ("normalizing gives unit norm", "the split covers every
index"), not retrace how it does it. When you rewrite the body but keep the
promise, the test should stay green.

This is research code, so aim for a few sharp tests of the math over blanket
coverage of glue. A loader that just forwards to `torch.load` doesn't need a
test; the loss function does. Don't chase a coverage number; a line that ran
isn't a line that was checked.

Test through the public API. Import what `__init__.py` exposes and exercise the
package the way a user would. If a private `_helper` is worth a direct test, that
usually means it wants to be public, or its caller is under-tested.

## Layout and naming

Each workspace member owns its tests, in a `tests/` directory beside `src/`:

```
packages/mypkg/
    src/mypkg/
        __init__.py
        _kernels.py
    tests/
        test_kernels.py
        conftest.py        # shared fixtures, optional
```

`--import-mode=importlib` (set in `pytest.ini`) means the `tests/` directory
needs no `__init__.py`. Mirror the module under test: `_kernels.py` is tested by
`test_kernels.py`. Name test functions for the claim they make, so a failure
reads as a broken promise:

```python
def test_rbf_is_one_on_the_diagonal() -> None: ...
def test_l2_normalize_gives_unit_norm() -> None: ...
def test_split_indices_covers_every_index() -> None: ...
```

## Anatomy of a test

Arrange, act, assert. One behavior per test, a docstring stating the claim, and
plain `assert`, which pytest rewrites to show both sides on failure, so you never
need `assertEqual`. Test code skips the `D`/`ANN` rules (see `ruff.toml`), so a
one-line docstring and untyped locals are fine; the test function still takes
`-> None`.

```python
from mypkg import hello


def test_hello_returns_greeting() -> None:
    """`hello` returns the canonical greeting string."""
    assert hello() == "hello world"
```

Keep the arrange step small. If setting up an input takes twenty lines, the
function under test is probably doing too much, or the setup belongs in a
fixture.

## Parametrize over cases

When the same logic should hold across several inputs, use
`@pytest.mark.parametrize` instead of copy-pasting the test or looping inside it.
Each case reports as its own test, and `ids=` gives each a readable name in the
output.

```python
import pytest

from mypkg import rbf


@pytest.mark.parametrize(
    ("length_scale", "expected"),
    [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0)],
    ids=["unit", "narrow", "wide"],
)
def test_rbf_is_one_on_the_diagonal(length_scale: float, expected: float) -> None:
    """`rbf(x, x)` is 1 on the diagonal regardless of length scale."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(4, 3, generator=generator)
    diagonal = torch.diagonal(rbf(x, x, length_scale=length_scale))
    torch.testing.assert_close(diagonal, torch.full_like(diagonal, expected))
```

A loop hides which case failed behind a single red test; parametrize tells you
`narrow` broke and `wide` didn't.

## Property-based testing with hypothesis

For numerical code you often know a property the output must satisfy without
knowing the answer for any given input. That's what `hypothesis` is for: you
state the property, it generates many inputs trying to break it, and when one
fails it shrinks the case to the smallest version that still fails and replays it
on every later run.

```python
import torch
from hypothesis import given
from hypothesis import strategies as st

from mypkg import l2_normalize


@given(batch=st.integers(1, 64), dim=st.integers(1, 128))
def test_l2_normalize_always_gives_unit_norm(batch: int, dim: int) -> None:
    """Rows come out with norm 1 for any shape."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(batch, dim, generator=generator)
    out = l2_normalize(x)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(batch))
```

Good properties to reach for: a transform and its inverse round-trip; an output
shape tracks the input shape; a normalization lands in its expected range; a
result is invariant to an order or a permutation it shouldn't depend on. Reserve
hypothesis for genuine invariants. When a function has one right answer for one
input, a plain parametrized test is clearer.

## Comparing tensors

Never `==` on floating-point tensors; rounding makes mathematically equal
results compare unequal. Use `torch.testing.assert_close`, which checks
`|a - b| <= atol + rtol * |b|` and reports the worst offending element:

```python
torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-7)
```

`atol` dominates near zero, where any relative difference looks huge; `rtol`
dominates for large values. Set them to the precision the computation actually
warrants rather than loosening until the test passes; a tolerance of `1e-1` is
usually a bug you're hiding. `assert_close` also checks dtype and device by
default, so a test catches a stray `.float()` or a tensor left on the wrong
device.

## Reproducibility in tests

The RNG rule from `ml-pytorch.md` holds in tests too: build a
`torch.Generator().manual_seed(...)` inside the test and pass it into whatever
needs randomness. Never touch the global RNG and never call `torch.manual_seed`.
A test whose result depends on global seed state is flaky by construction, and a
flaky test is a broken test; fix the seeding, don't add a retry.

## Fixtures and conftest.py

A fixture supplies a test with something it needs: a built model, a small
dataset, a tmp directory. Request it by putting its name in the test's
parameters.

```python
import pytest

import torch


@pytest.fixture
def small_batch() -> torch.Tensor:
    """A fixed 8x16 batch for shape and smoke tests."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(8, 16, generator=generator)


def test_encoder_preserves_batch_size(small_batch: torch.Tensor) -> None:
    """The encoder keeps the batch dimension."""
    assert encode(small_batch).shape[0] == small_batch.shape[0]
```

Default to function scope so each test gets a fresh value and tests can't leak
state into each other. Widen to `module` or `session` scope only for setup that
is both expensive and read-only. Put a fixture used across files in
`conftest.py`; pytest finds it automatically, with no import.

## Markers

Two markers are registered in `pytest.ini`. `slow` is for tests that run a real
training loop or a full integration path; `gpu` is for tests that need CUDA. Mark
them so the fast loop can skip them:

```python
@pytest.mark.slow
def test_trains_to_convergence() -> None: ...


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_kernel_matches_cpu_on_gpu() -> None: ...
```

`--strict-markers` means an unregistered marker is an error, so add new ones to
`pytest.ini` before using them. During development, `-m "not slow"` keeps the
loop quick.

## Doctests

`--doctest-modules` runs the `>>>` examples in docstrings as tests, so a code
example in the docs can't quietly go stale. Keep doctest examples small and
illustrative; they show the shape of the API, while the thorough cases live in
`tests/`. For noisy tensor output, the `ELLIPSIS` flag (on by default here) lets
you write `tensor([...])` instead of pinning every digit.

## What not to do

- Don't mock the thing under test. Research code should run for real on small
  inputs; a mock that returns a canned tensor tests the mock, not the math.
- Don't test trivial getters or one-line passthroughs. They add maintenance and
  catch nothing.
- Don't assert on private state to check a public result. Assert on what the
  function returns or raises.
- Don't catch exceptions to make a test pass. Let it fail with a real traceback;
  use `pytest.raises` only when raising is the behavior you're testing.

```python
import pytest


def test_gram_matrix_rejects_1d_input() -> None:
    """A non-2-D input raises rather than silently broadcasting."""
    with pytest.raises(ValueError, match="2-D"):
        gram_matrix(torch.zeros(3), linear)
```

## Running tests

```bash
uv run --all-packages pytest                # full suite
uv run --all-packages pytest -m "not slow"  # fast loop, skip slow tests
uv run --all-packages pytest packages/dummy-regression/tests/test_loss.py
```

`--all-packages` is required: a bare `uv run` doesn't sync the workspace members,
so imports like `from mypkg import ...` would fail. This is separate from the
commit gate (`uv run prek run`); run the suite yourself while iterating.
