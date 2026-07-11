# ML and PyTorch

Conventions for the modeling code: tensors carry named shapes, types declare
those shapes, and a run is reproducible from its config. Builds on `python.md`
(typing, pure cores, RNG at the edges).

## Reproducibility

- **Seed explicitly at the entry point.** Create the RNG where the run starts
  and thread it through as a `torch.Generator`. Never rely on PyTorch's default
  seeding, and never call `torch.manual_seed` deep inside library code.

```python
def main(cfg: Config) -> None:
    generator = torch.Generator().manual_seed(cfg.seed)
    data = load(cfg.data_path)
    model = build_model(cfg.model)
    train(model, data, cfg.train, generator=generator)
```

- **Surface hyperparameters and paths in config, never bury them.** A learning
  rate, a temperature, a checkpoint dir, a number of layers: these belong in a
  config object or a module constant, passed down. A value hard-coded inside a
  function is invisible to the logger and impossible to sweep, so the run can't
  be reproduced from what was recorded.

```python
# Wrong: 0.07 and 4096 are stranded inside the function.
def nt_xent(
    z1: Float[Tensor, "batch dim"], z2: Float[Tensor, "batch dim"]
) -> Float[Tensor, ""]:
    return loss(z1, z2, temperature=0.07, queue_size=4096)

# Right: they come from config, get logged, can be swept.
def nt_xent(
    z1: Float[Tensor, "batch dim"],
    z2: Float[Tensor, "batch dim"],
    *,
    temperature: float,
    queue_size: int,
) -> Float[Tensor, ""]:
    ...
```

## Devices and dtype

Code runs unchanged on `cpu`, `cuda`, and `mps`. The way to get there is to make
the device a value the code receives, not a fact it discovers.

- **Take `device` as an explicit parameter**, threaded from config like the
  `torch.Generator`. Never hard-code `"cuda"`, never call `.cuda()`, and never
  probe `torch.cuda.is_available()` deep in the code. Resolve the device **once**
  at the entry point and pass it down. That keeps `cpu` / `cuda` / `mps` a
  single config knob.

```python
def main(cfg: Config) -> None:
    device = torch.device(cfg.device)  # "cpu" | "cuda" | "mps", resolved once
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    model = build_model(cfg.model).to(device)
    train(model, data, cfg.train, device=device, generator=generator)
```

- **Create tensors on the target device**, don't build on CPU and copy. Pass
  `device=` at creation; the `.to()` afterward allocates twice and is easy to
  forget.

```python
# Wrong: allocates on CPU, then copies.
weights = torch.zeros(n, dim).to(device)

# Right: allocated where it's used.
weights = torch.zeros(n, dim, device=device)
```

- **The generator is device-bound.** A `torch.Generator` lives on a device and
  only seeds tensors created on that same device, so build it with
  `torch.Generator(device=device)`. This is the device half of the
  reproducibility rule above.

- **Derive new tensors from existing ones** so device and dtype follow for free.
  `torch.zeros_like(x)` and `x.new_empty(shape)` land on `x`'s device; reach for
  them over a bare `torch.zeros(..., device=x.device)`.

When a kernel genuinely doesn't exist on a device (some ops are CPU- or
CUDA-only on `mps`), that's an explicit, commented branch at the call site, the
exception rather than a habit of scattering `if device == ...` through the code.

## Tensor shapes: use einops

Reshaping operations are written with **einops**, with every axis named. A
named pattern documents the operation in place and fails loudly when an axis
doesn't match, instead of silently producing a wrong-but-valid shape.

Don't use `.permute`, `.transpose`, `.view`, `.reshape`, or `.flatten`. Each one
moves axes by position, so the reader has to count dimensions to know what
happened. The einops form says it outright.

```python
import einops

# flatten spatial dims (replaces .flatten / .reshape)
tokens = einops.rearrange(images, "b c h w -> b (h w) c")

# move channels (replaces .permute / .transpose)
chw = einops.rearrange(images, "b h w c -> b c h w")

# split a packed axis: the named size makes the split explicit
heads = einops.rearrange(qkv, "b n (h d) -> b h n d", h=n_heads)

# pool over space (replaces .mean(dim=(2, 3)))
pooled = einops.reduce(features, "b c h w -> b c", "mean")

# broadcast a per-sample vector to a map (replaces .expand / .repeat)
bias_map = einops.repeat(bias, "b c -> b c h w", h=height, w=width)

# attention scores: named indices beat a bare matmul with transposes
scores = einops.einsum(queries, keys, "b h i d, b h j d -> b h i j")
```

Pick axis names that mean something (`b`, `h`, `w`, `c`, `n`, `d`, `heads`), and
use the same name for the same thing across a function. When einops can't express
an op, reach for the explicit torch call, but that's the exception, and it gets
a comment saying why.

## Tensor types: use jaxtyping

Every tensor parameter and return is annotated with **jaxtyping**, giving its
dtype and named shape. **Never** annotate a tensor as plain `torch.Tensor`; that
says nothing about what's inside it. The annotation is the shape contract; a
reader sees `Float[Tensor, "batch n_classes"]` and knows the rank, the
dimensions, and the dtype without running anything.

```python
from jaxtyping import Float, Int, Bool
from torch import Tensor

def cross_entropy(
    logits: Float[Tensor, "batch n_classes"],
    targets: Int[Tensor, "batch"],
) -> Float[Tensor, ""]:
    """Mean cross-entropy. Empty shape "" is a scalar."""
    ...

def attention(
    queries: Float[Tensor, "batch heads seq d_head"],
    keys: Float[Tensor, "batch heads seq d_head"],
    values: Float[Tensor, "batch heads seq d_head"],
    mask: Bool[Tensor, "batch heads seq seq"] | None = None,
) -> Float[Tensor, "batch heads seq d_head"]:
    ...
```

The named dimensions are part of the contract. Reuse a name to assert two axes
match: in `attention`, every `seq` and every `d_head` is the same size, and
`batch` ties the inputs to the output. Use `*batch` for a leading group of
arbitrary dims when a function is rank-agnostic:

```python
def l2_normalize(
    x: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim"]:
    """Normalize the last axis; any number of leading dims."""
    ...
```

Annotate everything that carries a tensor: function parameters, returns, and
dataclass fields holding tensors. The shape names you pick here should match the
einops axis names in the body, so the type and the operations tell one story.

## Imports: reach through the module

Import a third-party object's module and qualify the name where you use it.
Write `import einops` and call `einops.rearrange(...)`; write `import torch` and
reference `torch.utils.data.DataLoader`. A qualified `einops.reduce` says where
it came from at the point you read it. A bare `reduce`, pulled out with `from
einops import reduce`, sends the reader back to the import block to place it. So
don't write `from einops import einsum` or `from torch.utils.data import
Dataset`.

Import the name directly in three cases, where qualifying it adds noise instead
of clarity:

- **jaxtyping.** `from jaxtyping import Float, Int, Bool`, and the tensor type
  they wrap, `from torch import Tensor`. `Float[Tensor, "b c"]` is the idiom;
  `Float[torch.Tensor, "b c"]` only clutters it.
- **The Python standard library.** Your judgment. `from typing import Any,
  Protocol`, `from collections.abc import Callable, Mapping`, `from pathlib
  import Path`: nobody wonders where `Any` or `Path` lives. Qualify a stdlib
  name only when it reads as ambiguous on its own.
- **Objects defined in this repo.** `from core import Data`, `from
  core._contracts import Batch`. The package facade exists to be imported from,
  and a reader can open the definition in the tree.

```python
# Avoid: rearrange / DataLoader / einsum in the body don't say where they're from.
from einops import einsum, rearrange
from torch.utils.data import DataLoader, Dataset

# Prefer: the module travels with the name.
import einops
import torch

tokens = einops.rearrange(images, "b c h w -> b (h w) c")
loader = torch.utils.data.DataLoader(dataset)
```
