# uv project

This project is managed **exclusively by uv**. uv owns the environment, the
lockfile, and the dependency set. Don't reach around it.

## Hard rules

- **Never** run `python <script>`, `pip install`, or `conda ...`. These bypass
  the project environment and desync it from the lockfile.
- **Never** activate `.venv` (`source .venv/bin/activate`). uv resolves and uses
  the right environment automatically for every `uv run`.
- **Never** install a dependency yourself. If you need a package, ask the user to
  add it. They run `uv add <pkg>`. Don't edit `pyproject.toml` deps or run
  `uv add` on your own.

## Running code

Everything runs through `uv run`, which syncs the environment first, then
executes:

```bash
uv run script.py              # run a script
uv run python -m mypkg.train  # run a module
```

Use `uv run python -c "..."` freely for throwaway work: testing a package's
API, inspecting data, or checking a function's output. Reach for it instead of
saving a scratch file:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python -c "from mypkg import load; d = load('data.pt'); print(d.shape, d.dtype)"
```

## Inspecting the environment

```bash
uv tree                       # full dependency tree
uv tree --package torch       # why a package is here / what it pulls in
```

Read `uv tree` when a version looks wrong or you need to know what depends on
what. Don't fix it yourself. Report it and let the user adjust deps.

## Linting, formatting, type checking

Tools: **ruff** (lint + format) and **ty** (type check). All run via `uv run`.

```bash
uv run ruff format .          # format the whole project
uv run ruff format script.py  # format one file
uv run ruff format --check .  # check formatting without writing (CI-style)
```

```bash
uv run ruff check .           # lint
uv run ruff check --fix .     # lint and apply safe fixes
uv run ruff check script.py   # lint one file
```

```bash
uv run ty check .             # type-check the project
uv run ty check src/mypkg     # type-check one package
```

Before considering a change done: `uv run ruff format <files>`, then
`uv run ruff check <files>`, then `uv run ty check <files>`. Fix what they report rather
than suppressing it. A `# noqa` or `# type: ignore` needs a real reason.
