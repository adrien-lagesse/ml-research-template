# Git and GitHub

How we commit and open pull requests. We follow the Angular commit convention,
and commits and PRs share the same title and body format.

## Hard rules

- **No AI attribution.** Don't add `Co-Authored-By: Claude`, "Generated with…"
  trailers, or any "this was written by an agent" note to commit messages or PR
  descriptions. The message is about the change, not the tool.
- **Always confirm before committing or opening a PR.** Show what you're about to
  do and wait for a yes. Never commit or push on your own initiative.
- **Run the hooks before committing**: `uv run prek run`. Hooks are defined in
  `prek.toml`.

## Pre-commit hooks

`prek` runs the project's pre-commit hooks, which already include ruff and ty:

```bash
uv run prek run               # run hooks on staged files
uv run prek run --all-files   # run hooks across the whole repo
```

Because prek covers ruff and ty, don't also run `ruff`/`ty` by hand on the same
change; that's a duplicate check. Run those directly only while iterating on a
file (see `uv-project.md`); for the commit gate, `prek` is the single pass.

## Splitting work

One commit per coherent change. If a single diff does two separable things, or a
reviewer would read it faster in stages, split it into several commits. A
refactor and the feature it enables are two commits, not one.

## Title

```
<type>(<scope>): <short summary>
```

`<type>` is one of the Angular types:

- **feat**: a new feature or capability.
- **fix**: a bug fix.
- **refactor**: a code change that neither fixes a bug nor adds a feature
  (renames, restructuring, extracting a function).
- **perf**: a change made to improve performance.
- **docs**: documentation only, including docstrings and the `.guidelines-ai`
  files.
- **test**: adding or correcting tests, with no production-code change.
- **style**: formatting, whitespace, import order; no behavior change.
- **build**: build system, packaging, or dependency changes (`pyproject.toml`,
  the lockfile).
- **ci**: CI configuration and scripts.
- **chore**: housekeeping that fits nowhere above (config bumps, repo cleanup).

`<scope>` is the area touched: a package, module, or subsystem (`kernels`,
`data`, `train`). Keep it short. Drop the parentheses if a scope doesn't help.

`<short summary>` describes the overall change in very few words. Imperative
mood, no capital on the first letter, no trailing period.

```
feat(kernels): add rbf kernel with length-scale parameter
fix(data): handle empty batch in collate
refactor(train): extract eval loop into pure function
docs(guidelines): add uv project rules
```

## Body

```
**Summary:**
<what problem this solves and how it solves it>

**Test plan:**
<how the change was verified>
```

**Summary** explains the problem the change addresses and how it does so, not a
restatement of the diff. It can reference related earlier changes when that's the
context a reviewer needs ("follows the kernel refactor in …").

**Test plan** is only there when it's worth it. A new function or a behavior
change needs one. Pure documentation edits, comment cleanups, or trivial renames
don't; skip the section entirely rather than writing "n/a".

All of this prose follows `avoid-ai-slop.md`: plain words, no filler, no
significance inflation.
