# ml-research-template

A [copier](https://copier.readthedocs.io/) template for ML research projects: a
uv workspace with a `core` training scaffold, a worked example package, Hydra
config groups, tooling config (ruff, ty, prek, pytest), the `.guidelines-ai/`
rulebook, a LaTeX paper build system, and a PDF-cataloguing workflow.

Unlike GitHub's "Use this template" button, copier keeps a link to the template
in each generated project, so improvements here can be pulled into projects that
were made earlier.

## Generate a project

```bash
uvx copier copy gh:adrien-lagesse/ml-research-template path/to/new-project
```

copier asks five questions — `project_slug`, `project_description`,
`author_name`, `author_email`, `paper_title` — and writes a project with those
filled in. Then:

```bash
cd path/to/new-project
git init && git add -A && git commit -m "chore: generate from ml-research-template"
uv run prek install          # install the git hooks
uv run --all-packages pytest # first run resolves and writes uv.lock
```

The template ships no `uv.lock`; each project resolves its own on the first
`uv run` (or run `uv lock` explicitly).

## Pull template updates into a project

From inside a generated project (clean working tree):

```bash
uvx copier update            # merge the latest template release
uvx copier update --vcs-ref v0.2.0   # or a specific version
```

copier reads the project's `.copier-answers.yml`, applies the diff between the
template version the project was made from and the target version, and leaves
any conflicts as merge markers to resolve. Review the result like any diff.

## Versioning

Releases are git tags: `v0.1.0`, `v0.2.0`, … `copier update` targets the latest
tag by default. To cut a release after changing the template:

```bash
git tag v0.2.0 && git push --tags
```

## What is templated

Only files named `*.jinja` are rendered (the suffix is stripped on output);
everything else is copied byte-for-byte. That keeps Jinja away from the
justfile's own `{{name}}`, Hydra's `${...}`, and the `__NAME__` placeholder in
`experiments/_template/`. The rendered files are the project's `pyproject.toml`,
`README.md`, `CLAUDE.md`, `ruff.toml`, the two package/script module docstrings,
`ressources/`'s README and converter, and the paper's `main.tex` / `talk.tex`
title and author. The LaTeX house style keeps its fixed name, `filao`.

The project skeleton lives under `template/`; `copier.yml` holds the questions.
