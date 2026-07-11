# filao

The house style. Unlike the venue folders, this one is committed — `filao.cls`
is versioned with the paper.

## filao.cls

A single-column preprint class: a thin layer over `article` that sets the page
geometry, the title block, the section headings, and a running header, plus an
optional first-page notice. It owns **layout only** — the
package stack (math, tables, plots, citations, theorems) stays in
`preamble.tex`, the same split the NeurIPS style uses. So the wrapper loads the
class, then `\input{preamble}`, and nothing double-loads.

`paper/main.tex` builds on it. Build from `paper/`, pointing tectonic at this
folder:

```bash
tectonic -Z search-path=styles/filao main.tex --outdir _build
```

### Options

On the `\documentclass[...]{filao}` line:

- `preprint` (default) — a small gray "Preprint." at the foot of the first page.
- `final` — no notice.
- Any other option (`10pt`, `twoside`, …) passes through to `article`.

### Commands the class adds

- `\runningtitle{...}` — the short title in the running header. Set it in the
  preamble of the wrapper; without it the header's left slot is empty.
- `\notice{...}` — override the first-page notice text (default `Preprint.`).

### Tuning the look

Everything worth changing sits at the top of `filao.cls`:

- **Accent colour** — `\definecolor{filaoaccent}{HTML}{0B4F6C}`. It colours the
  title rule. Set it to `0072B2` to match the plots' `oiBlue` exactly.
- **Measure and margins** — the `geometry` options (`textwidth=5.6in`, …).
- **Headings** — the `\titleformat` / `\titlespacing` block (titlesec).
- **Title block** — the `\maketitle` redefinition.

Because filao is single-column, `\floatcol` and `\floatfull` (from
`macros.tex`) both resolve to the one text width; a `figure*` spans the same
width as a `figure`, as it should here.
