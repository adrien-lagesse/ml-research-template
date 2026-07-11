# styles

A home for document classes and style files, kept out of the working document
so `main.tex` stays clean. Nothing here is compiled directly.

- `neurips/`, `iclr/`, `icml/` — drop each venue's **official** style files
  (`*.sty`/`*.cls`, downloaded from the call for papers) into the matching
  folder. They are the venue's copyright, so they are not shipped with the repo;
  each folder's README says how to fetch them. NeurIPS is wired up already — see
  `neurips/README.md`.
- `filao/` — the house style, committed with the paper. `filao/filao.cls` is the
  class (see `filao/README.md`); `main.tex` builds the paper on it.
- `filao-beamer/` — the house **presentation** theme, also committed.
  `filao-beamer/beamerthemefilao.sty` is a Beamer theme (see its README);
  `talk.tex` builds the slides on it, reusing the paper's own `assets/`
  figures.

## Targeting a venue

`sections/`, `results/`, `preamble.tex`, and `macros.tex` are venue-neutral, so
the content never moves. `main.tex` is one wrapper over them, on the house
`filao` class. To build for a venue instead, copy `main.tex` to a new file —
name it whatever you like — and change two things: the class/style line and the
title block. The shared `\input`s stay.

A venue's format is either a document **class** or a **package** over `article`,
and the wrapper's first line differs accordingly:

```latex
\documentclass{icml2025}                       % venue ships a .cls
% -- or --
\documentclass{article}                        % venue ships a .sty package
\usepackage[preprint, nonatbib]{neurips_2026}  % (NeurIPS is this kind)
```

Point tectonic at the style folder with `-Z search-path` (this tectonic does
not read `TEXINPUTS`), and build that wrapper. From `paper/`:

```bash
tectonic -Z search-path=styles/neurips your-neurips-wrapper.tex --outdir _build
```

The main thing a wrapper has to handle: pass `nonatbib` (or the venue's
equivalent) when the style auto-loads `natbib`, so `preamble.tex` keeps the
author-year setup. The house float widths need no attention — `\floatcol` and
`\floatfull` (in `macros.tex`) alias `\columnwidth`/`\textwidth`, so they take
the venue's real column and text widths on their own.
