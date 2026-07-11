# filao-beamer

The house presentation theme: the slide companion to `filao.cls`. Committed with
the paper, like `filao/`. `beamerthemefilao.sty` is one self-contained Beamer
theme (colour + font + inner + outer), built for looks first — a dark,
image-forward deck by default, a light editorial skin on request.

## The point: one set of figures

The deck reuses the paper's figures. Every plot and table in `talk.tex` is
the same `assets/<name>/asset.tex` the article builds: the assets read
`results/*.csv` and size to `\floatcol`/`\floatfull`, so they drop into a frame
at slide width with no second copy. Change a number in `results/`, and both the
paper and the talk move together. That is why the deck lives in `paper/` and
compiles from there — the relative paths and the float-width aliases only resolve
from that directory.

Notation is shared too: the deck `\input`s the same `macros.tex`. It does **not**
`\input{preamble.tex}` — that file loads article-only packages (`caption`,
`natbib`, `amsthm`) that fight Beamer, so the theme loads the plotting/math subset
the assets need (pgfplots, siunitx, booktabs, algpseudocodex, …) itself, and
copies the paper's Okabe-Ito palette and `paper` plot cycle verbatim. A plot on a
slide is the same plot as in the article.

## Build

From `paper/`, point tectonic at this folder with `-Z search-path` (this tectonic
does **not** read `TEXINPUTS`):

```bash
tectonic -Z search-path=styles/filao-beamer talk.tex --outdir _build
```

The first run fetches the TeX Gyre Heros font files from tectonic's bundle; later
runs are cached.

## Skins and options

On the `\usetheme[...]{filao}` line in `talk.tex`:

- *(no option)* — the **dark** skin: deep charcoal, a luminous accent, light text.
  The default, and the one the image/plot-heavy layout is tuned for.
- `light` — a near-white editorial skin with the paper's deep-blue accent. Prints
  well and sits closest to the article's own look.
- `fira` — swap the font to **Fira Sans** if it is installed on the build machine
  (needs the system font; the default needs none). Composes with the skin, e.g.
  `\usetheme[light, fira]{filao}`.

## Fonts, and why fontspec

Tectonic runs XeTeX in Unicode (`TU`) mode, where the 8-bit NFSS font packages
(`tgheros`, `helvet`) don't resolve — their shapes are `T1`-only, so they fall
back to Latin Modern **serif** silently. The theme instead loads TeX Gyre Heros
with `fontspec`, by its OpenType **filename** (`texgyreheros-*.otf`), which
tectonic fetches from its own bundle. No font install, no system font-database
lookup, and a genuinely sans deck. To use a different face, change the
`\setsansfont` block at the top of the theme, or pass `fira`.

The type is **coordinated**, so text, numbers, and code read as one family
(not the usual Beamer mix):

- **Mono** is Fira Mono (`\setmonofont`), so `\texttt` — the title email, any
  code — matches the sans instead of falling back to CM typewriter.
- **Digits** run through siunitx with `detect-all`, so `\num{}` in tables and
  inline picks up Heros: the results-table numbers match the body text. Figure
  **axis ticks are left as the article renders them** — the assets are reused
  verbatim, not restyled.
- **Symbolic math** ($\phi$, $\mathcal D$, $\bm W_\ell$) stays Computer Modern.
  That is a deliberate choice: the serif-math-on-sans-text contrast is a common
  academic look and keeps the slide math identical to the paper. To coordinate
  math too, load `unicode-math` with a sans math font (e.g. Fira Math) — but
  test it against `macros.tex` first (`bm`, `amssymb`, `\DeclareMathAlphabet`).

**Headings** are letterspaced small caps (frame titles, section dividers, the
title) via the `\filao@smallcaps` helper — editorial, but still bold and strong.

## Beauty over accessibility (on the chrome only)

The slide **chrome** — title bars, rules, blocks, bullets, section dividers —
goes for looks: a single saturated accent, flat blocks, a thin progress bar. That
is a deliberate break from the paper's colourblind-safe rule, and it applies to
the furniture only. The **data figures keep the Okabe-Ito palette**, so a plot is
still readable for a colourblind viewer and in grayscale. Don't recolour a plot to
match the accent.

## What the type owes to Pascal Michaillat (and what it doesn't)

The typographic craft here follows [Michaillat's slide
advice](https://pascalmichaillat.org/c/): one coordinated type family across
text/numbers/mono, small-caps headings, a little more line lead, styled
statement boxes, and quiet captions. What we **left** is his minimalism — his
templates are 4:3, strip colour from the chrome, drop section dividers, a
footline, and a closing slide, and hold to a single restrained hue. This theme
keeps all of that furniture on purpose: the accent chrome, the progress bar, the
dark skin, the section dividers, and the `\standout` slides are the point. Craft
from him, richness kept.

## Helper macros

Beyond the usual frames, the theme adds a few layout helpers:

- `\standout{<text>}` — a full-accent slide with one big centred line. The
  punchline.
- `\fullbleed{<image>}` — an edge-to-edge picture filling the frame. Use inside a
  `\begin{frame}[plain]`.
- `\splitframe{<left>}{<right>}` — a balanced two-column image-and-text split.
- `\slidenote{<text>}` — a quiet centred, muted note under a figure or table
  (the standardized caption; replaces hand-rolled caption spans).
- `theorem` / `definition` / `assumption` / `result` environments — a formal
  statement in a soft rounded surface box, with a small-caps accent label, an
  optional name, and an italic body: `\begin{theorem}[universal approximation]
  ... \end{theorem}`.
- Section dividers appear automatically at every `\section{...}` (a full-accent
  slide with the section name).

## Video and animation (viewer-dependent)

A PDF can carry video, but **only Adobe Acrobat/Reader and pdfpc play it**; every
other viewer shows the poster still. Two hooks, both documented as such:

- `\vidfull{<file.mp4>}` / `\vidinline{<file.mp4>}{<width>}` — `media9` players,
  autoplay on page open. Each wants a poster image at `<file>-poster` for the
  static fallback.
- `\animframes{<basename>}{<fps>}{<first>}{<last>}` — an `animate` sequence over
  PNG/PDF frames exported from `packages/` (e.g. a training-curve animation). Plays
  inside Adobe Reader; shows one still elsewhere.

Keep video and frame sequences under `assets/<name>/` next to the figure they
belong to. They are heavy binaries, so `*.mp4` and `frames/` directories are
git-ignored (see `paper/.gitignore`); the `asset.tex` fragments stay committed.

## Tuning the look

Everything worth changing sits near the top of `beamerthemefilao.sty`:

- **Skin colours** — the `\definecolor` block under *CHROME PALETTE* (background,
  surface, text, accent, warm). The accent is a brightened `oiBlue` on dark, the
  paper's `filaoaccent` on light.
- **Fonts** — the `\setsansfont` / `\setmonofont` block (or pass `fira`).
- **Small-caps headings** — the `\filao@smallcaps` helper (drop the
  `\addfontfeatures{LetterSpace=...}` for tighter caps, or the whole helper to
  return to plain bold titles).
- **Line lead / item spacing** — `\linespread` and the `itemize/enumerate body
  begin` template.
- **Statement boxes** — the `filao statement` colour and the `\filao@stmtbegin`
  padding.
- **Progress bar** — `\filao@barheight` and the `headline` template.
- **Frame title, footline, title page, section divider** — the `\setbeamertemplate`
  blocks, each labelled.
