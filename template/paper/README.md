# paper

The article, in LaTeX, with results generated from experiment data so the PDF
and the code stay in step. This README is the rulebook: a human or an agent that
follows it produces a consistent, high-quality paper. The rules are strict on
purpose. They are also *this project's* house style — another project may choose
differently, so the last section says where each rule is wired if you want to
change it.

## Build

With [Tectonic](https://tectonic-typesetting.github.io) (`brew install tectonic`).
`main.tex` uses the house `filao` class, so point tectonic at that style folder
(this tectonic does not read `TEXINPUTS`). From `paper/`:

```bash
tectonic -Z search-path=styles/filao main.tex --outdir _build   # -> _build/main.pdf
```

Tectonic runs its built-in bibtex, so the bibliography needs no external biber.
Packages that shell out (`svg` via Inkscape, `minted` via Pygments) need
`tectonic -Z shell-escape`; the default stack does not.

The **slides** are a second target, `talk.tex` — a beamer deck on the house
`filao-beamer` theme that reuses these same `assets/` figures, so the paper and
the talk never disagree:

```bash
tectonic -Z search-path=styles/filao-beamer talk.tex --outdir _build
```

See `styles/filao-beamer/README.md` for the skins (`dark`/`light`) and the
image/video helpers. To target a journal or conference, copy the paper into
another wrapper (name it whatever you like) that loads the venue's style — see
`styles/README.md`.

## Organization — what goes where

```
paper/
    main.tex          root: \input preamble, macros, then the sections
    preamble.tex      \usepackage loads ONLY — no definitions
    macros.tex        every \newcommand / \DeclareMathOperator (notation)
    sections/         one file per section
    results/<name>/   data.csv — experiment output, DATA ONLY, no LaTeX
    assets/<name>/    asset.tex — one rendered element (figure, table, algorithm)
    talk.tex          the slide deck: a beamer talk reusing the same assets
    styles/           document classes and venue style files (see styles/README.md)
    references.bib    bibliography
```

- **`preamble.tex` holds no definitions.** Package loads only. A `\newcommand`
  goes in `macros.tex`.
- **`results/` is data, `assets/` is presentation.** A run in `packages/` writes
  numbers to `results/<name>/data.csv`. An `assets/<name>/asset.tex` reads that
  CSV and renders it. A figure and a table over the same run are two assets over
  one CSV, so they cannot disagree.
- **One asset per element.** Each figure, table, or algorithm is its own
  `assets/<name>/asset.tex`, an `\input`-able fragment. The float, caption, and
  label live in the section, not the asset.
- **Numbers are never hand-typed.** Every number in a figure or table comes from
  `results/`. To change a number, rerun the experiment, not the `.tex`.

## Rules

### Floats: figures and tables

- **Figure caption goes *below* the figure. Table caption goes *above* the
  table.** In source: figure `= content, then \caption`; table `= \caption, then
  content`.
- **`\label` immediately after `\caption`**, and inside the float. Prefixes:
  `fig:`, `tab:`, `alg:`, `eq:`, `thm:`, `sec:`, `def:`, `prop:`.
- **Two widths, no others.** Every float is either one text **column** or the
  **full** width:
  - column → `width=\floatcol` in a plain `figure` / `table`.
  - full → `width=\floatfull` in a starred `figure*` / `table*`.

  Both are defined in `macros.tex` as aliases of LaTeX's live layout lengths:
  `\floatcol` = `\columnwidth`, `\floatfull` = `\textwidth`. They track whatever
  class is in force with no per-venue setup — a two-column venue makes `\floatcol`
  its narrow column and `\floatfull` the page span on its own; a single-column
  venue (the default `article`, NeurIPS) collapses the two to the one text width,
  which is correct, since there is only one. Never hardcode a width (`width=8cm`,
  `0.8\linewidth`) in an asset — reference `\floatcol` or `\floatfull`. Inside a
  `subfigure`, the content uses the subfigure box (`\linewidth`); the enclosing
  float still picks `\floatcol` or `\floatfull`.
- **Every float is referenced** from the prose, by `\cref`, before or near where
  it appears.

### Citations

- **Author–year style** (natbib + `plainnat`). Never numeric.
- **Never bare `\cite`.** Use:
  - `\citet{key}` when the authors are the subject — "\citet{he2015} show …" →
    "He et al. (2015) show …".
  - `\citep{key}` for a parenthetical aside — "… by backpropagation~\citep{…}" →
    "… (Rumelhart et al., 1986)".
- **Tie the citation to its word** with `~` before `\citep`.
- A new reference is a full entry in `references.bib` (author, title, year, and
  venue/journal).

### Cross-references

- **`\cref` / `\Cref`, never `\ref`.** cleveref supplies the type ("Figure 2",
  "eq. (3)") and stays correct when things move. `\Cref` at a sentence start,
  `\cref` mid-sentence.

### Tables

- **Always `booktabs`**: `\toprule`, `\midrule`, `\bottomrule`. **No vertical
  rules**, no `\hline`.
- Numbers go through `siunitx` — an `S` column or `\num{}` — so decimals align
  and formatting is uniform.
- Mean ± std over seeds: `\num{<mean> +- <std>}` (see the pattern below).

### Numbers and units

- **Every number through `siunitx`**: `\num{1.2e-3}`, `\qty{4}{\giga\byte}`.
  Never type `1.2 \times 10^{-3}` by hand.

### Figures

- **Vector, not raster.** pgfplots or TikZ for anything drawn; a PDF (or SVG
  source) for external art. Raster (PNG/JPG) only for photographs.
- Data figures are pgfplots reading a `results/` CSV. Show seed spread with error
  bars or a shaded band (below).
- A heavy figure that slows the build is compiled once to a cached
  `asset.pdf` (a `standalone` wrapper, compiled from `paper/` so the `results/`
  path resolves) and `\includegraphics`-ed. Cached `assets/**/*.pdf` are
  git-ignored.

### Color, colorblindness, and grayscale

The palette is chosen so a figure reads the same for a colorblind reader and on a
black-and-white printout. One rule carries most of the weight: **never let color
be the only signal.**

- **Use the house palette, in order.** Okabe-Ito, defined in `preamble.tex`:
  `oiBlue`, `oiVermillion`, `oiGreen`, `oiPurple`, `oiOrange`, `oiSky`. Assign
  from the front; never invent a hue or reach for a raw `red`/`green`/`blue`. The
  order is fixed, not cycled by rank — a series keeps its color when others come
  and go.
- **Redundant encoding is mandatory.** Every series is distinguished by color
  **and** a second channel — line style (`solid`, `densely dashed`, `dotted`) or
  marker shape. The default plot cycle (`cycle list name=paper`) already pairs
  each color with a dash and a marker, so a bare `\addplot` is safe; if you set a
  color by hand, add the dash/marker yourself. This is what makes the figure work
  in grayscale and for total color blindness — the same fix covers both.
- **A quantity keeps its encoding across figures.** In this paper validation is
  `oiBlue, solid` and train is `oiVermillion, densely dashed` in every panel.
- **At most ~6 categorical series.** A 7th is a sign to split into small
  multiples or group into "other", not to add a color the palette can't separate.
- **Sequential data (magnitude: heatmaps, surfaces) uses `viridis`** — set as the
  default colormap. It is perceptually uniform, CVD-safe, and monotonic in
  grayscale. **Never `jet`/rainbow.**
- **Diverging data (signed, around a midpoint)** uses two CVD-safe hues with a
  light neutral middle (e.g. `oiBlue` → near-white → `oiOrange`). Never a hue at
  the midpoint, never rainbow.
- **Ink stays ink.** Axis lines, ticks, and text are black; the grid is a
  recessive light gray. A series color belongs to the mark, not to the labels.
  Area fills use a light tint (`oiBlue!15`), not the full-strength hue.
- **Grayscale check.** Before shipping a figure, convert it to grayscale (print
  preview, or `pdftoppm -gray`) and confirm every series is still tellable apart.
  If it isn't, the second channel is missing.

The palette was validated for colorblind adjacent-pair separation (worst pair
ΔE ≈ 18, target ≥ 12). If you add or replace a color, re-validate before using it.

### Math and notation

- **All notation lives in `macros.tex`.** Reach for `\R`, `\E`, `\norm{·}`,
  `\argmin`; add new shared notation there, never in a section.
- The Deep Learning Book notation (`goodfeli/dlbook_notation`, modernized) will
  be pasted into `macros.tex`; when it lands, keep one definition per symbol and
  drop any duplicate above it.

### Prose

- **One sentence per line.** Diffs stay readable and a claim blames to a line. Do
  not hard-wrap mid-sentence.
- Follow `.guidelines-ai/avoid-ai-slop.md`: plain words, varied sentence length,
  a position taken rather than hedged.

### The never list

`\cite` · `\ref` · numeric citations · vertical table rules · `\hline` ·
hand-typed numbers or scientific notation · a width hardcoded in an asset ·
a number in a figure/table that isn't in a `results/` CSV · a `\newcommand` in a
section or the preamble · a rainbow/`jet` colormap · a raw `red`/`green`/`blue` ·
a series told apart by color alone.

## Seeded runs: error bars, bands, and ± tables

Runs over several seeds write a mean column and a matching `<name>_std` column.
Three patterns turn those into floats; all read the two columns directly. For a
trend, prefer the shaded band over error-bar whiskers (quieter, and it matches
the loss curve); reserve error bars for discrete points shown without a
connecting line.

**Error bars** (pgfplots): name the std column with `y error`.

```latex
\addplot+[mark=*, error bars/.cd, y dir=both, y explicit]
  table [col sep=comma, x=width, y=test_acc, y error=test_acc_std] {results/<name>/data.csv};
```

**Shaded ±1 std band** (`fillbetween`, loaded in the preamble): compute the edges
inline with `y expr` and `\thisrow{}`, draw them invisibly with `forget plot`,
fill between, then draw the mean.

```latex
\addplot [name path=hi, draw=none, forget plot]
  table [col sep=comma, x=epoch, y expr=\thisrow{val_loss}+\thisrow{val_loss_std}] {...};
\addplot [name path=lo, draw=none, forget plot]
  table [col sep=comma, x=epoch, y expr=\thisrow{val_loss}-\thisrow{val_loss_std}] {...};
\addplot [oiBlue!15, forget plot] fill between [of=hi and lo];
\addplot [oiBlue, thick] table [col sep=comma, x=epoch, y=val_loss] {...};   % mean
```

**Mean ± std in a table**: `csvsimple` reads the rows, `\num` combines the two
columns (siunitx `separate-uncertainty` is set in the preamble). Use positional
`\csvcol<n>` and wrap in `\num{}` — a `_std` header defeats named columns, and
`\num{}` avoids the siunitx `S`-column look-ahead bug with `late after line`.

```latex
\csvreader[separator=comma, late after line=\\]{results/<name>/data.csv}{}{%
  \csvcoli & \num{\csvcolv +- \csvcolvi}}
```

Worked examples live in `assets/width-sweep` (error bars), `assets/loss-curve`
(band), and `assets/results-table` (± column).

## Changing the house style

These rules are opinions, wired in a few places so a fork can flip them:

- **Citation style** — the `natbib` options in `preamble.tex` and
  `\bibliographystyle` in `main.tex`. Numeric instead of author–year: drop
  `authoryear` and switch to a numeric `.bst`.
- **Caption position** — a source convention (order of `\caption` vs content);
  enforce it mechanically with `floatrow` if you want the compiler to check.
- **Float widths** — the `\floatcol` / `\floatfull` aliases in `macros.tex`.
- **Palette** — the `\definecolor` set and `paper` cycle list in `preamble.tex`.
  Swap the hues, then re-validate colorblind separation before using them.
- **Notation** — `macros.tex`.
