# neurips

The NeurIPS style, and how to build the paper against it. You retarget the paper
with a small wrapper (a copy of `main.tex`, see below); this folder holds only
the style file that wrapper needs.

## Get the style file

The style file is the venue's copyright, so it is not committed. Fetch it (run
from this folder):

```bash
curl -sSL -o styles.zip https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip
unzip -j styles.zip 'neurips_2026.sty' && rm styles.zip
```

That leaves `neurips_2026.sty` here. The same zip also carries `neurips_2026.tex`
(the formatting-instructions example) and `checklist.tex` (the mandatory paper
checklist) if you want them. For a different year, the zip name and the package
name track the year — swap `2026` in the URL, in the file name above, and in the
`\usepackage[...]{neurips_2026}` line of your wrapper. The download link
is on that year's Call for Papers page at neurips.cc.

## Build

From `paper/`, point tectonic at this folder with `-Z search-path` (this
tectonic does **not** read `TEXINPUTS`):

```bash
tectonic -Z search-path=styles/neurips your-neurips-wrapper.tex --outdir _build
```

The first run downloads the Times font files; later runs are cached.

## Options

Set on the `\usepackage[...]{neurips_2026}` line in your wrapper:

- `preprint` — named authors, no line numbers, "Preprint. Under review." footer.
  This is the wrapper's default, and what an arXiv upload wants.
- *(no option)* — anonymous submission: authors hidden, line numbers on.
- `final` — camera-ready: named authors, the published-conference footer.
- `nonatbib` — kept on always. The style otherwise loads `natbib` in numeric
  mode; we suppress that so `preamble.tex` can load it author-year, the house
  citation rule. Leave it in.
- Track (2026): the default is the main conference; `position`, `eandd`,
  `creativeai`, and `education` switch the footer to that track.

## What the wrapper does

A NeurIPS wrapper is a copy of `main.tex` with two changes: load `neurips_2026`
as a package over `article` (with `nonatbib`) instead of the `filao` class, and
use the NeurIPS name/affiliation/email author block. Nothing else moves — the
sections, macros, assets, and bibliography are the same shared files. The house
float widths need no wrapper code: `\floatcol`/`\floatfull` alias
`\columnwidth`/`\textwidth`, so on this single-column style both resolve to the
5.5in text block automatically.
