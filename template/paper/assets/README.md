# assets

Every rendered element of the paper — figures, tables, algorithms, diagrams —
one folder per asset. `results/` holds the experiment data; assets that plot or
tabulate it read a CSV from there, while hand-authored assets (a TikZ diagram, an
algorithm) stand alone.

## An asset folder

- `asset.tex` — the raw fragment, `\input`-ed inside a float in a section. The
  source of truth, and the only file that has to exist.
- `asset.pdf` — optional cached render, git-ignored and rebuilt on demand. Use it
  when a heavy figure slows the main build: add a `standalone.tex` wrapping the
  fragment, compile it (from the `paper/` directory, so `results/` paths still
  resolve) to `asset.pdf`, and `\includegraphics{<name>/asset}` instead of
  `\input`.

## Using one

```latex
\begin{figure}[t]\centering
  \input{assets/demo-plot/asset}
  \caption{...}\label{fig:demo}
\end{figure}
```

A data-driven asset reads its numbers from `results/<name>/data.csv`, so a plot
and the table beside it can never disagree — they are two assets over one
dataset.
