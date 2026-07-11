"""Plot the training curves of a logged ``dummy_regression`` run.

Reads ``metrics.csv`` from a run directory under ``$LOG_ROOT`` (default
``.logs``) ``/dummy_regression/`` — the named run when one is given, otherwise the
lexicographically newest run that has a ``metrics.csv`` — and writes
``curves.png`` beside this script, then prints its path. The figure has one
panel per metric: train and validation as lines over global steps, the final
test value as a single labeled marker, and a log y-scale when a panel's
positive values span more than two decades. Metric columns are expected to be
named ``<split>/<metric>`` with a shared ``step`` column. Entry point:
``uv run experiments/dummy_regression/visualize.py [run_name]``.
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXPERIMENT_DIR.parents[1]
# Same root the LocalLogger writes under; relative paths are taken from the repo root.
_LOG_ROOT = Path(os.environ.get("LOG_ROOT", ".logs"))
_LOGS_DIR = (
    _LOG_ROOT if _LOG_ROOT.is_absolute() else _REPO_ROOT / _LOG_ROOT
) / _EXPERIMENT_DIR.name

# Categorical palette slots 1-3, in their CVD-safe order; grays are text inks.
_SPLIT_COLORS = {"train": "#2a78d6", "validation": "#1baf7a", "test": "#eda100"}
_INK = "#0b0b0b"
_INK_MUTED = "#52514e"
_GRID = "#e5e4e0"


def series(rows: list[dict[str, str]], column: str) -> tuple[list[int], list[float]]:
    """Extract one metric's trajectory from wide CSV rows.

    Args:
        rows: Rows of the run's metrics CSV in logged order; cells may be
            empty strings where a step did not report that metric.
        column: Metric column to extract, e.g. ``validation/loss``.

    Returns:
        The steps at which ``column`` has a value, and those values, aligned.
    """
    steps: list[int] = []
    values: list[float] = []
    for row in rows:
        if row.get(column):
            steps.append(int(row["step"]))
            values.append(float(row[column]))
    return steps, values


def metric_names(fieldnames: list[str]) -> list[str]:
    """List the metric suffixes present in the CSV header, first-seen order.

    Args:
        fieldnames: Header of the metrics CSV; metric columns are named
            ``<split>/<metric>``, e.g. ``validation/mae``.

    Returns:
        The distinct ``<metric>`` suffixes, e.g. ``["loss", "mae"]``.
    """
    names: list[str] = []
    for field in fieldnames:
        if "/" in field:
            name = field.split("/", 1)[1]
            if name not in names:
                names.append(name)
    return names


def plot_curves(
    rows: list[dict[str, str]], fieldnames: list[str], title: str, out_path: Path
) -> None:
    """Render one panel per metric and save the figure.

    Train and validation appear as lines over steps; test, which logs a single
    end-of-run value, appears as one marker with its value printed beside it.
    A panel switches to a log y-scale when its values are positive and span
    more than two decades.

    Args:
        rows: Rows of the run's metrics CSV in logged order.
        fieldnames: Header of the metrics CSV.
        title: Figure title, typically the run directory name.
        out_path: Where the PNG is written.
    """
    metrics = metric_names(fieldnames)
    assert metrics, "no <split>/<metric> columns in the CSV header"
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(4.4 * len(metrics), 3.4),
        layout="constrained",
        squeeze=False,
    )
    axis_list = list(axes[0])

    for ax, metric in zip(axis_list, metrics, strict=True):
        panel_values: list[float] = []
        for split in ("train", "validation"):
            column = f"{split}/{metric}"
            if column not in fieldnames:
                continue
            steps, values = series(rows, column)
            ax.plot(steps, values, color=_SPLIT_COLORS[split], linewidth=2, label=split)
            panel_values.extend(values)
        test_column = f"test/{metric}"
        if test_column in fieldnames:
            steps, values = series(rows, test_column)
            ax.scatter(
                steps[-1:],
                values[-1:],
                color=_SPLIT_COLORS["test"],
                s=60,
                zorder=3,
                edgecolors="white",
                linewidths=1.5,
                label="test",
            )
            if values:
                ax.annotate(
                    f"{values[-1]:.3g}",
                    xy=(steps[-1], values[-1]),
                    xytext=(-9, 7),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color=_INK_MUTED,
                )
                panel_values.extend(values[-1:])

        if (
            panel_values
            and min(panel_values) > 0
            and (max(panel_values) / min(panel_values) > 100)
        ):
            ax.set_yscale("log")
        ax.set_title(metric, color=_INK, fontsize=11)
        ax.set_xlabel("step", color=_INK_MUTED, fontsize=9)
        ax.grid(visible=True, color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_INK_MUTED)
        ax.tick_params(colors=_INK_MUTED, labelsize=8)

    axis_list[0].legend(frameon=False, fontsize=9, labelcolor=_INK)
    fig.suptitle(title, color=_INK, fontsize=12)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    """Plot the requested (or newest) run and print the output path.

    Raises:
        FileNotFoundError: If this experiment has no logs directory, no run
            contains a ``metrics.csv``, or the named run lacks one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run",
        nargs="?",
        help="run directory name under the experiment's logs; newest when omitted",
    )
    args = parser.parse_args()

    if not _LOGS_DIR.is_dir():
        raise FileNotFoundError(f"no logged runs under {_LOGS_DIR}")
    if args.run:
        run_dir = _LOGS_DIR / args.run
    else:
        run_dirs = sorted(
            d for d in _LOGS_DIR.iterdir() if (d / "metrics.csv").is_file()
        )
        if not run_dirs:
            raise FileNotFoundError(f"no metrics.csv in any run under {_LOGS_DIR}")
        run_dir = run_dirs[-1]
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"no metrics.csv in {run_dir}")

    with metrics_path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    out_path = _EXPERIMENT_DIR / "curves.png"
    plot_curves(rows, fieldnames, run_dir.name, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
