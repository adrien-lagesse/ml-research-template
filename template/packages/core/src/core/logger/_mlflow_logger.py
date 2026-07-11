"""Logger that streams metrics and scalar params to an MLflow run.

The tracking location comes from the ``MLFLOW_TRACKING_URI`` environment
variable that MLflow reads on its own (``paths.env`` sets it); the logger never
sets it. Metrics and params always go to that store, browsable in the UI. Files
are optional: with ``artifacts`` off (the default) this logger writes none, so
checkpoints and the JSON summaries stay with the LocalLogger under one per-run
directory; with ``artifacts`` on, checkpoints and summaries upload too, making a
remote run self-contained.
"""

from collections.abc import Mapping
from pathlib import Path
import tempfile

import mlflow

from core._checkpoint import CheckpointState
from core._checkpoint import save_checkpoint_files
from core.logger._logger import Logger
from core.logger._logger import _scalar_entries


class MLflowLogger(Logger):
    """Logger that records a run's metrics and scalar params to an MLflow store.

    On construction it selects (creating if absent) the experiment
    ``experiment_name`` in the store named by ``MLFLOW_TRACKING_URI``, and starts
    a run named ``run_name`` that stays active until ``close``. Metrics go to the
    store via ``mlflow.log_metrics``; a summary's scalar top level goes to the
    searchable params table. When ``artifacts`` is set, summary documents and
    checkpoints upload as run artifacts; otherwise the logger writes no files and
    leaves them to the LocalLogger.
    """

    def __init__(
        self,
        *,
        experiment_name: str,
        run_name: str,
        artifacts: bool = False,
    ) -> None:
        """Select the experiment and open a run in the environment's MLflow store.

        Args:
            experiment_name: MLflow experiment the run is grouped under; created
                in the store if no experiment of that name exists.
            run_name: Display name for the run.
            artifacts: When true, ``log_summary`` and ``save_checkpoint`` upload
                files as run artifacts; when false, both write nothing.
        """
        self.artifacts = artifacts
        mlflow.set_experiment(experiment_name)
        # The logger drives one run through MLflow's fluent API: log_metrics,
        # log_params, and log_artifacts all target this active run, and close
        # ends it. One run is active at a time, so no handle is threaded around.
        mlflow.start_run(run_name=run_name)

    def _log(self, values: dict[str, float], *, step: int) -> None:
        mlflow.log_metrics(values, step=step)

    def log_summary(self, summary: Mapping[str, object], *, name: str) -> None:
        """Mirror the summary's scalars into params, uploading the document too.

        The scalar top-level entries always go to the searchable params table;
        nested and non-scalar values are skipped. When ``artifacts`` is set, the
        full document is also uploaded as ``<name>.json``.

        Args:
            summary: JSON-serializable metadata to record with the run.
            name: Artifact stem for the ``.json`` document; used only when
                artifacts are enabled.
        """
        params = _scalar_entries(summary)
        if params:
            mlflow.log_params(params)
        if self.artifacts:
            mlflow.log_dict(dict(summary), f"{name}.json")

    def save_checkpoint(self, state: CheckpointState, *, name: str) -> None:
        """Upload ``state`` as run artifacts under ``checkpoints/`` when enabled.

        Args:
            state: The snapshot to persist.
            name: Filename stem shared by the checkpoint's two files.
        """
        if not self.artifacts:
            return
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint_files(state, Path(tmp), name=name)
            mlflow.log_artifacts(tmp, artifact_path="checkpoints")

    def close(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()
