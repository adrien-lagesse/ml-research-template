"""Tests for MLflowLogger against a temporary SQLite tracking store."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
import torch

from core import CheckpointState
from core.logger import MLflowLogger


@pytest.fixture
def tracking_uri(tmp_path: Path) -> Iterator[str]:
    """A SQLite tracking store with artifacts and an ``exp`` experiment in tmp_path.

    MLflow refuses the deprecated file store, so the tests use a SQLite backend
    like the real server. The experiment is created here with an explicit
    artifact location under ``tmp_path`` so nothing lands in the repo. The run a
    test leaves active (one whose ``close`` an assertion pre-empted) is ended on
    teardown, so it can't fail the next test's ``start_run``.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("exp", artifact_location=(tmp_path / "artifacts").as_uri())
    yield uri
    if mlflow.active_run() is not None:
        mlflow.end_run()


def _sample_state(*, global_step: int = 7) -> CheckpointState:
    """A full checkpoint state built from a one-layer model."""
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=0
    )
    return CheckpointState(
        model=model.state_dict(),
        loss={},
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        global_step=global_step,
    )


def test_mlflow_logger_records_metric_history(tracking_uri: str) -> None:
    """`log_dict` reaches the run as a prefixed metric keyed by step."""
    logger = MLflowLogger(experiment_name="exp", run_name="run")
    run = mlflow.active_run()
    assert run is not None
    run_id = run.info.run_id
    logger.log_dict({"loss": torch.tensor(2.5)}, step=3, prefix="train")
    logger.close()

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    history = client.get_metric_history(run_id, "train/loss")
    assert [(point.value, point.step) for point in history] == [(2.5, 3)]


def test_mlflow_logger_summary_logs_params_without_artifacts(tracking_uri: str) -> None:
    """With artifacts off, `log_summary` logs scalar params and writes no files."""
    logger = MLflowLogger(experiment_name="exp", run_name="run")
    run = mlflow.active_run()
    assert run is not None
    run_id = run.info.run_id
    logger.log_summary(
        {"seed": 0, "device": "cpu", "model": {"width": 16}}, name="config"
    )
    logger.close()

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    params = client.get_run(run_id).data.params
    assert params["seed"] == "0"
    assert params["device"] == "cpu"
    assert "model" not in params  # a nested mapping is not a searchable param
    assert client.list_artifacts(run_id) == []  # no files when artifacts are off


def test_mlflow_logger_uploads_summary_and_checkpoint_when_enabled(
    tracking_uri: str,
) -> None:
    """With artifacts on, the summary document and both checkpoint files upload."""
    logger = MLflowLogger(experiment_name="exp", run_name="run", artifacts=True)
    run = mlflow.active_run()
    assert run is not None
    run_id = run.info.run_id
    logger.log_summary({"seed": 0}, name="config")
    logger.save_checkpoint(_sample_state(), name="best")
    logger.close()

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    root = {artifact.path for artifact in client.list_artifacts(run_id)}
    assert "config.json" in root
    checkpoints = {
        artifact.path for artifact in client.list_artifacts(run_id, "checkpoints")
    }
    assert checkpoints == {
        "checkpoints/best.safetensors",
        "checkpoints/best.state.pt",
    }


def test_mlflow_logger_skips_checkpoint_when_artifacts_off(tracking_uri: str) -> None:
    """With artifacts off, `save_checkpoint` uploads nothing."""
    logger = MLflowLogger(experiment_name="exp", run_name="run")
    run = mlflow.active_run()
    assert run is not None
    run_id = run.info.run_id
    logger.save_checkpoint(_sample_state(), name="best")
    logger.close()

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    assert client.list_artifacts(run_id) == []
