"""Tests for plexe.retrain.retrain_model using a fabricated xgboost model package."""

import json
import tarfile
from unittest.mock import MagicMock

import joblib
import pandas as pd
import pytest

from plexe.models import Solution
from plexe.retrain import retrain_model


@pytest.fixture
def retraining_inputs(tmp_path):
    """Create a minimal retraining-ready model.tar.gz plus matching dataset."""
    pkg = tmp_path / "pkg"
    (pkg / "artifacts").mkdir(parents=True)
    (pkg / "src").mkdir(parents=True)

    df = pd.DataFrame(
        {
            "x1": range(40),
            "x2": [(i * 7) % 13 for i in range(40)],
            "y": [(i * 3) % 5 for i in range(40)],
        }
    )

    from xgboost import XGBRegressor

    model = XGBRegressor(n_estimators=2, max_depth=1)
    model.fit(df[["x1", "x2"]], df["y"])
    joblib.dump(model, pkg / "artifacts" / "model.pkl")

    metadata = {"model_type": "xgboost", "target_column": "y", "task_type": "regression", "experiment_id": "orig"}
    (pkg / "artifacts" / "metadata.json").write_text(json.dumps(metadata))
    (pkg / "src" / "trainer.py").write_text("# trainer\n")
    (pkg / "src" / "pipeline.py").write_text(
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "pipeline = Pipeline([('scaler', StandardScaler())])\n"
    )

    tarball = tmp_path / "model.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for item in pkg.rglob("*"):
            if item.is_file():
                tar.add(item, arcname=item.relative_to(pkg).as_posix())

    dataset = tmp_path / "new_data.parquet"
    df.to_parquet(dataset, index=False)
    return tarball, dataset


def test_retrain_model_returns_solution(retraining_inputs, tmp_path):
    """retrain_model must complete and return a valid Solution after training."""
    tarball, dataset = retraining_inputs

    artifacts_dir = tmp_path / "training_run" / "model_artifacts"
    (artifacts_dir / "artifacts").mkdir(parents=True)
    runner = MagicMock()
    runner.run_training.return_value = artifacts_dir
    config = MagicMock()
    config.training_timeout = 60

    solution, metrics = retrain_model(
        original_model_uri=str(tarball),
        train_dataset_uri=str(dataset),
        experiment_id="exp-retrain",
        work_dir=tmp_path / "work",
        runner=runner,
        config=config,
    )

    assert isinstance(solution, Solution)
    assert solution.model_type == "xgboost"
    assert metrics["metric"] == "unknown"  # fabricated package intentionally has no model.yaml
    assert (tmp_path / "work" / "model.tar.gz").exists()
