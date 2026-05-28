"""Tests for the Plexe MLE-bench adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ADAPTER_PATH = Path(__file__).parents[2] / "benchmark" / "mlebench" / "plexe" / "run_mlebench.py"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("plexe_mlebench_adapter", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_training_dataset_prefers_train_csv(tmp_path):
    adapter = _load_adapter()
    (tmp_path / "test.csv").write_text("id,x\n1,a\n", encoding="utf-8")
    (tmp_path / "train.csv").write_text("id,y\n1,0\n", encoding="utf-8")
    (tmp_path / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    assert adapter.find_training_dataset(tmp_path) == tmp_path / "train.csv"


def test_find_training_dataset_ignores_submission_and_test_files(tmp_path):
    adapter = _load_adapter()
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "sample_submission.csv").write_text("id,target\n1,0\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("id,x\n1,a\n", encoding="utf-8")
    (nested / "fold_train.parquet").write_text("not real parquet", encoding="utf-8")

    assert adapter.find_training_dataset(tmp_path) == nested / "fold_train.parquet"


def test_coerce_predictions_matches_sample_submission_columns(tmp_path):
    adapter = _load_adapter()
    sample_submission = tmp_path / "sample_submission.csv"
    sample_submission.write_text("PassengerId,Transported\n0001_01,False\n0002_01,False\n", encoding="utf-8")
    test_df = pd.DataFrame({"PassengerId": ["0001_01", "0002_01"], "Cabin": ["A/1/S", "B/2/P"]})
    predictions = pd.DataFrame({"prediction": [True, False]})

    submission = adapter.coerce_predictions_to_submission(
        predictions=predictions,
        test_df=test_df,
        sample_submission=sample_submission,
        id_column="PassengerId",
    )

    assert list(submission.columns) == ["PassengerId", "Transported"]
    assert submission["PassengerId"].tolist() == ["0001_01", "0002_01"]
    assert submission["Transported"].tolist() == [True, False]


def test_copy_existing_submission_ignores_destination(tmp_path):
    adapter = _load_adapter()
    work_dir = tmp_path / "work"
    submission_dir = tmp_path / "submission"
    work_dir.mkdir()
    submission_dir.mkdir()
    existing = work_dir / "nested" / "submission.csv"
    existing.parent.mkdir()
    existing.write_text("id,prediction\n1,0\n", encoding="utf-8")
    destination = submission_dir / "submission.csv"

    assert adapter.copy_existing_submission(work_dir, destination) is True
    assert destination.read_text(encoding="utf-8") == "id,prediction\n1,0\n"
