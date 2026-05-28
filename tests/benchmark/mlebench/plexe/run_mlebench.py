"""Run Plexe inside the OpenAI MLE-bench agent container."""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
SUPPORTED_DATASET_SUFFIXES = (".csv", ".parquet", ".json", ".jsonl", ".tsv")


@dataclass(frozen=True)
class MLEBenchPaths:
    """Filesystem paths provided by the MLE-bench container."""

    data_dir: Path
    submission_dir: Path
    logs_dir: Path
    code_dir: Path
    work_dir: Path


def configure_logging(logs_dir: Path) -> None:
    """Configure console and file logging for the adapter."""

    logs_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / "plexe_mlebench.log"),
    ]
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers, force=True)


def resolve_paths() -> MLEBenchPaths:
    """Resolve MLE-bench container paths from environment variables."""

    code_dir = Path(os.environ.get("CODE_DIR", "/home/code"))
    return MLEBenchPaths(
        data_dir=Path(os.environ.get("DATA_DIR", "/home/data")),
        submission_dir=Path(os.environ.get("SUBMISSION_DIR", "/home/submission")),
        logs_dir=Path(os.environ.get("LOGS_DIR", "/home/logs")),
        code_dir=code_dir,
        work_dir=Path(os.environ.get("PLEXE_WORK_DIR", str(code_dir / "plexe-work"))),
    )


def read_competition_description(data_dir: Path) -> str:
    """Read the MLE-bench competition description."""

    description_path = data_dir / "description.md"
    if not description_path.exists():
        return "Build the best possible machine learning model for this competition."
    return description_path.read_text(encoding="utf-8")


def find_training_dataset(data_dir: Path) -> Path:
    """Find the most likely public training dataset in an MLE-bench data directory."""

    preferred_names = (
        "train.csv",
        "training.csv",
        "train.parquet",
        "training.parquet",
        "train.jsonl",
        "train.json",
        "train.tsv",
    )
    for name in preferred_names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate

    candidates = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES
        and "sample_submission" not in path.name.lower()
        and "submission" not in path.name.lower()
        and "test" not in path.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(f"No supported training dataset found under {data_dir}")
    return candidates[0]


def find_test_dataset(data_dir: Path) -> Path:
    """Find the most likely public test dataset in an MLE-bench data directory."""

    preferred_names = ("test.csv", "test.parquet", "test.jsonl", "test.json", "test.tsv")
    for name in preferred_names:
        candidate = data_dir / name
        if candidate.exists():
            return candidate

    candidates = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_DATASET_SUFFIXES
        and "sample_submission" not in path.name.lower()
        and "submission" not in path.name.lower()
        and "test" in path.name.lower()
    )
    if not candidates:
        raise FileNotFoundError(f"No supported test dataset found under {data_dir}")
    return candidates[0]


def find_sample_submission(data_dir: Path) -> Path | None:
    """Return a sample submission file when the competition provides one."""

    candidates = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and ("sample_submission" in path.name.lower() or path.name.lower() == "submission.csv")
    )
    return candidates[0] if candidates else None


def infer_id_column(sample_submission: Path | None, test_dataset: Path) -> str | None:
    """Infer the submission id column from sample submission or test data."""

    if sample_submission is not None:
        columns = list(pd.read_csv(sample_submission, nrows=0).columns)
        if columns:
            return columns[0]

    test_columns = list(read_tabular_sample(test_dataset, n_rows=1).columns)
    for candidate in ("id", "Id", "ID"):
        if candidate in test_columns:
            return candidate
    return test_columns[0] if test_columns else None


def read_tabular_sample(path: Path, n_rows: int | None = None) -> pd.DataFrame:
    """Read a small tabular sample for submission shaping."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=n_rows)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=n_rows)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.head(n_rows) if n_rows is not None else df
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl").head(n_rows)
    raise ValueError(f"Unsupported dataset suffix: {path.suffix}")


def load_predictor(package_dir: Path) -> Any:
    """Load Plexe's packaged predictor object from a completed model package."""

    model_yaml = package_dir / "model.yaml"
    predictor_file = package_dir / "predictor.py"
    if not model_yaml.exists() or not predictor_file.exists():
        raise FileNotFoundError(f"Missing packaged predictor files in {package_dir}")

    model_metadata = yaml.safe_load(model_yaml.read_text(encoding="utf-8")) or {}
    model_type = model_metadata.get("model_type")
    class_map = {
        "xgboost": "XGBoostPredictor",
        "catboost": "CatBoostPredictor",
        "lightgbm": "LightGBMPredictor",
        "keras": "KerasPredictor",
        "pytorch": "PyTorchPredictor",
    }
    class_name = class_map.get(model_type, f"{str(model_type).capitalize()}Predictor")

    spec = importlib.util.spec_from_file_location("plexe_mlebench_predictor", predictor_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import predictor from {predictor_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["plexe_mlebench_predictor"] = module
    spec.loader.exec_module(module)
    predictor_cls = getattr(module, class_name)
    return predictor_cls(model_dir=str(package_dir))


def coerce_predictions_to_submission(
    predictions: pd.DataFrame | pd.Series | list[Any],
    test_df: pd.DataFrame,
    sample_submission: Path | None,
    id_column: str | None,
) -> pd.DataFrame:
    """Shape Plexe predictions into the CSV columns expected by MLE-bench."""

    if isinstance(predictions, pd.Series):
        prediction_df = predictions.to_frame(name="prediction")
    elif isinstance(predictions, pd.DataFrame):
        prediction_df = predictions.copy()
    else:
        prediction_df = pd.DataFrame({"prediction": predictions})

    if sample_submission is not None:
        sample_df = pd.read_csv(sample_submission)
        output_columns = list(sample_df.columns)
        submission = pd.DataFrame(index=range(len(test_df)), columns=output_columns)
        if output_columns and id_column and id_column in test_df.columns:
            submission[output_columns[0]] = test_df[id_column].values
        elif output_columns:
            submission[output_columns[0]] = sample_df.iloc[: len(test_df), 0].values

        prediction_columns = output_columns[1:] or ["prediction"]
        for column in prediction_columns:
            source_column = column if column in prediction_df.columns else prediction_df.columns[0]
            submission[column] = prediction_df[source_column].values[: len(test_df)]
        return submission

    if id_column and id_column in test_df.columns:
        return pd.DataFrame({id_column: test_df[id_column].values, "prediction": prediction_df.iloc[:, 0].values})
    return pd.DataFrame({"prediction": prediction_df.iloc[:, 0].values})


def copy_existing_submission(work_dir: Path, submission_path: Path) -> bool:
    """Copy an existing submission.csv from Plexe artifacts if one exists."""

    candidates = sorted(work_dir.rglob("submission.csv"))
    for candidate in candidates:
        if candidate.resolve() != submission_path.resolve():
            shutil.copy2(candidate, submission_path)
            return True
    return False


def write_submission_from_package(
    work_dir: Path,
    test_dataset: Path,
    sample_submission: Path | None,
    submission_path: Path,
) -> None:
    """Create `submission.csv` by running the packaged Plexe predictor on test rows."""

    package_dir = work_dir / "model"
    predictor = load_predictor(package_dir)
    test_df = read_tabular_sample(test_dataset)
    id_column = infer_id_column(sample_submission, test_dataset)
    feature_df = test_df.drop(columns=[id_column], errors="ignore") if id_column else test_df
    predictions = predictor.predict(feature_df)
    submission = coerce_predictions_to_submission(predictions, test_df, sample_submission, id_column)
    submission.to_csv(submission_path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_metadata(paths: MLEBenchPaths, metadata: dict[str, Any]) -> None:
    """Persist adapter metadata for debugging and benchmark result review."""

    metadata_path = paths.logs_dir / "plexe_mlebench_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def run_plexe(paths: MLEBenchPaths, train_dataset: Path, description: str) -> None:
    """Run Plexe on the selected training dataset."""

    from plexe.main import main as plexe_main

    max_iterations = int(os.environ.get("PLEXE_MAX_ITERATIONS", "10"))
    provider = os.environ.get("PLEXE_PROVIDER")

    logging.info(
        "Running Plexe: train_dataset=%s max_iterations=%s provider=%s", train_dataset, max_iterations, provider
    )
    plexe_main(
        intent=description,
        train_dataset_uri=str(train_dataset),
        work_dir=paths.work_dir,
        user_id="mlebench",
        experiment_id=os.environ.get("COMPETITION_ID", "mlebench"),
        max_iterations=max_iterations,
        enable_final_evaluation=False,
    )


def main() -> None:
    """Execute the Plexe MLE-bench adapter."""

    paths = resolve_paths()
    configure_logging(paths.logs_dir)
    paths.submission_dir.mkdir(parents=True, exist_ok=True)
    paths.code_dir.mkdir(parents=True, exist_ok=True)
    paths.work_dir.mkdir(parents=True, exist_ok=True)

    submission_path = paths.submission_dir / "submission.csv"
    train_dataset = find_training_dataset(paths.data_dir)
    test_dataset = find_test_dataset(paths.data_dir)
    sample_submission = find_sample_submission(paths.data_dir)
    description = read_competition_description(paths.data_dir)

    metadata: dict[str, Any] = {
        "competition_id": os.environ.get("COMPETITION_ID"),
        "train_dataset": str(train_dataset),
        "test_dataset": str(test_dataset),
        "sample_submission": str(sample_submission) if sample_submission else None,
        "work_dir": str(paths.work_dir),
    }

    try:
        run_plexe(paths, train_dataset, description)
        if not copy_existing_submission(paths.work_dir, submission_path):
            write_submission_from_package(paths.work_dir, test_dataset, sample_submission, submission_path)
        metadata["submission_created"] = submission_path.exists()
        metadata["status"] = "ok"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        logging.exception("Plexe MLE-bench run failed")
        raise
    finally:
        write_metadata(paths, metadata)

    logging.info("Wrote MLE-bench submission to %s", submission_path)


if __name__ == "__main__":
    main()
