"""Shared fixtures and helpers for staged integration tests."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from plexe.checkpointing import load_checkpoint
from plexe.config import detect_installed_frameworks, get_config, setup_litellm, setup_logging
from plexe.constants import DirNames, PhaseNames
from plexe.execution.dataproc.dataset_io import DatasetNormalizer
from plexe.execution.dataproc.session import get_or_create_spark_session, stop_spark_session
from plexe.execution.training.local_runner import LocalProcessRunner
from plexe.integrations.standalone import StandaloneIntegration
from plexe.search.tree_policy import TreeSearchPolicy
from plexe.utils.tracing import setup_opentelemetry
from plexe.workflow import build_model

DATASET_SPECS: dict[str, dict[str, str]] = {
    "classification": {
        "dataset_relpath": "examples/datasets/spaceship-titanic/train.parquet",
        "intent": "predict whether a passenger was transported",
        "target_column": "Transported",
    },
    "regression": {
        "dataset_relpath": "examples/datasets/house-prices/train.csv",
        "intent": "predict house sale price",
        "target_column": "SalePrice",
    },
}

# Integration test matrix: keep this list explicit and simple.
INTEGRATION_MODEL_CANDIDATES = ["xgboost", "catboost", "lightgbm", "pytorch"]

MODEL_DATASET_KIND = {
    "xgboost": "classification",
    "catboost": "classification",
    "lightgbm": "classification",
    "pytorch": "regression",
}

PREDICTOR_CLASS_BY_MODEL = {
    "xgboost": "XGBoostPredictor",
    "catboost": "CatBoostPredictor",
    "lightgbm": "LightGBMPredictor",
    "pytorch": "PyTorchPredictor",
}

_installed_frameworks = set(detect_installed_frameworks())
INSTALLED_MODEL_TYPES = [m for m in INTEGRATION_MODEL_CANDIDATES if m in _installed_frameworks]
REQUIRED_SEED_DATASET_KINDS = list(dict.fromkeys(MODEL_DATASET_KIND[m] for m in INSTALLED_MODEL_TYPES))


def _build_model_type_params() -> list[Any]:
    """Return model-type params with explicit skips for missing optional frameworks."""
    params = []
    for model_type in INTEGRATION_MODEL_CANDIDATES:
        if model_type in _installed_frameworks:
            params.append(pytest.param(model_type, id=model_type))
        else:
            params.append(
                pytest.param(
                    model_type,
                    id=model_type,
                    marks=pytest.mark.skip(reason=f"{model_type} not installed"),
                )
            )
    return params


MODEL_TYPE_PARAMS = _build_model_type_params()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def run_id() -> str:
    """Return deterministic run identifier for staged artifacts."""
    return os.getenv("PLEXE_IT_RUN_ID", f"manual-{uuid.uuid4().hex[:8]}")


@pytest.fixture(scope="session")
def artifact_root(repo_root: Path, run_id: str) -> Path:
    """Return base path for staged integration artifacts."""
    root = repo_root / ".pytest_cache" / "integration" / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session", autouse=True)
def configure_integration_environment(repo_root: Path) -> None:
    """Set environment variables needed by the integration suite."""
    config_path = repo_root / "tests" / "integration" / "integration_config.yaml"
    os.environ["CONFIG_FILE"] = str(config_path)

    if not INSTALLED_MODEL_TYPES:
        pytest.skip("No supported integration model frameworks are installed")

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is required for tests/integration")


@pytest.fixture(scope="session", autouse=True)
def cleanup_spark_session() -> None:
    """Stop Spark session after tests complete."""
    yield
    stop_spark_session()


def seed_path(artifact_root: Path, dataset_kind: str) -> Path:
    """Return seed directory path for a dataset kind."""
    return artifact_root / "seeds" / dataset_kind


def model_run_path(artifact_root: Path, model_type: str) -> Path:
    """Return model-specific run directory path."""
    return artifact_root / "runs" / model_type


def checkpoint_file(work_dir: Path, phase_name: str) -> Path:
    """Return path to a checkpoint file."""
    return work_dir / "checkpoints" / f"{phase_name}.json"


def _replace_path_prefix(value: Any, old_root: str, new_root: str) -> Any:
    """Recursively replace old root path prefix with new root path."""
    if isinstance(value, str):
        return value.replace(old_root, new_root)
    if isinstance(value, dict):
        return {k: _replace_path_prefix(v, old_root, new_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_path_prefix(v, old_root, new_root) for v in value]
    return value


def copy_seed_to_model_run(seed_dir: Path, model_dir: Path) -> None:
    """
    Copy a seed workdir into a model run workdir and rewrite checkpoint paths.

    This prevents resumed runs from writing into the seed directory.
    """
    if not seed_dir.exists():
        raise FileNotFoundError(f"Seed directory does not exist: {seed_dir}")

    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(seed_dir, model_dir)

    seed_root = str(seed_dir)
    model_root = str(model_dir)
    checkpoints_dir = model_dir / "checkpoints"
    for checkpoint_path in checkpoints_dir.glob("*.json"):
        checkpoint_data = json.loads(checkpoint_path.read_text())
        context = checkpoint_data.get("context", {})
        old_work_dir = context.get("work_dir", seed_root)

        checkpoint_data = _replace_path_prefix(checkpoint_data, old_work_dir, model_root)
        if old_work_dir != seed_root:
            checkpoint_data = _replace_path_prefix(checkpoint_data, seed_root, model_root)

        checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2, default=str))


def assert_stage_prereqs(stage: str, artifact_root: Path) -> None:
    """Assert required artifacts from prior stages exist."""
    if stage == "search":
        missing = []
        for dataset_kind in REQUIRED_SEED_DATASET_KINDS:
            required = checkpoint_file(seed_path(artifact_root, dataset_kind), PhaseNames.BUILD_BASELINES)
            if not required.exists():
                missing.append(str(required))
        if missing:
            pytest.fail(
                "Stage 2 requires stage 1 seed checkpoints. Missing:\n" + "\n".join(f" - {path}" for path in missing)
            )

    if stage == "eval":
        missing = []
        for model_type in INSTALLED_MODEL_TYPES:
            required = checkpoint_file(model_run_path(artifact_root, model_type), PhaseNames.SEARCH_MODELS)
            if not required.exists():
                missing.append(str(required))
        if missing:
            pytest.fail(
                "Stage 3 requires stage 2 search checkpoints. Missing:\n" + "\n".join(f" - {path}" for path in missing)
            )


def _build_runtime_config(max_iterations: int, allowed_model_types: list[str] | None) -> Any:
    """Build runtime config used by staged integration tests."""
    config = get_config()
    config.spark_mode = "local"
    config.max_search_iterations = max_iterations
    if allowed_model_types is not None:
        config.allowed_model_types = allowed_model_types
    return config


def _normalize_dataset_uri_if_needed(
    spark,
    dataset_input: Path,
    config,
    integration: StandaloneIntegration,
    experiment_id: str,
    work_dir: Path,
) -> str:
    """Normalize non-parquet dataset inputs to parquet for build_model."""
    if dataset_input.suffix == ".parquet":
        return str(dataset_input)

    normalizer = DatasetNormalizer(spark)
    output_uri = integration.get_artifact_location("normalized", str(dataset_input), experiment_id, work_dir)
    csv_options = {"sep": config.csv_delimiter, "header": config.csv_header}
    normalized_uri, _ = normalizer.normalize(
        input_uri=str(dataset_input), output_uri=output_uri, read_options=csv_options
    )
    return normalized_uri


def build_seed_workflow(work_dir: Path, dataset_input: Path, intent: str, experiment_id: str) -> Any:
    """Run stages 1-3 and pause after baseline creation."""
    config = _build_runtime_config(max_iterations=1, allowed_model_types=INSTALLED_MODEL_TYPES)
    setup_logging(config)
    setup_litellm(config)
    setup_opentelemetry(config)

    work_dir.mkdir(parents=True, exist_ok=True)
    integration = StandaloneIntegration(user_id="integration_test")
    integration.prepare_workspace(experiment_id, work_dir)

    spark = get_or_create_spark_session(config)
    train_dataset_uri = _normalize_dataset_uri_if_needed(
        spark, dataset_input, config, integration, experiment_id, work_dir
    )

    runner = LocalProcessRunner(work_dir=str(work_dir / DirNames.BUILD_DIR / "search" / "runs"))
    search_policy = TreeSearchPolicy()

    return build_model(
        spark=spark,
        train_dataset_uri=train_dataset_uri,
        test_dataset_uri=None,
        user_id="integration_test",
        intent=intent,
        experiment_id=experiment_id,
        work_dir=work_dir,
        runner=runner,
        search_policy=search_policy,
        config=config,
        integration=integration,
        enable_final_evaluation=True,
        pause_points=[PhaseNames.BUILD_BASELINES],
    )


def _load_resume_context(work_dir: Path) -> dict[str, str]:
    """Load resume context from baseline or search checkpoints."""
    checkpoint_data = load_checkpoint(PhaseNames.SEARCH_MODELS, work_dir)
    if checkpoint_data is None:
        checkpoint_data = load_checkpoint(PhaseNames.BUILD_BASELINES, work_dir)
    if checkpoint_data is None:
        raise RuntimeError(f"Cannot resume; no suitable checkpoint in {work_dir}")

    context = checkpoint_data.get("context", {})
    return {
        "dataset_uri": context["dataset_uri"],
        "intent": context["intent"],
        "experiment_id": context["experiment_id"],
        "user_id": context["user_id"],
    }


def resume_workflow(
    work_dir: Path,
    allowed_model_types: list[str],
    pause_points: list[str] | None,
    enable_final_evaluation: bool,
    max_iterations: int = 1,
) -> Any:
    """Resume a staged integration workflow from existing checkpoints."""
    resume_context = _load_resume_context(work_dir)
    config = _build_runtime_config(max_iterations=max_iterations, allowed_model_types=allowed_model_types)
    setup_logging(config)
    setup_litellm(config)
    setup_opentelemetry(config)

    integration = StandaloneIntegration(user_id="integration_test")
    integration.prepare_workspace(resume_context["experiment_id"], work_dir)
    spark = get_or_create_spark_session(config)

    runner = LocalProcessRunner(work_dir=str(work_dir / DirNames.BUILD_DIR / "search" / "runs"))
    search_policy = TreeSearchPolicy()

    return build_model(
        spark=spark,
        train_dataset_uri=resume_context["dataset_uri"],
        test_dataset_uri=None,
        user_id=resume_context["user_id"],
        intent=resume_context["intent"],
        experiment_id=resume_context["experiment_id"],
        work_dir=work_dir,
        runner=runner,
        search_policy=search_policy,
        config=config,
        integration=integration,
        enable_final_evaluation=enable_final_evaluation,
        pause_points=pause_points,
    )


def load_predictor_class(model_dir: Path, model_type: str) -> type:
    """Load predictor class from packaged model/predictor.py."""
    predictor_path = model_dir / "predictor.py"
    module_name = f"predictor_{model_type}_{abs(hash(str(model_dir)))}"
    spec = importlib.util.spec_from_file_location(module_name, predictor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load predictor module at {predictor_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_name = PREDICTOR_CLASS_BY_MODEL[model_type]
    if not hasattr(module, class_name):
        raise RuntimeError(f"Predictor class {class_name} not found in {predictor_path}")
    return getattr(module, class_name)


def load_prediction_input(repo_root: Path, dataset_kind: str, n_rows: int = 8) -> pd.DataFrame:
    """Load a small feature sample used for predictor checks."""
    spec = DATASET_SPECS[dataset_kind]
    dataset_path = repo_root / spec["dataset_relpath"]
    if dataset_path.suffix == ".parquet":
        data = pd.read_parquet(dataset_path)
    else:
        data = pd.read_csv(dataset_path)

    target_column = spec["target_column"]
    features = data.drop(columns=[target_column]).head(n_rows)
    return features
