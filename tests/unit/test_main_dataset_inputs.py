"""Unit tests for main() dataset input handling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic_settings")
pytest.importorskip("pyspark")

import plexe.main as main_module


class _FakeIntegration:
    def __init__(self):
        self.workspace_calls: list[tuple[str, Path]] = []

    def prepare_workspace(self, experiment_id: str, work_dir: Path) -> None:
        self.workspace_calls.append((experiment_id, work_dir))

    def get_artifact_location(self, artifact_type: str, dataset_uri: str, experiment_id: str, work_dir: Path) -> str:
        _ = dataset_uri, experiment_id
        return str(work_dir / ".build" / "data" / artifact_type)

    def ensure_local(self, uris: list[str], work_dir: Path) -> list[str]:
        _ = work_dir
        return uris

    def prepare_original_model(self, model_reference: str, work_dir: Path) -> str:
        _ = work_dir
        return model_reference

    def on_checkpoint(self, experiment_id: str, phase_name: str, checkpoint_path: Path, work_dir: Path) -> None:
        _ = experiment_id, phase_name, checkpoint_path, work_dir

    def on_completion(self, experiment_id: str, work_dir: Path, final_metrics: dict, evaluation_report) -> None:
        _ = experiment_id, work_dir, final_metrics, evaluation_report

    def on_failure(self, experiment_id: str, error: Exception) -> None:
        _ = experiment_id, error

    def on_pause(self, phase_name: str) -> None:
        _ = phase_name


def _patch_main_dependencies(monkeypatch, build_model_spy: dict, normalize_calls: list[tuple]):
    class _FakeConfig(SimpleNamespace):
        def model_dump(self):
            return self.__dict__.copy()

        @classmethod
        def model_validate(cls, payload):
            return cls(**payload)

    fake_config = _FakeConfig(
        max_search_iterations=10,
        spark_mode="local",
        nn_max_epochs=10,
        nn_default_epochs=10,
        allowed_model_types=None,
        global_seed=None,
        csv_delimiter=",",
        csv_header=True,
        enable_otel=False,
        otel_endpoint=None,
        otel_headers={},
        routing_config=None,
        max_parallel_variants=1,
    )

    monkeypatch.setattr(main_module, "get_config", lambda: fake_config)
    monkeypatch.setattr(main_module, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "setup_litellm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "setup_opentelemetry", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "stop_spark_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main_module, "get_or_create_spark_session", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(main_module, "TreeSearchPolicy", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "LocalProcessRunner", lambda *args, **kwargs: object())

    def _fake_build_model(**kwargs):
        build_model_spy["kwargs"] = kwargs
        return SimpleNamespace(performance=0.82), {"performance": 0.82}, None

    monkeypatch.setattr(main_module, "build_model", _fake_build_model)

    class _FakeNormalizer:
        def __init__(self, _spark):
            pass

        def normalize(self, input_uri, output_uri, read_options):
            _ = read_options
            output_path = Path(output_uri)
            if output_path.name.endswith(".parquet"):
                split_name = output_path.stem
            else:
                split_name = "train"
            normalize_calls.append((split_name, input_uri))
            return f"normalized_{split_name}.parquet", SimpleNamespace(value="csv")

    monkeypatch.setattr(main_module, "DatasetNormalizer", _FakeNormalizer)


def test_main_prefers_train_dataset_uri_and_forwards_optional_splits(monkeypatch, tmp_path):
    fake_integration = _FakeIntegration()
    build_model_spy: dict = {}
    normalize_calls: list[tuple] = []
    _patch_main_dependencies(monkeypatch, build_model_spy, normalize_calls)

    result = main_module.main(
        intent="predict churn",
        train_dataset_uri="s3://bucket/new-train.csv",
        data_refs=["s3://bucket/legacy-train.csv"],
        val_dataset_uri="s3://bucket/val.csv",
        test_dataset_uri="s3://bucket/test.csv",
        integration=fake_integration,
        spark_mode="local",
        work_dir=tmp_path,
        user_id="user-1",
        experiment_id="exp-1",
        enable_final_evaluation=True,
    )

    assert result[0].performance == pytest.approx(0.82)
    assert fake_integration.workspace_calls == [("exp-1", tmp_path)]
    assert normalize_calls == [
        ("train", "s3://bucket/new-train.csv"),
        ("val", "s3://bucket/val.csv"),
        ("test", "s3://bucket/test.csv"),
    ]
    assert build_model_spy["kwargs"]["train_dataset_uri"] == "normalized_train.parquet"
    assert build_model_spy["kwargs"]["val_dataset_uri"] == "normalized_val.parquet"
    assert build_model_spy["kwargs"]["test_dataset_uri"] == "normalized_test.parquet"


def test_main_auto_enables_final_evaluation_when_test_dataset_is_provided(monkeypatch, tmp_path):
    fake_integration = _FakeIntegration()
    build_model_spy: dict = {}
    normalize_calls: list[tuple] = []
    _patch_main_dependencies(monkeypatch, build_model_spy, normalize_calls)

    main_module.main(
        intent="predict churn",
        train_dataset_uri="s3://bucket/train.csv",
        test_dataset_uri="s3://bucket/test.csv",
        integration=fake_integration,
        spark_mode="local",
        work_dir=tmp_path,
        user_id="user-1",
        experiment_id="exp-1",
    )

    assert normalize_calls == [("train", "s3://bucket/train.csv"), ("test", "s3://bucket/test.csv")]
    assert build_model_spy["kwargs"]["enable_final_evaluation"] is True


def test_main_nn_max_epochs_override_clamps_default_when_only_cap_is_set(monkeypatch, tmp_path):
    fake_integration = _FakeIntegration()
    build_model_spy: dict = {}
    normalize_calls: list[tuple] = []
    _patch_main_dependencies(monkeypatch, build_model_spy, normalize_calls)

    main_module.main(
        intent="predict churn",
        train_dataset_uri="s3://bucket/train.csv",
        nn_max_epochs=5,
        integration=fake_integration,
        spark_mode="local",
        work_dir=tmp_path,
        user_id="user-1",
        experiment_id="exp-1",
    )

    used_config = build_model_spy["kwargs"]["config"]
    assert used_config.nn_max_epochs == 5
    assert used_config.nn_default_epochs == 5


def test_main_uses_data_refs_fallback_when_train_dataset_uri_missing(monkeypatch, tmp_path):
    fake_integration = _FakeIntegration()
    build_model_spy: dict = {}
    normalize_calls: list[tuple] = []
    _patch_main_dependencies(monkeypatch, build_model_spy, normalize_calls)

    main_module.main(
        intent="predict churn",
        data_refs=["s3://bucket/train-a.csv", "s3://bucket/train-b.csv"],
        integration=fake_integration,
        spark_mode="local",
        work_dir=tmp_path,
        user_id="user-1",
        experiment_id="exp-1",
    )

    assert fake_integration.workspace_calls == [("exp-1", tmp_path)]
    assert normalize_calls == [("train", "s3://bucket/train-a.csv")]
    assert build_model_spy["kwargs"]["train_dataset_uri"] == "normalized_train.parquet"


def test_main_requires_train_dataset_uri_or_data_refs(monkeypatch, tmp_path):
    fake_integration = _FakeIntegration()
    build_model_spy: dict = {}
    normalize_calls: list[tuple] = []
    _patch_main_dependencies(monkeypatch, build_model_spy, normalize_calls)

    with pytest.raises(ValueError, match="train_dataset_uri is required"):
        main_module.main(
            intent="predict churn",
            train_dataset_uri=None,
            data_refs=None,
            integration=fake_integration,
            spark_mode="local",
            work_dir=tmp_path,
            user_id="user-1",
            experiment_id="exp-1",
        )
