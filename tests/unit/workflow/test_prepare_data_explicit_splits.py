"""Unit tests for prepare_data split resolution with explicit val/test datasets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("dataclasses_json")

from plexe.models import BuildContext
import plexe.workflow as workflow


class _DummyIntegration:
    def get_artifact_location(self, artifact_type, dataset_uri, experiment_id, work_dir):  # noqa: D401
        _ = dataset_uri, experiment_id
        return str(work_dir / ".build" / "data" / artifact_type)

    def ensure_local(self, uris, work_dir):  # noqa: D401
        _ = work_dir
        return uris


class _SchemaFrame:
    def __init__(self, columns: list[str]):
        self.columns = columns


class _SchemaReader:
    def __init__(self, schemas: dict[str, list[str]]):
        self._schemas = schemas

    def parquet(self, uri: str) -> _SchemaFrame:
        return _SchemaFrame(self._schemas[uri])


class _SchemaSpark:
    def __init__(self, schemas: dict[str, list[str]]):
        self.read = _SchemaReader(schemas)


def _make_context(tmp_path) -> BuildContext:
    context = BuildContext(
        user_id="user-1",
        experiment_id="exp-1",
        dataset_uri="train_input.parquet",
        work_dir=tmp_path,
        intent="predict churn",
    )
    context.output_targets = ["target"]
    context.task_analysis = {"recommended_split": {"ratios": {"train": 0.6, "val": 0.2, "test": 0.2}}}
    return context


def test_prepare_data_uses_all_provided_splits_without_running_splitter(monkeypatch, tmp_path):
    context = _make_context(tmp_path)
    config = SimpleNamespace(train_sample_size=100, val_sample_size=40)
    integration = _DummyIntegration()
    calls = {"materialize": [], "sampler": None}

    def _materialize(_spark, dataset_uri, split_name, _context, output_dir):
        calls["materialize"].append((split_name, dataset_uri))
        assert output_dir == str(tmp_path / ".build" / "data" / "splits")
        return f"copied_{split_name}.parquet"

    class _FailingSplitter:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Splitter should not be used when val and test are both provided")

    class _FakeSampler:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, train_uri, val_uri, train_sample_size, val_sample_size, output_dir):
            calls["sampler"] = (train_uri, val_uri, train_sample_size, val_sample_size, output_dir)
            return "train_sample.parquet", "val_sample.parquet"

    monkeypatch.setattr(workflow, "_materialize_explicit_split", _materialize)
    monkeypatch.setattr(workflow, "DatasetSplitterAgent", _FailingSplitter)
    monkeypatch.setattr(workflow, "SamplingAgent", _FakeSampler)
    monkeypatch.setattr(workflow, "_save_phase_checkpoint", lambda *args, **kwargs: None)

    workflow.prepare_data(
        spark=_SchemaSpark(
            {
                "train_input.parquet": ["feature", "target"],
                "copied_val.parquet": ["feature", "target", "val_only_column"],
                "copied_test.parquet": ["feature", "target", "test_only_column"],
            }
        ),
        training_dataset_uri="train_input.parquet",
        val_dataset_uri="val_input.parquet",
        test_dataset_uri="test_input.parquet",
        context=context,
        config=config,
        integration=integration,
        generate_test_set=True,
    )

    assert context.train_uri == "train_input.parquet"
    assert context.val_uri == "copied_val.parquet"
    assert context.test_uri == "copied_test.parquet"
    assert calls["materialize"] == [("val", "val_input.parquet"), ("test", "test_input.parquet")]
    assert calls["sampler"][0] == "train_input.parquet"
    assert calls["sampler"][1] == "copied_val.parquet"


def test_prepare_data_generates_missing_test_when_only_val_is_provided(monkeypatch, tmp_path):
    context = _make_context(tmp_path)
    config = SimpleNamespace(train_sample_size=100, val_sample_size=40)
    integration = _DummyIntegration()
    calls = {"split_ratios": None, "split_output_dir": None}

    def _materialize(_spark, dataset_uri, split_name, _context, output_dir):
        assert split_name == "val"
        assert output_dir == str(tmp_path / ".build" / "data" / "splits")
        return "copied_val.parquet"

    class _FakeSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, split_ratios, output_dir):
            calls["split_ratios"] = split_ratios
            calls["split_output_dir"] = output_dir
            return "split_train.parquet", "generated_test.parquet", None

    class _FakeSampler:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, train_uri, val_uri, train_sample_size, val_sample_size, output_dir):
            return "train_sample.parquet", "val_sample.parquet"

    monkeypatch.setattr(workflow, "_materialize_explicit_split", _materialize)
    monkeypatch.setattr(workflow, "DatasetSplitterAgent", _FakeSplitter)
    monkeypatch.setattr(workflow, "SamplingAgent", _FakeSampler)
    monkeypatch.setattr(workflow, "_save_phase_checkpoint", lambda *args, **kwargs: None)

    workflow.prepare_data(
        spark=_SchemaSpark({"train_input.parquet": ["feature", "target"], "copied_val.parquet": ["feature", "target"]}),
        training_dataset_uri="train_input.parquet",
        val_dataset_uri="val_input.parquet",
        test_dataset_uri=None,
        context=context,
        config=config,
        integration=integration,
        generate_test_set=True,
    )

    assert calls["split_ratios"] == {"train": 0.8, "val": 0.2}
    assert calls["split_output_dir"] == str(tmp_path / ".build" / "data" / "splits" / "generated")
    assert context.train_uri == "split_train.parquet"
    assert context.val_uri == "copied_val.parquet"
    assert context.test_uri == "generated_test.parquet"


def test_prepare_data_generates_missing_val_when_only_test_is_provided(monkeypatch, tmp_path):
    context = _make_context(tmp_path)
    config = SimpleNamespace(train_sample_size=100, val_sample_size=40)
    integration = _DummyIntegration()
    calls = {"split_ratios": None}

    def _materialize(_spark, dataset_uri, split_name, _context, output_dir):
        assert split_name == "test"
        assert output_dir == str(tmp_path / ".build" / "data" / "splits")
        return "copied_test.parquet"

    class _FakeSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, split_ratios, output_dir):
            calls["split_ratios"] = split_ratios
            return "split_train.parquet", "split_val.parquet", None

    class _FakeSampler:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, train_uri, val_uri, train_sample_size, val_sample_size, output_dir):
            return "train_sample.parquet", "val_sample.parquet"

    monkeypatch.setattr(workflow, "_materialize_explicit_split", _materialize)
    monkeypatch.setattr(workflow, "DatasetSplitterAgent", _FakeSplitter)
    monkeypatch.setattr(workflow, "SamplingAgent", _FakeSampler)
    monkeypatch.setattr(workflow, "_save_phase_checkpoint", lambda *args, **kwargs: None)

    workflow.prepare_data(
        spark=_SchemaSpark(
            {"train_input.parquet": ["feature", "target"], "copied_test.parquet": ["feature", "target"]}
        ),
        training_dataset_uri="train_input.parquet",
        val_dataset_uri=None,
        test_dataset_uri="test_input.parquet",
        context=context,
        config=config,
        integration=integration,
        generate_test_set=False,
    )

    assert calls["split_ratios"] == {"train": 0.8, "val": 0.2}
    assert context.train_uri == "split_train.parquet"
    assert context.val_uri == "split_val.parquet"
    assert context.test_uri == "copied_test.parquet"


@pytest.mark.parametrize(
    ("split_name", "val_dataset_uri", "test_dataset_uri", "materialized_uri", "missing_column"),
    [
        ("validation", "val_input.parquet", None, "copied_val.parquet", "target"),
        ("validation", "val_input.parquet", None, "copied_val.parquet", "feature"),
        ("test", None, "test_input.parquet", "copied_test.parquet", "target"),
        ("test", None, "test_input.parquet", "copied_test.parquet", "feature"),
    ],
)
def test_prepare_data_rejects_explicit_split_missing_training_column(
    monkeypatch, tmp_path, split_name, val_dataset_uri, test_dataset_uri, materialized_uri, missing_column
):
    context = _make_context(tmp_path)
    config = SimpleNamespace(train_sample_size=100, val_sample_size=40)
    schemas = {
        "train_input.parquet": ["feature", "target"],
        materialized_uri: [column for column in ("feature", "target") if column != missing_column],
    }

    monkeypatch.setattr(
        workflow,
        "_materialize_explicit_split",
        lambda _spark, _uri, _name, _context, _output_dir: materialized_uri,
    )

    class _UnexpectedDownstreamWork:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Explicit split schemas must be validated before downstream work")

    monkeypatch.setattr(workflow, "DatasetSplitterAgent", _UnexpectedDownstreamWork)
    monkeypatch.setattr(workflow, "SamplingAgent", _UnexpectedDownstreamWork)
    monkeypatch.setattr(workflow, "_save_phase_checkpoint", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match=f"{split_name}.*{missing_column}"):
        workflow.prepare_data(
            spark=_SchemaSpark(schemas),
            training_dataset_uri="train_input.parquet",
            val_dataset_uri=val_dataset_uri,
            test_dataset_uri=test_dataset_uri,
            context=context,
            config=config,
            integration=_DummyIntegration(),
            generate_test_set=True,
        )
