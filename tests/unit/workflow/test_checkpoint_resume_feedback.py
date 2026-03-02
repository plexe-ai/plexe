"""Tests for checkpoint resume feedback and persisted search journal behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from plexe.config import Config
from plexe.constants import PhaseNames
from plexe.models import Baseline, BuildContext, Metric, Solution
from plexe.search.journal import SearchJournal
import plexe.workflow as workflow


class _DummySparkDataFrame:
    def __init__(self, pdf: pd.DataFrame):
        self._pdf = pdf

    def count(self) -> int:
        return len(self._pdf)

    def limit(self, _rows: int) -> "_DummySparkDataFrame":
        return self

    def toPandas(self) -> pd.DataFrame:
        return self._pdf


class _DummySparkReader:
    def __init__(self, pdf: pd.DataFrame):
        self._pdf = pdf

    def parquet(self, _uri: str) -> _DummySparkDataFrame:
        return _DummySparkDataFrame(self._pdf)


class _DummySpark:
    def __init__(self, pdf: pd.DataFrame):
        self.read = _DummySparkReader(pdf)


class _DummySearchPolicy:
    def decide_next_solution(self, journal, context, iteration, max_iterations):  # noqa: D401
        return None

    def should_stop(self, journal, iteration, max_iterations):  # noqa: D401
        return True


class _DummyIntegration:
    def __init__(self, transformed_path: str):
        self._transformed_path = transformed_path

    def get_artifact_location(self, artifact_type, dataset_uri, experiment_id, work_dir):  # noqa: D401
        return self._transformed_path


class _DummyPlannerAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        plan = SimpleNamespace(
            parent_solution_id=-1,
            model=SimpleNamespace(model_type="xgboost"),
        )
        return [plan, plan]


class _DummyCoreMetrics:
    primary_metric_value = 0.91
    all_metrics = {"accuracy": 0.91}


class _DummyEvaluationReport:
    verdict = "PASS"
    deployment_ready = True
    summary = "ok"
    core_metrics = _DummyCoreMetrics()

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "deployment_ready": self.deployment_ready}


class _DummyModelEvaluatorAgent:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, solution, test_sample_df, predictor):  # noqa: D401
        return _DummyEvaluationReport()


def _make_context(tmp_path) -> BuildContext:
    return BuildContext(
        user_id="user-1",
        experiment_id="exp-1",
        dataset_uri="/tmp/dataset.parquet",
        work_dir=tmp_path,
        intent="predict churn",
    )


def test_search_models_preserves_user_feedback_for_all_variants(monkeypatch, tmp_path):
    context = _make_context(tmp_path)
    context.metric = Metric(name="accuracy", optimization_direction="higher")
    context.output_targets = ["target"]
    context.train_sample_uri = "/tmp/train.parquet"
    context.val_sample_uri = "/tmp/val.parquet"
    context.heuristic_baseline = Baseline(name="baseline", model_type="baseline", performance=0.5)
    feedback = {"comments": "Prefer neural networks over trees"}
    context.scratch["_user_feedback"] = feedback

    config = Config()
    config.max_search_iterations = 1
    config.max_parallel_variants = 2

    captured_feedback: list[dict | None] = []

    def _fake_execute_variant(
        plan,
        solution_id,
        journal,
        spark,
        config,
        runner,
        pipelines_dir,
        transformed_output_base,
        variant_context,
    ) -> Solution:
        captured_feedback.append(variant_context.scratch.get("_user_feedback"))
        return Solution(
            solution_id=solution_id,
            feature_pipeline=object(),
            model=object(),
            model_type="xgboost",
            performance=0.8,
        )

    monkeypatch.setattr(workflow, "PlannerAgent", _DummyPlannerAgent)
    monkeypatch.setattr(workflow, "_execute_variant", _fake_execute_variant)
    monkeypatch.setattr(workflow, "retrain_on_full_dataset", lambda **kwargs: kwargs["best_solution"])
    monkeypatch.setattr(workflow, "_save_phase_checkpoint", lambda *args, **kwargs: None)

    result = workflow.search_models(
        spark=object(),
        context=context,
        runner=object(),
        search_policy=_DummySearchPolicy(),
        config=config,
        integration=_DummyIntegration(str(tmp_path / "transformed")),
    )

    assert result is not None
    assert len(captured_feedback) == 2
    assert all(item == feedback for item in captured_feedback)


def test_evaluate_final_checkpoint_persists_search_journal(monkeypatch, tmp_path):
    context = _make_context(tmp_path)
    context.metric = Metric(name="accuracy", optimization_direction="higher")
    context.output_targets = ["target"]
    context.test_uri = "/tmp/test.parquet"

    search_journal = SearchJournal(baseline=Baseline(name="baseline", model_type="baseline", performance=0.5))
    context.scratch["_search_journal"] = search_journal

    model_dir = tmp_path / "solution_artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "predictor.py").write_text(
        "class XGBoostPredictor:\n" "    def __init__(self, model_dir):\n" "        self.model_dir = model_dir\n"
    )

    solution = Solution(
        solution_id=1,
        feature_pipeline=object(),
        model=object(),
        model_type="xgboost",
        model_artifacts_path=model_dir,
    )

    captured: dict = {}

    def _capture_checkpoint(
        phase_name,
        context,
        on_checkpoint_saved,
        search_journal=None,
        insight_store=None,
    ):
        captured["phase_name"] = phase_name
        captured["search_journal"] = search_journal

    monkeypatch.setattr(workflow, "ModelEvaluatorAgent", _DummyModelEvaluatorAgent)
    monkeypatch.setattr(workflow, "_save_phase_checkpoint", _capture_checkpoint)
    monkeypatch.setattr(workflow, "save_report", lambda *args, **kwargs: None)

    metrics = workflow.evaluate_final(
        spark=_DummySpark(pd.DataFrame({"feature": [1, 2], "target": [0, 1]})),
        context=context,
        solution=solution,
        config=Config(),
    )

    assert metrics is not None
    assert captured["phase_name"] == PhaseNames.EVALUATE_FINAL
    assert captured["search_journal"] is search_journal


def test_package_final_checkpoint_persists_search_journal(monkeypatch, tmp_path):
    context = _make_context(tmp_path)
    context.metric = Metric(name="accuracy", optimization_direction="higher")
    context.output_targets = ["target"]
    context.task_analysis = {"task_type": "binary_classification"}
    context.stats = {"total_rows": 2, "total_columns": 2}
    context.train_sample_uri = "/tmp/train_sample.parquet"
    context.heuristic_baseline = Baseline(name="baseline", model_type="baseline", performance=0.5)

    search_journal = SearchJournal(baseline=context.heuristic_baseline)
    context.scratch["_search_journal"] = search_journal

    model_artifacts_source = tmp_path / "model_artifacts_source"
    model_artifacts_source.mkdir(parents=True, exist_ok=True)
    (model_artifacts_source / "artifacts").mkdir(parents=True, exist_ok=True)
    (model_artifacts_source / "src").mkdir(parents=True, exist_ok=True)
    (model_artifacts_source / "src" / "pipeline.py").write_text("pipeline = None\n")
    (model_artifacts_source / "artifacts" / "metadata.json").write_text("{}")

    solution = Solution(
        solution_id=1,
        feature_pipeline=object(),
        model=object(),
        model_type="xgboost",
        model_artifacts_path=model_artifacts_source,
        performance=0.8,
    )

    captured: dict = {}

    def _capture_checkpoint(
        phase_name,
        context,
        on_checkpoint_saved,
        search_journal=None,
        insight_store=None,
    ):
        captured["phase_name"] = phase_name
        captured["search_journal"] = search_journal

    monkeypatch.setattr(workflow, "_save_phase_checkpoint", _capture_checkpoint)
    monkeypatch.setattr(workflow, "generate_model_card", lambda context, final_metrics, evaluation_report: "# Card\n")

    package_dir = workflow.package_final_model(
        spark=_DummySpark(pd.DataFrame({"feature": [1, 2], "target": [0, 1]})),
        context=context,
        solution=solution,
        final_metrics={"performance": 0.9, "test_samples": 2},
    )

    assert package_dir.exists()
    assert captured["phase_name"] == PhaseNames.PACKAGE_FINAL_MODEL
    assert captured["search_journal"] is search_journal
