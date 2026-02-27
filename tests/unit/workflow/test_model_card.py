"""Unit tests for model card generation."""

import json
from pathlib import Path

import yaml

from plexe.models import (
    Baseline,
    BaselineComparisonReport,
    BuildContext,
    CoreMetricsReport,
    DiagnosticReport,
    EvaluationReport,
    ExplainabilityReport,
    Metric,
    RobustnessReport,
)
from plexe.templates.packaging.model_card_template import generate_model_card


def _write_package_files(work_dir: Path) -> None:
    package_dir = work_dir / "model"
    (package_dir / "schemas").mkdir(parents=True, exist_ok=True)
    (package_dir / "config").mkdir(parents=True, exist_ok=True)

    input_schema = {
        "type": "object",
        "properties": {
            "age": {"type": "number"},
            "tenure": {"type": "number"},
            "balance": {"type": "number"},
        },
        "required": ["age", "tenure", "balance"],
    }
    (package_dir / "schemas" / "input.json").write_text(
        json.dumps(input_schema),
        encoding="utf-8",
    )

    hyperparameters = {"max_depth": 6, "learning_rate": 0.1, "subsample": 0.8}
    (package_dir / "config" / "hyperparameters.json").write_text(
        json.dumps(hyperparameters),
        encoding="utf-8",
    )

    model_metadata = {
        "model_format": "plexe_v1",
        "intent": "Predict churn",
        "model_type": "xgboost",
        "task_type": "binary_classification",
        "target_column": "churn",
        "output_targets": ["churn"],
        "metric": {
            "name": "roc_auc",
            "value": 0.85,
            "optimization_direction": "higher",
            "baseline": 0.6,
        },
        "training": {"features_count": 3, "train_samples": 800, "val_samples": 100},
        "metadata": {
            "created_at": "2025-01-01T00:00:00Z",
            "experiment_id": "exp_001",
            "user_id": "user_123",
            "trained_by": "plexe",
        },
    }
    (package_dir / "model.yaml").write_text(
        yaml.safe_dump(model_metadata, sort_keys=False),
        encoding="utf-8",
    )


def _make_full_context(work_dir: Path) -> BuildContext:
    context = BuildContext(
        user_id="user_123",
        experiment_id="exp_001",
        dataset_uri="s3://bucket/train.parquet",
        work_dir=work_dir,
        intent="Predict churn",
    )
    context.metric = Metric(name="roc_auc", optimization_direction="higher")
    context.task_analysis = {
        "task_type": "binary_classification",
        "data_challenges": ["class imbalance", "missing values"],
        "key_insights": ["Age correlates with churn", "Balance is highly predictive"],
        "input_description": {
            "type": "tabular",
            "num_features": 3,
            "feature_columns": ["age", "tenure", "balance"],
        },
    }
    context.stats = {
        "total_rows": 1000,
        "total_columns": 4,
        "quality_issues": ["age has 10% missing values"],
    }
    context.heuristic_baseline = Baseline(
        name="most_frequent",
        model_type="heuristic",
        performance=0.6,
        metadata={"strategy": "most_frequent"},
    )
    context.excluded_columns = [{"column": "leaky_feature", "reason": "data leakage"}]
    return context


def _make_evaluation_report() -> EvaluationReport:
    core_metrics = CoreMetricsReport(
        task_type="binary_classification",
        primary_metric_name="roc_auc",
        primary_metric_value=0.85,
        primary_metric_ci_lower=0.8,
        primary_metric_ci_upper=0.9,
        all_metrics={"roc_auc": 0.85, "accuracy": 0.8, "brier_score": 0.12},
        metric_confidence_intervals=None,
        statistical_notes="Solid performance",
        visualizations=None,
    )
    diagnostics = DiagnosticReport(
        worst_predictions=[],
        error_patterns=["Errors concentrated on low tenure"],
        subgroup_analysis=None,
        key_insights=["Misclassifications skew toward new customers"],
        error_distribution_summary="Errors cluster around tenure < 6 months",
    )
    robustness = RobustnessReport(
        perturbation_tests={"noise": {"impact": "low"}},
        consistency_score=0.92,
        robustness_grade="B",
        concerns=["Sensitive to rare categories"],
        recommendations=["Collect more rare-category samples"],
    )
    explainability = ExplainabilityReport(
        feature_importance={"age": 0.5, "tenure": 0.3, "balance": 0.2},
        method_used="shap",
        top_features=["age", "tenure", "balance"],
        confidence_intervals=None,
        interpretation="Age and tenure drive predictions",
    )
    baseline_comparison = BaselineComparisonReport(
        baseline_name="heuristic",
        baseline_type="heuristic",
        baseline_description="Most frequent class",
        baseline_performance={"roc_auc": 0.6},
        model_performance={"roc_auc": 0.85},
        performance_delta={"roc_auc": 0.25},
        performance_delta_pct={"roc_auc": 41.67},
        interpretation="Model outperforms baseline",
    )

    return EvaluationReport(
        verdict="PASS",
        summary="Model meets quality bar",
        deployment_ready=True,
        key_concerns=["Monitor drift"],
        core_metrics=core_metrics,
        diagnostics=diagnostics,
        robustness=robustness,
        explainability=explainability,
        baseline_comparison=baseline_comparison,
        recommendations=[{"priority": "HIGH", "action": "Monitor drift", "rationale": "Data shift risk"}],
    )


def test_generate_model_card_full_context(tmp_path: Path) -> None:
    _write_package_files(tmp_path)
    context = _make_full_context(tmp_path)
    evaluation_report = _make_evaluation_report()

    final_metrics = {"metric": "roc_auc", "performance": 0.85, "test_samples": 200, "all_metrics": {}}

    model_card = generate_model_card(context, final_metrics, evaluation_report)

    expected_headers = [
        "# Model Card",
        "## Summary",
        "## Dataset",
        "## Features Used",
        "## Excluded Columns",
        "## Performance",
        "## Evaluation Verdict",
        "## Hyperparameters",
        "## Known Limitations",
        "## Reproducibility",
    ]

    for header in expected_headers:
        assert header in model_card

    assert "`age`" in model_card
    assert "max_depth" in model_card
    assert "leaky_feature" in model_card


def test_generate_model_card_minimal_context(tmp_path: Path) -> None:
    context = BuildContext(
        user_id="user_min",
        experiment_id="exp_min",
        dataset_uri="s3://bucket/min.parquet",
        work_dir=tmp_path,
        intent="Predict outcomes",
    )

    final_metrics = {"metric": "accuracy", "performance": 0.5}

    model_card = generate_model_card(context, final_metrics, evaluation_report=None)

    expected_headers = [
        "# Model Card",
        "## Summary",
        "## Dataset",
        "## Features Used",
        "## Performance",
        "## Evaluation Verdict",
        "## Hyperparameters",
        "## Known Limitations",
        "## Reproducibility",
    ]

    for header in expected_headers:
        assert header in model_card

    assert "## Excluded Columns" not in model_card
    assert "Not available" in model_card
