"""
Model card template generator.

Builds a comprehensive MODEL_CARD.md from packaged artifacts and evaluation outputs.
"""

from __future__ import annotations

import json
import numbers
import re
from pathlib import Path
from typing import Any

import yaml


MODEL_CARD_TITLE = "Model Card"
DEFAULT_NOT_AVAILABLE = "Not available"


def generate_model_card(context, final_metrics: dict, evaluation_report: Any | None) -> str:
    """
    Generate a Markdown model card for the final package.

    Args:
        context: BuildContext
        final_metrics: Final metrics dict from evaluation
        evaluation_report: EvaluationReport or dict (optional)

    Returns:
        Markdown string
    """
    package_dir = Path(context.work_dir) / "model"

    model_metadata = _safe_load_yaml(package_dir / "model.yaml") or {}
    input_schema = _safe_load_json(package_dir / "schemas" / "input.json") or {}
    hyperparameters = _safe_load_json(package_dir / "config" / "hyperparameters.json") or {}
    evaluation_data = _normalize_evaluation_report(evaluation_report, package_dir) or {}

    task_analysis = context.task_analysis or {}
    stats = context.stats or {}

    model_type = model_metadata.get("model_type") or DEFAULT_NOT_AVAILABLE
    task_type = model_metadata.get("task_type") or task_analysis.get("task_type") or DEFAULT_NOT_AVAILABLE

    metric_name = _resolve_primary_metric_name(context, final_metrics, evaluation_data)
    metric_value = _resolve_primary_metric_value(final_metrics, evaluation_data)

    lines: list[str] = []
    lines.append(f"# {MODEL_CARD_TITLE}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append(f"- Intent: {context.intent or DEFAULT_NOT_AVAILABLE}")
    lines.append(f"- Task type: {task_type}")
    lines.append(f"- Model type: {model_type}")
    lines.append(f"- Primary metric ({metric_name}): {_format_metric(metric_value)}")
    lines.append("")

    # Dataset
    lines.append("## Dataset")
    train_samples = _get_nested(model_metadata, ["training", "train_samples"])
    val_samples = _get_nested(model_metadata, ["training", "val_samples"])
    features_count = _resolve_feature_count(input_schema, model_metadata, task_analysis)
    test_samples = final_metrics.get("test_samples")

    lines.append(f"- Training samples: {_format_count(train_samples)}")
    lines.append(f"- Validation samples: {_format_count(val_samples)}")
    lines.append(f"- Test samples: {_format_count(test_samples)}")
    lines.append(f"- Features: {_format_count(features_count)}")

    notable_characteristics = _collect_notable_characteristics(task_analysis, stats)
    if notable_characteristics:
        lines.append("Notable data characteristics:")
        for item in notable_characteristics:
            lines.append(f"- {item}")
    lines.append("")

    # Features Used
    lines.append("## Features Used")
    features = _resolve_features(input_schema, task_analysis)
    if features:
        lines.append(f"Input features ({len(features)}):")
        for feature in features:
            lines.append(f"- `{feature}`")
    else:
        lines.append(DEFAULT_NOT_AVAILABLE)

    explainability = _get_explainability_report(evaluation_data)
    feature_importance = _get_feature_importance(explainability)
    if feature_importance:
        method_used = _get_value(explainability, "method_used")
        if method_used:
            lines.append("")
            lines.append(f"Feature importance ({method_used}):")
        else:
            lines.append("")
            lines.append("Feature importance:")
        lines.extend(_format_feature_importance_table(feature_importance, explainability))

    lines.append("")

    # Excluded Columns
    excluded_columns = getattr(context, "excluded_columns", [])
    excluded_lines = _format_excluded_columns(excluded_columns)
    if excluded_lines:
        lines.append("## Excluded Columns")
        lines.extend(excluded_lines)
        lines.append("")

    # Performance
    lines.append("## Performance")
    lines.append(f"- Primary metric ({metric_name}): {_format_metric(metric_value)}")
    baseline_lines = _format_baseline_comparison(
        metric_name=metric_name,
        metric_value=metric_value,
        evaluation_data=evaluation_data,
        context=context,
        model_metadata=model_metadata,
    )
    if baseline_lines:
        lines.extend(baseline_lines)

    additional_metrics = _collect_additional_metrics(metric_name, final_metrics, evaluation_data)
    if additional_metrics:
        lines.append("Additional metrics:")
        lines.extend(_format_metrics_table(additional_metrics))
    else:
        lines.append("Additional metrics: Not available")
    lines.append("")

    # Evaluation Verdict
    lines.append("## Evaluation Verdict")
    if evaluation_data:
        verdict = evaluation_data.get("verdict") or DEFAULT_NOT_AVAILABLE
        deployment_ready = evaluation_data.get("deployment_ready")
        summary = evaluation_data.get("summary") or DEFAULT_NOT_AVAILABLE
        key_concerns = evaluation_data.get("key_concerns") or []

        lines.append(f"- Verdict: {verdict}")
        lines.append(
            f"- Deployment ready: {deployment_ready if deployment_ready is not None else DEFAULT_NOT_AVAILABLE}"
        )
        lines.append(f"- Summary: {summary}")
        if key_concerns:
            lines.append("Key concerns:")
            for concern in key_concerns:
                lines.append(f"- {concern}")

        recommendations = evaluation_data.get("recommendations") or []
        if recommendations:
            lines.append("Recommendations:")
            for rec in recommendations:
                lines.append(f"- {_format_recommendation(rec)}")
    else:
        lines.append(DEFAULT_NOT_AVAILABLE)
    lines.append("")

    # Hyperparameters
    lines.append("## Hyperparameters")
    if hyperparameters:
        lines.extend(_format_hyperparameters_table(hyperparameters))
    else:
        lines.append(DEFAULT_NOT_AVAILABLE)
    lines.append("")

    # Known Limitations
    lines.append("## Known Limitations")
    data_challenges = task_analysis.get("data_challenges") or []
    recommendations = evaluation_data.get("recommendations") if evaluation_data else []

    if data_challenges:
        lines.append("Data challenges:")
        for challenge in data_challenges:
            lines.append(f"- {challenge}")
    if recommendations:
        lines.append("Evaluation recommendations:")
        for rec in recommendations:
            lines.append(f"- {_format_recommendation(rec)}")

    if not data_challenges and not recommendations:
        lines.append(DEFAULT_NOT_AVAILABLE)
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    metadata = model_metadata.get("metadata", {})
    training_timestamp = metadata.get("created_at") or DEFAULT_NOT_AVAILABLE
    experiment_id = metadata.get("experiment_id") or context.experiment_id or DEFAULT_NOT_AVAILABLE
    plexe_version = _resolve_plexe_version() or metadata.get("version") or DEFAULT_NOT_AVAILABLE

    lines.append(f"- Experiment ID: {experiment_id}")
    lines.append(f"- Training timestamp: {training_timestamp}")
    lines.append(f"- Plexe version: {plexe_version}")

    return "\n".join(lines).strip() + "\n"


# ============================================
# Helpers
# ============================================


def _safe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
            return loaded if isinstance(loaded, dict) else None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _safe_load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            return loaded if isinstance(loaded, dict) else None
    except (yaml.YAMLError, OSError, UnicodeDecodeError):
        return None


def _normalize_evaluation_report(evaluation_report: Any | None, package_dir: Path) -> dict | None:
    if evaluation_report is None:
        fallback = _safe_load_json(package_dir / "evaluation" / "reports" / "evaluation.json")
        return _to_plain_dict(fallback)

    if isinstance(evaluation_report, dict):
        return _to_plain_dict(evaluation_report)

    if hasattr(evaluation_report, "to_dict"):
        try:
            return _to_plain_dict(evaluation_report.to_dict())
        except Exception:
            pass

    if hasattr(evaluation_report, "__dict__"):
        try:
            return _to_plain_dict(dict(evaluation_report.__dict__))
        except Exception:
            return None

    return None


def _to_plain_structure(value: Any) -> Any:
    if value is None or isinstance(value, str | bytes | bool | numbers.Real):
        return value
    if isinstance(value, dict):
        return {k: _to_plain_structure(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain_structure(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _to_plain_structure(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _to_plain_structure(dict(value.__dict__))
        except Exception:
            pass
    return value


def _to_plain_dict(value: Any) -> dict | None:
    normalized = _to_plain_structure(value)
    if isinstance(normalized, dict):
        return normalized
    return None


def _get_nested(data: dict, keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _get_value(data: Any, key: str) -> Any:
    if data is None:
        return None
    if isinstance(data, dict):
        return data.get(key)
    return getattr(data, key, None)


def _resolve_primary_metric_name(context, final_metrics: dict, evaluation_data: dict) -> str:
    core_metrics = evaluation_data.get("core_metrics") if evaluation_data else None
    metric_name = None
    if isinstance(core_metrics, dict):
        metric_name = core_metrics.get("primary_metric_name")
    if not metric_name and context.metric:
        metric_name = context.metric.name
    if not metric_name:
        metric_name = final_metrics.get("metric")
    return metric_name or DEFAULT_NOT_AVAILABLE


def _resolve_primary_metric_value(final_metrics: dict, evaluation_data: dict) -> Any:
    core_metrics = evaluation_data.get("core_metrics") if evaluation_data else None
    if isinstance(core_metrics, dict) and core_metrics.get("primary_metric_value") is not None:
        return core_metrics.get("primary_metric_value")
    return final_metrics.get("performance")


def _resolve_feature_count(input_schema: dict, model_metadata: dict, task_analysis: dict) -> int | None:
    if input_schema.get("properties"):
        return len(input_schema.get("properties", {}))
    count = _get_nested(model_metadata, ["training", "features_count"])
    if count is not None:
        return count
    input_description = task_analysis.get("input_description") or {}
    if isinstance(input_description, dict):
        return input_description.get("num_features")
    return None


def _resolve_features(input_schema: dict, task_analysis: dict) -> list[str]:
    if input_schema.get("properties"):
        return sorted(input_schema.get("properties", {}).keys())
    input_description = task_analysis.get("input_description") or {}
    if isinstance(input_description, dict):
        columns = input_description.get("feature_columns") or input_description.get("columns")
        if isinstance(columns, list):
            return columns
    return []


def _collect_notable_characteristics(task_analysis: dict, stats: dict) -> list[str]:
    notable = []
    key_insights = task_analysis.get("key_insights") or []
    if isinstance(key_insights, list):
        notable.extend(str(item) for item in key_insights if item)

    quality_issues = stats.get("quality_issues") or []
    if isinstance(quality_issues, list):
        notable.extend(str(item) for item in quality_issues if item)

    input_description = task_analysis.get("input_description")
    if not notable and isinstance(input_description, dict):
        summary = json.dumps(input_description, ensure_ascii=True)
        notable.append(f"Input description: {summary}")

    return notable


def _get_explainability_report(evaluation_data: dict) -> dict | None:
    if not evaluation_data:
        return None
    return evaluation_data.get("explainability") or evaluation_data.get("explainability_report")


def _get_feature_importance(explainability: dict | None) -> dict | None:
    if not explainability:
        return None
    feature_importance = explainability.get("feature_importance")
    if isinstance(feature_importance, dict) and feature_importance:
        return feature_importance
    return None


def _format_feature_importance_table(feature_importance: dict, explainability: dict | None) -> list[str]:
    lines = ["| Feature | Importance |", "| --- | --- |"]
    top_features = []
    if explainability:
        top_features = explainability.get("top_features") or []

    if top_features:
        for feature in top_features:
            value = feature_importance.get(feature)
            lines.append(f"| `{feature}` | {_format_metric(value)} |")
        return lines

    for feature, value in sorted(
        feature_importance.items(),
        key=lambda item: _feature_importance_sort_key(item[1]),
        reverse=True,
    ):
        lines.append(f"| `{feature}` | {_format_metric(value)} |")

    return lines


def _feature_importance_sort_key(value: Any) -> float:
    if _is_number(value):
        return float(value)
    return float("-inf")


def _format_excluded_columns(excluded_columns: Any) -> list[str]:
    if not excluded_columns:
        return []

    lines: list[str] = []
    if isinstance(excluded_columns, dict):
        for col, reason in excluded_columns.items():
            reason_text = _format_reason(reason)
            lines.append(_format_excluded_line(col, reason_text))
    elif isinstance(excluded_columns, list):
        for item in excluded_columns:
            if isinstance(item, dict):
                col = item.get("column") or item.get("name") or item.get("feature") or item.get("col")
                reason = item.get("reason") or item.get("issue") or item.get("notes") or item.get("why")
                if col:
                    reason_text = _format_reason(reason)
                    lines.append(_format_excluded_line(col, reason_text))
                else:
                    lines.append(f"- {json.dumps(item, ensure_ascii=True)}")
            else:
                lines.append(f"- `{item}`")
    else:
        lines.append(f"- `{excluded_columns}`")

    return lines


def _format_excluded_line(column: str, reason: str | None) -> str:
    if reason:
        return f"- `{column}` - {reason}"
    return f"- `{column}`"


def _format_reason(reason: Any) -> str | None:
    if reason is None:
        return None
    if isinstance(reason, list):
        return "; ".join(str(item) for item in reason if item)
    return str(reason)


def _format_baseline_comparison(
    metric_name: str,
    metric_value: Any,
    evaluation_data: dict,
    context,
    model_metadata: dict,
) -> list[str]:
    baseline_value = None
    baseline_name = None

    baseline_comparison = evaluation_data.get("baseline_comparison") if evaluation_data else None
    if isinstance(baseline_comparison, dict):
        baseline_name = baseline_comparison.get("baseline_name")
        baseline_performance = baseline_comparison.get("baseline_performance") or {}
        if isinstance(baseline_performance, dict):
            baseline_value = baseline_performance.get(metric_name)

    if baseline_value is None:
        baseline_value = _get_nested(model_metadata, ["metric", "baseline"])

    if baseline_value is None and getattr(context, "heuristic_baseline", None):
        baseline_value = context.heuristic_baseline.performance
        baseline_name = context.heuristic_baseline.name

    if baseline_value is None or metric_value is None:
        return []

    metric_direction = _get_nested(model_metadata, ["metric", "optimization_direction"])
    if not metric_direction and getattr(context, "metric", None):
        metric_direction = context.metric.optimization_direction

    improvement = _calculate_improvement(metric_value, baseline_value, metric_direction)
    pct_improvement = _calculate_percent_improvement(improvement, baseline_value)

    baseline_label = f" ({baseline_name})" if baseline_name else ""

    lines = [
        f"- Baseline{baseline_label}: {_format_metric(baseline_value)}",
        f"- Improvement over baseline: {_format_metric(improvement)}"
        + (f" ({_format_percent(pct_improvement)})" if pct_improvement is not None else ""),
    ]

    return lines


def _calculate_improvement(metric_value: Any, baseline_value: Any, direction: str | None) -> float | None:
    if not _is_number(metric_value) or not _is_number(baseline_value):
        return None

    if direction == "lower":
        return float(baseline_value) - float(metric_value)
    return float(metric_value) - float(baseline_value)


def _calculate_percent_improvement(improvement: float | None, baseline_value: Any) -> float | None:
    if improvement is None or not _is_number(baseline_value):
        return None
    if float(baseline_value) == 0:
        return None
    return (improvement / abs(float(baseline_value))) * 100


def _collect_additional_metrics(metric_name: str, final_metrics: dict, evaluation_data: dict) -> dict:
    core_metrics = evaluation_data.get("core_metrics") if evaluation_data else None
    all_metrics = {}

    if isinstance(core_metrics, dict):
        all_metrics.update(core_metrics.get("all_metrics") or {})
    if not all_metrics:
        all_metrics.update(final_metrics.get("all_metrics") or {})

    if metric_name in all_metrics:
        all_metrics = {k: v for k, v in all_metrics.items() if k != metric_name}

    return all_metrics


def _format_metrics_table(metrics: dict) -> list[str]:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for name, value in sorted(metrics.items()):
        lines.append(f"| `{name}` | {_format_metric(value)} |")
    return lines


def _format_hyperparameters_table(hyperparameters: dict) -> list[str]:
    lines = ["| Hyperparameter | Value |", "| --- | --- |"]
    for key in sorted(hyperparameters.keys()):
        value = hyperparameters[key]
        lines.append(f"| `{key}` | `{_stringify(value)}` |")
    return lines


def _format_recommendation(recommendation: Any) -> str:
    if isinstance(recommendation, dict):
        priority = recommendation.get("priority")
        action = recommendation.get("action")
        rationale = recommendation.get("rationale")
        if priority and action and rationale:
            return f"{priority}: {action} - {rationale}"
        if action and rationale:
            return f"{action} - {rationale}"
        if action:
            return str(action)
    return str(recommendation)


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _format_metric(value: Any) -> str:
    if value is None:
        return DEFAULT_NOT_AVAILABLE
    if _is_number(value):
        return f"{float(value):.4f}"
    return str(value)


def _format_count(value: Any) -> str:
    if value is None:
        return DEFAULT_NOT_AVAILABLE
    if _is_number(value):
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def _format_percent(value: float | None) -> str:
    if value is None:
        return DEFAULT_NOT_AVAILABLE
    return f"{value:.2f}%"


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real)


def _resolve_plexe_version() -> str | None:
    pyproject_path = _find_pyproject()
    if not pyproject_path:
        return None

    try:
        content = pyproject_path.read_text()
    except Exception:
        return None

    match = re.search(r"^version\s*=\s*\"([^\"]+)\"", content, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def _find_pyproject() -> Path | None:
    current = Path(__file__).resolve()
    for _ in range(6):
        candidate = current.parent / "pyproject.toml"
        if candidate.exists():
            return candidate
        current = current.parent
    return None
