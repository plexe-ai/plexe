"""
Helper functions for workflow.

Deterministic utilities for data operations and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, ndcg_score

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from plexe.config import ModelType, StandardMetric, DEFAULT_MODEL_TYPES, TASK_COMPATIBLE_MODELS
from plexe.models import DataLayout

logger = logging.getLogger(__name__)


def select_viable_model_types(data_layout: DataLayout, selected_frameworks: list[str] | None = None) -> list[str]:
    """
    Select viable model types using three-tier filtering.

    Implements the framework selection strategy:
    1. SELECTED FRAMEWORKS (user config + defaults)
    2. TASK-COMPATIBLE FRAMEWORKS (based on data layout)
    3. VIABLE FRAMEWORKS (intersection of 1 and 2)

    Args:
        data_layout: Physical structure of dataset (FLAT_NUMERIC, IMAGE_PATH, TEXT_STRING)
        selected_frameworks: User-selected frameworks, or None for defaults

    Returns:
        List of viable model type strings (e.g., ["xgboost", "catboost"])

    Raises:
        ValueError: If no compatible frameworks found
    """
    # Step 1: Selected frameworks (user override or defaults)
    if selected_frameworks is None:
        selected = DEFAULT_MODEL_TYPES
        logger.info(f"Using default model types: {selected}")
    else:
        selected = selected_frameworks
        logger.info(f"User-selected model types: {selected}")

    # Step 2: Task-compatible frameworks (based on data layout)
    task_compatible = TASK_COMPATIBLE_MODELS.get(data_layout, [])
    logger.info(f"Task-compatible model types for {data_layout.value}: {task_compatible}")

    # Step 3: Intersection (viable frameworks)
    viable = [mt for mt in selected if mt in task_compatible]

    if not viable:
        # No compatible frameworks - error with helpful message
        raise ValueError(
            f"No compatible model types for task.\n"
            f"Data layout: {data_layout.value}\n"
            f"Selected frameworks: {selected}\n"
            f"Task-compatible frameworks: {task_compatible}\n"
            f"Intersection: empty\n\n"
            f"Suggestion: For {data_layout.value} data, use one of: {task_compatible}"
        )

    logger.info(f"Viable model types: {viable}")
    return viable


def evaluate_on_sample(
    spark: SparkSession,
    sample_uri: str,
    model_artifacts_path: Path,
    model_type: str,
    metric: str,
    target_columns: list[str],
    group_column: str | None = None,
) -> float:
    """
    Evaluate model on sample (fast).

    Args:
        spark: SparkSession
        sample_uri: Sample data URI
        model_artifacts_path: Path to model artifacts
        model_type: "xgboost", "keras", or "pytorch"
        metric: Metric name
        target_columns: Target column names
        group_column: Optional group column for ranking metrics (query_id, session_id)

    Returns:
        Performance value
    """

    logger.info(f"Evaluating on sample with metric: {metric}")

    # Load Sample
    sample_df = spark.read.parquet(sample_uri).toPandas()

    # Extract group IDs if ranking task
    group_ids = sample_df[group_column].values if group_column and group_column in sample_df.columns else None

    # Use column names instead of positional indexing to handle target columns in any position
    columns_to_drop = list(target_columns)
    if group_column and group_column in sample_df.columns:
        columns_to_drop.append(group_column)

    X_sample = sample_df.drop(columns=columns_to_drop)
    y_sample = sample_df[target_columns[0]]

    # Load Predictor
    if model_type == ModelType.XGBOOST:
        from plexe.templates.inference.xgboost_predictor import XGBoostPredictor

        predictor = XGBoostPredictor(str(model_artifacts_path))
    elif model_type == ModelType.CATBOOST:
        from plexe.templates.inference.catboost_predictor import CatBoostPredictor

        predictor = CatBoostPredictor(str(model_artifacts_path))
    elif model_type == ModelType.KERAS:
        from plexe.templates.inference.keras_predictor import KerasPredictor

        predictor = KerasPredictor(str(model_artifacts_path))
    else:
        from plexe.templates.inference.pytorch_predictor import PyTorchPredictor

        predictor = PyTorchPredictor(str(model_artifacts_path))

    # Predict and compute metric on predictions
    predictions = predictor.predict(X_sample)["prediction"].values
    performance = compute_metric(y_sample, predictions, metric, group_ids=group_ids)

    logger.info(f"Sample performance ({metric}): {performance:.4f}")

    return performance


def compute_metric_hardcoded(y_true, y_pred, metric_name: str) -> float:
    """
    Compute metric using hardcoded sklearn implementations.

    Supports 30+ standard ML metrics.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        metric_name: Metric name

    Returns:
        Metric value

    Raises:
        ValueError: If metric not in StandardMetric enum
    """
    from sklearn.metrics import (
        precision_score,
        recall_score,
        log_loss,
        hamming_loss,
        matthews_corrcoef,
        cohen_kappa_score,
        mean_absolute_error,
        mean_absolute_percentage_error,
        median_absolute_error,
        max_error,
        explained_variance_score,
        roc_auc_score,
    )
    import numpy as np

    # Normalize to lowercase
    metric = metric_name.lower().strip()

    # Classification - Simple
    if metric == StandardMetric.ACCURACY.value:
        return float(accuracy_score(y_true, y_pred))

    # Classification - F1 variants
    elif metric in [StandardMetric.F1_SCORE.value, StandardMetric.F1_WEIGHTED.value]:
        return float(f1_score(y_true, y_pred, average="weighted"))
    elif metric == StandardMetric.F1_MACRO.value:
        return float(f1_score(y_true, y_pred, average="macro"))
    elif metric == StandardMetric.F1_MICRO.value:
        return float(f1_score(y_true, y_pred, average="micro"))

    # Classification - Precision variants
    elif metric in [StandardMetric.PRECISION.value, StandardMetric.PRECISION_WEIGHTED.value]:
        return float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    elif metric == StandardMetric.PRECISION_MACRO.value:
        return float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    elif metric == StandardMetric.PRECISION_MICRO.value:
        return float(precision_score(y_true, y_pred, average="micro", zero_division=0))

    # Classification - Recall variants
    elif metric in [StandardMetric.RECALL.value, StandardMetric.RECALL_WEIGHTED.value]:
        return float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    elif metric == StandardMetric.RECALL_MACRO.value:
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    elif metric == StandardMetric.RECALL_MICRO.value:
        return float(recall_score(y_true, y_pred, average="micro", zero_division=0))

    # Classification - ROC AUC variants
    elif metric in [StandardMetric.ROC_AUC.value, StandardMetric.ROC_AUC_OVR.value]:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_pred))
        else:
            return float(roc_auc_score(y_true, y_pred, multi_class="ovr"))
    elif metric == StandardMetric.ROC_AUC_OVO.value:
        return float(roc_auc_score(y_true, y_pred, multi_class="ovo"))

    # Classification - Other
    elif metric == StandardMetric.LOG_LOSS.value:
        return float(log_loss(y_true, y_pred))
    elif metric == StandardMetric.HAMMING_LOSS.value:
        return float(hamming_loss(y_true, y_pred))
    elif metric == StandardMetric.MATTHEWS_CORRCOEF.value:
        return float(matthews_corrcoef(y_true, y_pred))
    elif metric == StandardMetric.COHEN_KAPPA.value:
        return float(cohen_kappa_score(y_true, y_pred))

    # Regression
    elif metric == StandardMetric.RMSE.value:
        # Use numpy to avoid sklearn squared= parameter (deprecated in sklearn 1.4+)
        mse = np.mean((y_true - y_pred) ** 2)
        return float(np.sqrt(mse))
    elif metric == StandardMetric.MSE.value:
        return float(mean_squared_error(y_true, y_pred))
    elif metric == StandardMetric.MAE.value:
        return float(mean_absolute_error(y_true, y_pred))
    elif metric in [StandardMetric.R2_SCORE.value, "r2"]:
        return float(r2_score(y_true, y_pred))
    elif metric == StandardMetric.MAPE.value:
        return float(mean_absolute_percentage_error(y_true, y_pred))
    elif metric == StandardMetric.MEDIAN_ABSOLUTE_ERROR.value:
        return float(median_absolute_error(y_true, y_pred))
    elif metric == StandardMetric.MAX_ERROR.value:
        return float(max_error(y_true, y_pred))
    elif metric == StandardMetric.EXPLAINED_VARIANCE.value:
        return float(explained_variance_score(y_true, y_pred))

    # Ranking - Note: Requires special handling, see compute_metric() for group support
    elif metric in [StandardMetric.NDCG.value, StandardMetric.MAP.value, StandardMetric.MRR.value]:
        raise ValueError(
            f"Ranking metric '{metric_name}' requires group_ids parameter. "
            f"Use compute_metric(y_true, y_pred, metric_name, group_ids=...) instead of compute_metric_hardcoded()."
        )

    # Unsupported
    else:
        raise ValueError(
            f"Unsupported metric: '{metric_name}'. "
            f"Supported: {[m.value for m in StandardMetric]}. "
            f"For custom metrics, use MetricImplementationAgent."
        )


def compute_metric(y_true, y_pred, metric_name: str, group_ids=None) -> float:
    """
    Compute metric value.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        metric_name: Metric name
        group_ids: Optional group IDs for ranking metrics (query_id, session_id)

    Returns:
        Metric value
    """
    # Handle ranking metrics separately (require grouping)
    metric = metric_name.lower().strip()

    if metric == StandardMetric.NDCG.value:
        if group_ids is None:
            # No grouping - treat as single query (fallback)
            logger.warning("NDCG computed without group_ids - treating all samples as single query")
            y_true_2d = np.array([y_true]) if y_true.ndim == 1 else y_true
            y_pred_2d = np.array([y_pred]) if y_pred.ndim == 1 else y_pred
            return float(ndcg_score(y_true_2d, y_pred_2d))

        # Proper per-query NDCG
        df = pd.DataFrame({"group": group_ids, "true": y_true, "pred": y_pred})
        ndcg_scores = []

        for _, group_df in df.groupby("group"):
            if len(group_df) == 1:
                # Single item: NDCG=1.0 if relevant, 0.0 if not
                ndcg_scores.append(1.0 if group_df["true"].iloc[0] > 0 else 0.0)
            else:
                # Multiple items: compute NDCG
                y_true_group = np.array([group_df["true"].values])
                y_pred_group = np.array([group_df["pred"].values])
                ndcg_scores.append(ndcg_score(y_true_group, y_pred_group))

        return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

    elif metric == StandardMetric.MAP.value:
        # Mean Average Precision
        if group_ids is None:
            logger.warning("MAP computed without group_ids - treating all samples as single query")
            # Single query MAP
            sorted_indices = np.argsort(y_pred)[::-1]
            sorted_true = y_true[sorted_indices]
            precisions = []
            relevant_count = 0
            for i, rel in enumerate(sorted_true):
                if rel > 0:
                    relevant_count += 1
                    precisions.append(relevant_count / (i + 1))
            return float(np.mean(precisions)) if precisions else 0.0

        # Per-query MAP
        df = pd.DataFrame({"group": group_ids, "true": y_true, "pred": y_pred})
        map_scores = []

        for _, group_df in df.groupby("group"):
            if len(group_df) == 1:
                # Single item: MAP=1.0 if relevant, 0.0 if not
                map_scores.append(1.0 if group_df["true"].iloc[0] > 0 else 0.0)
            else:
                # Multiple items: compute MAP
                sorted_indices = np.argsort(group_df["pred"].values)[::-1]
                sorted_true = group_df["true"].values[sorted_indices]
                precisions = []
                relevant_count = 0
                for i, rel in enumerate(sorted_true):
                    if rel > 0:
                        relevant_count += 1
                        precisions.append(relevant_count / (i + 1))
                # Include groups with no relevant items (MAP=0.0)
                map_scores.append(np.mean(precisions) if precisions else 0.0)

        return float(np.mean(map_scores)) if map_scores else 0.0

    elif metric == StandardMetric.MRR.value:
        # Mean Reciprocal Rank
        if group_ids is None:
            logger.warning("MRR computed without group_ids - treating all samples as single query")
            # Single query MRR (reciprocal rank of first relevant item)
            sorted_indices = np.argsort(y_pred)[::-1]
            sorted_true = y_true[sorted_indices]
            for i, rel in enumerate(sorted_true):
                if rel > 0:
                    return float(1.0 / (i + 1))
            return 0.0

        # Per-query MRR
        df = pd.DataFrame({"group": group_ids, "true": y_true, "pred": y_pred})
        mrr_scores = []

        for _, group_df in df.groupby("group"):
            if len(group_df) == 1:
                # Single item: MRR=1.0 if relevant, 0.0 if not
                mrr_scores.append(1.0 if group_df["true"].iloc[0] > 0 else 0.0)
            else:
                # Multiple items: find first relevant
                sorted_indices = np.argsort(group_df["pred"].values)[::-1]
                sorted_true = group_df["true"].values[sorted_indices]
                for i, rel in enumerate(sorted_true):
                    if rel > 0:
                        mrr_scores.append(1.0 / (i + 1))
                        break
                else:
                    # No relevant items in this group
                    mrr_scores.append(0.0)

        return float(np.mean(mrr_scores)) if mrr_scores else 0.0

    # Non-ranking metrics use existing implementation
    return compute_metric_hardcoded(y_true, y_pred, metric_name)
