"""
Validation functions for pipelines, models, and other agent outputs.

Simple structural validation - no fuzzy reasoning.
"""

import os
import logging
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from plexe.config import ModelType

logger = logging.getLogger(__name__)


# ============================================
# Pipeline Validation
# ============================================


def validate_sklearn_pipeline(
    pipeline: Pipeline, sample_df: pd.DataFrame, target_columns: list[str]
) -> tuple[bool, str]:
    """
    Validate that an sklearn Pipeline is well-formed and functional.

    CRITICAL: Pipeline is validated on FEATURES ONLY (target columns removed).
    This ensures validation matches real-world usage where pipelines never see targets.

    Args:
        pipeline: Pipeline to validate
        sample_df: Small sample DataFrame for testing (may include target columns)
        target_columns: List of target column names to exclude from validation

    Returns:
        (is_valid, error_message)
    """

    # ============================================
    # Step 1: Type Check
    # ============================================
    if not isinstance(pipeline, Pipeline):
        return False, f"Must be sklearn.pipeline.Pipeline, got {type(pipeline)}"

    # ============================================
    # Step 2: Check for Spark-Incompatible Transformers
    # ============================================
    # Walk through all transformers in pipeline to detect problematic components
    incompatible = _check_spark_compatible_transformers(pipeline)
    if incompatible:
        return False, incompatible

    # ============================================
    # Step 3: Has Required Methods
    # ============================================
    if not hasattr(pipeline, "transform"):
        return False, "Pipeline must have 'transform' method"

    if not hasattr(pipeline, "fit"):
        return False, "Pipeline must have 'fit' method"

    # ============================================
    # Step 4: Prepare Features-Only Sample
    # ============================================
    # Remove target columns - pipeline should only see features
    sample_features = sample_df.drop(columns=target_columns, errors="ignore")

    # ============================================
    # Step 5: Test on Sample Data
    # ============================================
    try:
        # Fit pipeline on features only
        pipeline.fit(sample_features)

        # Transform sample
        transformed = pipeline.transform(sample_features)

        # Convert to DataFrame for validation
        transformed_df = pd.DataFrame(transformed)

        # Check for NaN values - ML models cannot handle them
        if transformed_df.isnull().any().any():
            return False, "Pipeline produces NaN values. All nulls must be handled via imputation or indicators."

        # Check output shape - row filtering is a data prep operation, not feature engineering
        if len(transformed_df) != len(sample_features):
            return (
                False,
                f"Pipeline changed number of rows: {len(sample_features)} → {len(transformed_df)}. Row filtering should happen in data preparation, not feature engineering.",
            )

        # Check output dtypes - object types are not ML-compatible
        object_cols = transformed_df.select_dtypes(include=["object"]).columns.tolist()
        if object_cols:
            return (
                False,
                f"Pipeline outputs object-type columns: {object_cols}. "
                f"Object dtypes are not ML-compatible. Use encoding/transformation to convert to primitive types.",
            )

    except Exception as e:
        return False, f"Pipeline transform failed: {str(e)}"

    return True, ""


def validate_pipeline_consistency(
    pipeline: Pipeline,
    train_sample: pd.DataFrame,
    val_sample: pd.DataFrame,
    target_columns: list[str],
) -> tuple[bool, str]:
    """
    Validate pipeline produces consistent output shape on train/val samples.

    This catches one-hot encoding issues where train/val have different categories,
    which is a common failure mode (e.g., CabinDeck_T appears in train but not val).

    Args:
        pipeline: Unfitted sklearn Pipeline
        train_sample: Training sample DataFrame (with targets)
        val_sample: Validation sample DataFrame (with targets)
        target_columns: List of target column names to exclude

    Returns:
        (is_valid, error_message)
    """

    # Prepare features-only samples
    train_features = train_sample.drop(columns=target_columns, errors="ignore")
    val_features = val_sample.drop(columns=target_columns, errors="ignore")

    try:
        # Fit on train
        pipeline.fit(train_features)

        # Transform both
        train_out = pipeline.transform(train_features)
        val_out = pipeline.transform(val_features)

        # Check shape consistency
        train_shape = train_out.shape
        val_shape = val_out.shape

        if train_shape[1] != val_shape[1]:
            return (
                False,
                f"Pipeline produces inconsistent feature counts: train={train_shape[1]}, val={val_shape[1]}. "
                f"This is usually caused by one-hot encoding creating different categories between splits. "
                f"Ensure consistent categories or use handle_unknown='ignore' in encoders.",
            )

        logger.info(f"Consistency check passed: train={train_shape}, val={val_shape}")

    except Exception as e:
        return False, f"Consistency validation failed: {str(e)}"

    return True, ""


# ============================================
# Model Definition Validation
# ============================================


def validate_xgboost_params(params: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate XGBoost hyperparameters.

    Args:
        params: XGBoost parameters dict

    Returns:
        (is_valid, error_message)
    """

    # ============================================
    # Step 1: Type Check
    # ============================================
    if not isinstance(params, dict):
        return False, f"XGBoost params must be dict, got {type(params)}"

    # ============================================
    # Step 2: Validate Common Params (Type/Range Only)
    # ============================================
    if "max_depth" in params:
        max_depth = params["max_depth"]
        if not isinstance(max_depth, int) or max_depth < 1:
            return False, f"max_depth must be positive integer, got {max_depth}"

    if "n_estimators" in params:
        n_estimators = params["n_estimators"]
        if not isinstance(n_estimators, int) or n_estimators < 1:
            return False, f"n_estimators must be positive integer, got {n_estimators}"

    if "learning_rate" in params:
        learning_rate = params["learning_rate"]
        if not isinstance(learning_rate, int | float) or learning_rate <= 0:
            return False, f"learning_rate must be positive number, got {learning_rate}"

    return True, ""


def validate_catboost_params(params: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate CatBoost hyperparameters.

    Args:
        params: CatBoost parameters dict

    Returns:
        (is_valid, error_message)
    """

    # ============================================
    # Step 1: Type Check
    # ============================================
    if not isinstance(params, dict):
        return False, f"CatBoost params must be dict, got {type(params)}"

    # ============================================
    # Step 2: Validate Common Params (Type/Range Only)
    # ============================================
    if "depth" in params:
        depth = params["depth"]
        if not isinstance(depth, int) or depth < 1 or depth > 16:
            return False, f"depth must be integer in [1, 16], got {depth}"

    if "iterations" in params:
        iterations = params["iterations"]
        if not isinstance(iterations, int) or iterations < 1:
            return False, f"iterations must be positive integer, got {iterations}"

    if "learning_rate" in params:
        learning_rate = params["learning_rate"]
        if not isinstance(learning_rate, int | float) or learning_rate <= 0:
            return False, f"learning_rate must be positive number, got {learning_rate}"

    if "l2_leaf_reg" in params:
        l2_leaf_reg = params["l2_leaf_reg"]
        if not isinstance(l2_leaf_reg, int | float) or l2_leaf_reg < 0:
            return False, f"l2_leaf_reg must be non-negative number, got {l2_leaf_reg}"

    return True, ""


def validate_model_definition(model_type: str, definition: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate model definition based on model type.

    Args:
        model_type: "xgboost", "catboost", "keras", or "pytorch"
        definition: Model configuration dict

    Returns:
        (is_valid, error_message)
    """

    if model_type == ModelType.XGBOOST:
        return validate_xgboost_params(definition)
    elif model_type == ModelType.CATBOOST:
        return validate_catboost_params(definition)
    else:
        return False, f"Unknown model type: {model_type}"


# ============================================
# Metric Function Validation
# ============================================


def validate_metric_function_object(func) -> tuple[bool, str]:
    """
    Validate metric computation function object.

    Args:
        func: Callable function

    Returns:
        (is_valid, error_message)
    """
    import inspect

    # Step 1: Check it's callable
    if not callable(func):
        return False, f"Must be callable function, got {type(func)}"

    # Step 2: Check function signature
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        if len(params) != 2:
            return False, f"Function must take exactly 2 arguments, got {len(params)}: {params}"

        if params != ["y_true", "y_pred"]:
            return False, f"Arguments must be named 'y_true' and 'y_pred', got {params}"

    except Exception as e:
        return False, f"Could not inspect function signature: {e}"

    # Step 3: Test function with sample data
    try:
        import numpy as np

        # Test with simple arrays
        y_true_test = np.array([0, 1, 0, 1, 1])
        y_pred_test = np.array([0, 1, 1, 1, 0])

        result = func(y_true_test, y_pred_test)

        # Check result is numeric
        if not isinstance(result, int | float | np.number):
            return False, f"Function must return numeric value, got {type(result)}"

    except Exception as e:
        return False, f"Function failed on test data: {e}"

    logger.info("Metric function validation passed")
    return True, ""


def validate_dataset_splits(
    spark, train_uri: str, val_uri: str, test_uri: str | None, expected_ratios: dict[str, float]
) -> tuple[bool, str]:
    """
    Validate that dataset splits were created correctly.

    Uses Spark to check existence, making this validation infrastructure-agnostic
    (works with local paths, S3 URIs, DBFS paths, etc.).

    Args:
        spark: SparkSession for reading parquet files
        train_uri: URI to train split (local path or s3://)
        val_uri: URI to validation split (local path or s3://)
        test_uri: URI to test split (None for 2-way splits)
        expected_ratios: Expected split ratios

    Returns:
        (is_valid, error_message)
    """
    # Validate row counts (existence check implicit - count() fails if dataset doesn't exist)
    try:
        train_count = spark.read.parquet(train_uri).count()
        val_count = spark.read.parquet(val_uri).count()

        if test_uri:
            test_count = spark.read.parquet(test_uri).count()
            total = train_count + val_count + test_count
        else:
            test_count = 0
            total = train_count + val_count
    except Exception as e:
        return False, f"Failed to read split datasets: {e}"

    logger.info(f"Split sizes: train={train_count}, val={val_count}, test={test_count}, total={total}")

    # Check ratios are within reasonable tolerance (10%)
    actual_ratios = {"train": train_count / total, "val": val_count / total, "test": test_count / total}

    # Only check splits that exist in actual_ratios (ignore extra keys like "rationale")
    for split_name in actual_ratios.keys():
        if split_name not in expected_ratios:
            continue
        expected = expected_ratios[split_name]
        actual = actual_ratios[split_name]
        diff = abs(actual - expected)

        if diff > 0.10:  # More than 10% off
            logger.warning(
                f"Split ratio significantly off for {split_name}: "
                f"expected {expected:.2%}, got {actual:.2%} (diff: {diff:.2%})"
            )
        elif diff > 0.05:  # More than 5% off
            logger.info(f"Split ratio slightly off for {split_name}: " f"expected {expected:.2%}, got {actual:.2%}")

    logger.info("Split validation passed")
    return True, ""


def _check_spark_compatible_transformers(pipeline: Pipeline) -> str | None:
    """
    Check if pipeline contains transformers that are incompatible with Spark distribution.

    Returns:
        Error message if incompatible transformers found, None otherwise
    """
    from sklearn.compose import ColumnTransformer

    def check_transformer(transformer, path=""):
        """Recursively check a transformer and its nested components."""
        # Get the actual transformer class
        transformer_class = type(transformer)
        module = transformer_class.__module__
        class_name = transformer_class.__name__

        # Check 1: Custom transformer classes (not from sklearn)
        if not module.startswith("sklearn"):
            return (
                f"Custom transformer '{class_name}' from module '{module}' found at {path or 'root'}.\n"
                f"SOLUTION: Use only standard sklearn transformers. Use FunctionTransformer for custom logic.\n"
                f"Available transformers: sklearn.preprocessing.*, sklearn.impute.*, sklearn.feature_selection.*, etc."
            )

        # Check 2: Recursively check ColumnTransformer
        if isinstance(transformer, ColumnTransformer):
            # Use transformers (not transformers_) - works on unfitted pipelines
            for name, sub_transformer, _ in transformer.transformers:
                if sub_transformer == "drop" or sub_transformer == "passthrough":
                    continue
                # Recursively check sub-transformers
                if isinstance(sub_transformer, Pipeline):
                    for step_name, step_transformer in sub_transformer.steps:
                        error = check_transformer(step_transformer, f"{path}/{name}/{step_name}")
                        if error:
                            return error
                else:
                    error = check_transformer(sub_transformer, f"{path}/{name}")
                    if error:
                        return error

        # Check 3: Recursively check Pipeline
        if isinstance(transformer, Pipeline):
            for step_name, step_transformer in transformer.steps:
                error = check_transformer(step_transformer, f"{path}/{step_name}")
                if error:
                    return error

        return None

    # Start checking from root pipeline
    return check_transformer(pipeline)


# ============================================
# Keras Validation
# ============================================


def validate_keras_model(model: Any, task_analysis: dict) -> tuple[bool, str]:
    """
    Validate Keras 3 model structure.

    Args:
        model: Keras model object to validate
        task_analysis: Task analysis with task_type, num_classes

    Returns:
        (is_valid, error_message)
    """
    # Set backend before importing keras
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    try:
        import keras
    except ImportError:
        return False, "keras not installed (required for keras model type)"

    # Check it's a Keras model
    if not isinstance(model, keras.Model):
        return False, f"Expected keras.Model or keras.Sequential, got {type(model)}"

    # Check model has been built (has layers)
    try:
        if not model.layers:
            return False, "Model has no layers"
    except Exception as e:
        return False, f"Could not access model.layers: {e}"

    # Check output shape matches task
    task_type = task_analysis.get("task_type", "")
    num_classes = task_analysis.get("num_classes", 0)

    try:
        output_shape = model.output_shape
        if output_shape is None:
            return False, "Model output_shape is None - model may not be built"

        # Output shape is (batch_size, output_dim) or just (batch_size,)
        output_dim = output_shape[-1] if len(output_shape) > 1 else 1

        if "classification" in task_type:
            if num_classes and output_dim != num_classes:
                return False, f"Classification task with {num_classes} classes but model output_dim={output_dim}"
        elif "regression" in task_type:
            if output_dim != 1:
                return False, f"Regression task but model output_dim={output_dim} (should be 1)"

    except Exception as e:
        return False, f"Could not validate output shape: {e}"

    return True, ""


def validate_keras_optimizer(optimizer: Any) -> tuple[bool, str]:
    """
    Validate Keras 3 optimizer.

    Args:
        optimizer: Optimizer object to validate

    Returns:
        (is_valid, error_message)
    """
    # Set backend before importing keras
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    try:
        import keras
    except ImportError:
        return False, "keras not installed (required for keras model type)"

    # Check it's a Keras optimizer
    if not isinstance(optimizer, keras.optimizers.Optimizer):
        return False, f"Expected keras.optimizers.Optimizer, got {type(optimizer)}"

    return True, ""


def validate_keras_loss(loss: Any) -> tuple[bool, str]:
    """
    Validate Keras 3 loss function.

    Args:
        loss: Loss object to validate

    Returns:
        (is_valid, error_message)
    """
    # Set backend before importing keras
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    try:
        import keras
    except ImportError:
        return False, "keras not installed (required for keras model type)"

    # Check it's a Keras loss
    if not isinstance(loss, keras.losses.Loss):
        return False, f"Expected keras.losses.Loss, got {type(loss)}"

    return True, ""
