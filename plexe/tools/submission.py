"""
Submission tools for agents.

Mix of @tool (for structured inputs) and @agentinspectable (for complex objects).
"""

import logging
import types
from typing import Any

import pandas as pd
from plexe.utils.tooling import agentinspectable
from sklearn.pipeline import Pipeline
from smolagents import tool

from plexe.constants import DirNames
from plexe.models import BuildContext, Metric, Hypothesis, UnifiedPlan
from plexe.search.insight_store import InsightStore
from plexe.utils.tracing import tool_span
from plexe.validation.validators import validate_sklearn_pipeline, validate_pipeline_consistency

logger = logging.getLogger(__name__)


def get_save_pipeline_fn(context: BuildContext, sample_df: pd.DataFrame):
    """
    Factory: Returns pipeline submission function.

    Args:
        context: Build context for storing result
        sample_df: Sample data for validation

    Returns:
        Configured save_pipeline function
    """

    @tool_span
    @agentinspectable
    def save_pipeline(pipeline: Pipeline) -> str:
        """
        Submit your feature engineering pipeline.

        This function validates and saves your sklearn Pipeline.

        Args:
            pipeline: sklearn.pipeline.Pipeline object

        Returns:
            Confirmation message

        Raises:
            ValueError: If pipeline validation fails
        """

        # Step 1: Validate
        is_valid, error_msg = validate_sklearn_pipeline(pipeline, sample_df, context.output_targets)

        if not is_valid:
            logger.error(f"Pipeline validation failed: {error_msg}")
            raise ValueError(f"Pipeline validation failed: {error_msg}")

        # Step 2: Save to Context
        context.scratch["_saved_pipeline"] = pipeline

        logger.info("Pipeline validated and saved successfully")
        return "Pipeline validated and saved successfully"

    return save_pipeline


def get_save_pipeline_code_tool(context: BuildContext, train_sample_df: pd.DataFrame, val_sample_df: pd.DataFrame):
    """
    Factory: Returns pipeline code submission tool.

    Args:
        context: Build context for storing result
        train_sample_df: Training sample data for validation (features only, no target)
        val_sample_df: Validation sample data for consistency checking (features only, no target)

    Returns:
        Configured save_pipeline_code function
    """

    @tool
    @tool_span
    def save_pipeline_code(code: str) -> str:
        """
        Submit your feature engineering pipeline as code.

        This function validates and saves your sklearn Pipeline code.
        Your code MUST define a variable named 'pipeline'.

        Args:
            code: Python code string that creates sklearn Pipeline and assigns it to variable 'pipeline'.

        Returns:
            Confirmation message

        Raises:
            ValueError: If pipeline validation fails
        """

        # Step 1: Check for forbidden patterns
        # Ban pd.get_dummies() - it's non-deterministic in FunctionTransformer
        if "get_dummies" in code:
            error_msg = (
                "Pipeline code contains 'get_dummies()' which is FORBIDDEN.\n\n"
                "PROBLEM: pd.get_dummies() is non-deterministic when used in FunctionTransformer - "
                "it creates different columns based on whatever categories are present in the current data, "
                "causing train/val feature count mismatches.\n\n"
                "SOLUTION: Use sklearn.preprocessing.OneHotEncoder instead:\n"
                "  from sklearn.preprocessing import OneHotEncoder\n"
                "  from sklearn.compose import ColumnTransformer\n\n"
                "  encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n"
                "  # Or use ColumnTransformer for multiple categorical columns\n\n"
                "OneHotEncoder learns categories during fit() and applies them consistently during transform()."
            )
            logger.debug(error_msg)
            raise ValueError(error_msg)

        # Step 2: Execute code to get pipeline object
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            error_msg = f"Code execution failed: {e}\n\nGenerated code:\n{code[:500]}..."
            logger.debug(error_msg)
            raise ValueError(error_msg)

        # Step 2: Check that 'pipeline' variable exists
        if "pipeline" not in namespace:
            error_msg = "Code must define a variable named 'pipeline'"
            logger.debug(error_msg)
            raise ValueError(error_msg)

        pipeline = namespace["pipeline"]

        # Step 3: Basic validation on train sample
        is_valid, error_msg = validate_sklearn_pipeline(pipeline, train_sample_df, context.output_targets)

        if not is_valid:
            logger.debug(f"Pipeline validation failed: {error_msg}")
            raise ValueError(f"Pipeline validation failed: {error_msg}")

        # Step 4: Consistency validation (train vs val feature counts)
        is_valid, error_msg = validate_pipeline_consistency(
            pipeline, train_sample_df, val_sample_df, context.output_targets
        )

        if not is_valid:
            logger.debug(f"Pipeline consistency validation failed: {error_msg}")
            raise ValueError(f"Pipeline consistency validation failed: {error_msg}")

        # Step 5: Save both code and object to context
        context.scratch["_saved_pipeline"] = pipeline
        context.scratch["_saved_pipeline_code"] = code

        logger.info("Pipeline code validated and saved successfully")
        return "Pipeline code validated and saved successfully"

    return save_pipeline_code


def get_save_model_fn(context: BuildContext, model_type: str, max_epochs: int = None):
    """
    Factory: Returns model submission function.

    Args:
        context: Build context for storing result
        model_type: "xgboost", "keras", or "pytorch"
        max_epochs: Maximum allowed epochs for neural nets (enforced in validation)

    Returns:
        Configured save_model function (signature varies by model_type)
    """

    def _validate_training_params(epochs: int, batch_size: int) -> None:
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"epochs must be integer >= 1, got {epochs}")

        if max_epochs is not None and epochs > max_epochs:
            raise ValueError(f"epochs must be ≤ {max_epochs} (configured cap), got {epochs}")

        if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 1024:
            raise ValueError(f"batch_size must be integer in [1, 1024], got {batch_size}")

    def _save_nn_components(model: Any, optimizer: Any, loss: Any, epochs: int, batch_size: int) -> str:
        context.scratch["_saved_model"] = model
        context.scratch["_saved_optimizer"] = optimizer
        context.scratch["_saved_loss"] = loss
        context.scratch["_nn_epochs"] = epochs
        context.scratch["_nn_batch_size"] = batch_size

        logger.info(
            f"{model_type} components saved: model={type(model).__name__}, optimizer={type(optimizer).__name__}, loss={type(loss).__name__}, epochs={epochs}, batch_size={batch_size}"
        )
        return f"{model_type} model saved: {epochs} epochs, batch_size={batch_size}"

    if model_type == "keras":
        # Neural networks need model + optimizer + loss + training params
        @tool_span
        @agentinspectable
        def save_model(model: Any, optimizer: Any, loss: Any, epochs: int, batch_size: int) -> str:
            """
            Submit your Keras model, optimizer, loss, and training configuration.

            This function validates and saves all components needed for training.

            Args:
                model: Keras model instance (keras.Model)
                optimizer: Optimizer instance (keras.optimizers.Optimizer)
                loss: Loss instance (keras.losses.Loss)
                epochs: Number of training epochs (e.g., 50)
                batch_size: Batch size for training (e.g., 32)

            Returns:
                Confirmation message

            Raises:
                ValueError: If validation fails
            """
            from plexe.validation.validators import (
                validate_keras_model,
                validate_keras_optimizer,
                validate_keras_loss,
            )

            is_valid, error_msg = validate_keras_model(model, context.task_analysis)
            if not is_valid:
                logger.debug(f"Keras model validation failed: {error_msg}")
                raise ValueError(f"Keras model validation failed: {error_msg}")

            is_valid, error_msg = validate_keras_optimizer(optimizer)
            if not is_valid:
                logger.debug(f"Keras optimizer validation failed: {error_msg}")
                raise ValueError(f"Keras optimizer validation failed: {error_msg}")

            is_valid, error_msg = validate_keras_loss(loss)
            if not is_valid:
                logger.debug(f"Keras loss validation failed: {error_msg}")
                raise ValueError(f"Keras loss validation failed: {error_msg}")

            _validate_training_params(epochs, batch_size)
            return _save_nn_components(model, optimizer, loss, epochs, batch_size)

        return save_model

    elif model_type == "pytorch":
        # Neural networks need model + optimizer + loss + training params
        @tool_span
        @agentinspectable
        def save_model(model: Any, optimizer: Any, loss: Any, epochs: int, batch_size: int) -> str:
            """
            Submit your PyTorch model, optimizer, loss, and training configuration.

            This function validates and saves all components needed for training.

            Args:
                model: PyTorch model instance (torch.nn.Module)
                optimizer: Optimizer instance (torch.optim.Optimizer)
                loss: Loss instance (torch.nn.Module)
                epochs: Number of training epochs (e.g., 50)
                batch_size: Batch size for training (e.g., 32)

            Returns:
                Confirmation message

            Raises:
                ValueError: If validation fails
            """
            import torch.nn as nn
            import torch.optim

            if not isinstance(model, nn.Module):
                error_msg = f"Expected torch.nn.Module, got {type(model)}"
                logger.debug(error_msg)
                raise ValueError(error_msg)

            if not isinstance(optimizer, torch.optim.Optimizer):
                error_msg = f"Expected torch.optim.Optimizer, got {type(optimizer)}"
                logger.debug(error_msg)
                raise ValueError(error_msg)

            if not isinstance(loss, nn.Module):
                error_msg = f"Expected torch.nn.Module (loss function), got {type(loss)}"
                logger.debug(error_msg)
                raise ValueError(error_msg)

            _validate_training_params(epochs, batch_size)
            return _save_nn_components(model, optimizer, loss, epochs, batch_size)

        return save_model

    else:
        # XGBoost/CatBoost/LightGBM: just model object
        @tool_span
        @agentinspectable
        def save_model(model: Any) -> str:
            """
            Submit your model object.

            This function validates and saves your XGBoost, CatBoost, or LightGBM model object.

            Args:
                model: Model object (XGBClassifier, XGBRegressor, CatBoostClassifier, CatBoostRegressor, or LGBMClassifier, etc)

            Returns:
                Confirmation message

            Raises:
                ValueError: If model validation fails
            """

            # Validate model type
            if model_type == "xgboost":
                from xgboost import XGBClassifier, XGBRegressor, XGBRanker

                if not isinstance(model, XGBClassifier | XGBRegressor | XGBRanker):
                    error_msg = f"Expected XGBClassifier, XGBRegressor, or XGBRanker, got {type(model)}"
                    logger.debug(error_msg)
                    raise ValueError(error_msg)

                logger.info(f"XGBoost model validated: {type(model).__name__}")

            elif model_type == "catboost":
                from catboost import CatBoostClassifier, CatBoostRegressor

                if not isinstance(model, CatBoostClassifier | CatBoostRegressor):
                    error_msg = f"Expected CatBoostClassifier or CatBoostRegressor, got {type(model)}"
                    logger.debug(error_msg)
                    raise ValueError(error_msg)

                logger.info(f"CatBoost model validated: {type(model).__name__}")

            elif model_type == "lightgbm":
                from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker

                if not isinstance(model, LGBMClassifier | LGBMRegressor | LGBMRanker):
                    error_msg = f"Expected LGBMClassifier, LGBMRegressor, or LGBMRanker, got {type(model)}"
                    logger.debug(error_msg)
                    raise ValueError(error_msg)

                logger.info(f"LightGBM model validated: {type(model).__name__}")

            else:
                error_msg = f"Unknown model_type: {model_type}"
                logger.debug(error_msg)
                raise ValueError(error_msg)

            # Save to context
            context.scratch["_saved_model"] = model

            logger.info(f"{model_type} model saved successfully")
            return f"{model_type} model saved successfully"

        return save_model


def get_submit_metric_choice_tool(context: BuildContext):
    """
    Factory: Returns metric choice submission tool for MetricSelectorAgent.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def submit_metric_choice(rationale: str, metric_name: str, optimization_direction: str) -> str:
        """
        Submit your evaluation metric choice for this ML task.

        Choose the metric that best reflects what "good performance" means for this task.
        Consider the user's intent, task type, class balance, and data characteristics.

        Args:
            rationale: Explain why this metric is appropriate for this task
            metric_name: Name of the metric (e.g., "accuracy", "f1_score", "rmse", "mae", "roc_auc", "r2_score", "mape")
                - Can also provide a custom metric name if standard metrics don't fit
            optimization_direction: MUST be "higher" or "lower"
                - "higher" for metrics where bigger is better (accuracy, f1_score, r2_score, roc_auc)
                - "lower" for metrics where smaller is better (mse, rmse, mae, mape)

        Quick reference:
            - Balanced classification → accuracy (higher), precision (higher), recall (higher)
            - Imbalanced classification → f1_score (higher), roc_auc (higher)
            - Regression → rmse (lower), mae (lower), r2_score (higher), mape (lower)

        Examples:
            submit_metric_choice(
                rationale="Dataset is imbalanced (98% non-fraud). F1-score balances precision and recall.",
                metric_name="f1_score",
                optimization_direction="higher"
            )

            submit_metric_choice(
                rationale="Predicting house prices. RMSE penalizes large errors appropriately.",
                metric_name="rmse",
                optimization_direction="lower"
            )

        Returns:
            Confirmation message

        Raises:
            ValueError: If validation fails
        """

        # Validate optimization_direction
        if optimization_direction not in ["higher", "lower"]:
            raise ValueError(f"optimization_direction must be 'higher' or 'lower', got: {optimization_direction}")

        # Validate inputs are non-empty
        if not rationale or not rationale.strip():
            raise ValueError("rationale cannot be empty - explain why you chose this metric")
        if not metric_name or not metric_name.strip():
            raise ValueError("metric_name cannot be empty")

        # Create Metric object and save to context
        metric = Metric(name=metric_name.strip(), optimization_direction=optimization_direction)
        context.metric = metric

        # Also save to scratch with rationale (for reporting)
        context.scratch["_metric_selection"] = {
            "name": metric.name,
            "optimization_direction": metric.optimization_direction,
            "rationale": rationale.strip(),
        }

        logger.info(
            f"Metric choice submitted: {metric.name} ({optimization_direction} is better)\nRationale: {rationale}"
        )
        return f"Metric choice submitted: {metric.name} ({optimization_direction} is better)"

    return submit_metric_choice


def get_register_statistical_profile_tool(context: BuildContext):
    """
    Factory: Returns statistical profile submission tool.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def save_statistical_profile(
        total_rows: int,
        total_columns: int,
        numeric_columns: list[str],
        string_columns: list[str],
        other_columns: list[str],
        missing_value_summary: dict[str, float],
        string_stats: dict[str, dict],
        quality_issues: list[str],
        key_insights: list[str],
        numeric_stats: dict[str, dict] | None = None,
    ) -> str:
        """
        Submit your statistical analysis results.

        Provide complete statistics for all columns, adapting analysis to column content.

        Args:
            total_rows: Total number of rows in the dataset
            total_columns: Total number of columns
            numeric_columns: Names of numeric columns (int, float, double)
            string_columns: Names of string columns (categorical values, file paths, or text content)
            other_columns: Names of other column types (datetime, boolean)
            missing_value_summary: For each column, fraction missing as decimal (e.g., {"age": 0.15, "name": 0.02})
            string_stats: For each string column, dict with stats appropriate to content type:
                - For categorical: {"cardinality": int, "mode": str, "mode_frequency": float, "unique_count": int}
                - For file paths: {"sample_paths": [str, ...], "common_extensions": [str, ...], "valid_paths_checked": int}
                - For text: {"avg_length": float, "min_length": int, "max_length": int, "sample_texts": [str, ...]}
            quality_issues: List of data quality problems you identified (e.g., ["Column X has 45% missing", "Invalid image paths detected"])
            key_insights: List of important statistical findings relevant to the data type
            numeric_stats: Optional - for each numeric column, dict with: mean, std, min, max, q25, median, q75, skewness, kurtosis (omit if no numeric columns except target)

        Example (Tabular):
            save_statistical_profile(
                total_rows=8693,
                total_columns=13,
                numeric_columns=["Age", "RoomService", "FoodCourt"],
                string_columns=["HomePlanet", "Cabin", "Destination"],
                other_columns=["CryoSleep", "VIP"],
                missing_value_summary={"Age": 0.18, "Cabin": 0.21},
                string_stats={"HomePlanet": {"cardinality": 3, "mode": "Earth", "mode_frequency": 0.5, "unique_count": 3}},
                quality_issues=["Age has 18% missing values"],
                key_insights=["Most passengers from Earth"],
                numeric_stats={"Age": {"mean": 28.8, "std": 14.5, "min": 0.0, "max": 80.0, "q25": 19.0, "median": 27.0, "q75": 38.0}}
            )

        Example (Images):
            save_statistical_profile(
                total_rows=10000,
                total_columns=2,
                numeric_columns=[],
                string_columns=["image_path"],
                other_columns=["label"],
                missing_value_summary={"image_path": 0.01},
                string_stats={"image_path": {"sample_paths": ["/data/img001.jpg", "/data/img002.png"], "common_extensions": ["jpg", "png"], "valid_paths_checked": 100}},
                quality_issues=["1% missing image paths"],
                key_insights=["Mix of JPG and PNG formats"],
                numeric_stats=None
            )

        Returns:
            Confirmation message
        """

        # Build structured profile
        profile = {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "numeric_columns": numeric_columns,
            "string_columns": string_columns,
            "other_columns": other_columns,
            "missing_value_summary": missing_value_summary,
            "string_stats": string_stats,
            "numeric_stats": numeric_stats,
            "quality_issues": quality_issues,
            "key_insights": key_insights,
        }

        # Save to context
        context.scratch["_statistical_profile"] = profile

        logger.info(f"Statistical profile saved: {total_columns} columns analyzed")
        return f"Statistical profile saved successfully: {total_columns} columns analyzed"

    return save_statistical_profile


def get_register_layout_tool(context: BuildContext):
    """
    Factory: Returns layout detection submission tool.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def register_layout(data_layout: str, primary_input_column: str = None, reason: str = None) -> str:
        """
        Submit your data layout detection results.

        Args:
            data_layout: The detected data layout - MUST be one of: "flat_numeric", "image_path", "text_string", "unsupported"
            primary_input_column: Name of the primary input column (required for "image_path" and "text_string", None for "flat_numeric" and "unsupported")
            reason: Explanation why layout is unsupported (REQUIRED when data_layout="unsupported", otherwise optional)

        Examples:
            # For tabular data
            register_layout(data_layout="flat_numeric", primary_input_column=None)

            # For image data
            register_layout(data_layout="image_path", primary_input_column="image_path")

            # For text data
            register_layout(data_layout="text_string", primary_input_column="review_text")

            # For unsupported data
            register_layout(data_layout="unsupported", reason="Dataset contains multiple image columns (front_image, side_image) which requires multi-input architecture not supported in v1")

        Returns:
            Confirmation message

        Raises:
            ValueError: If validation fails
        """
        from plexe.models import DataLayout

        # Validate data_layout
        valid_layouts = [layout.value for layout in DataLayout]
        if data_layout not in valid_layouts:
            raise ValueError(f"data_layout must be one of {valid_layouts}, got: {data_layout}")

        # Handle UNSUPPORTED layout
        if data_layout == "unsupported":
            if not reason or not reason.strip():
                raise ValueError(
                    "reason parameter is REQUIRED when data_layout='unsupported'. "
                    "Provide a clear explanation of why this data structure is not supported."
                )
            # Ensure primary_input_column is None for unsupported
            if primary_input_column is not None:
                logger.warning(
                    f"primary_input_column provided for unsupported layout (will be ignored): {primary_input_column}"
                )
                primary_input_column = None

        # Validate primary_input_column requirement for supported non-tabular layouts
        elif data_layout in ["image_path", "text_string"]:
            if not primary_input_column or not primary_input_column.strip():
                raise ValueError(
                    f"primary_input_column is required for data_layout='{data_layout}' (must specify which column contains the data)"
                )
        elif data_layout == "flat_numeric":
            if primary_input_column is not None:
                logger.warning(
                    f"primary_input_column provided for flat_numeric layout (will be ignored): {primary_input_column}"
                )
                primary_input_column = None

        # Save layout info to context
        layout_info = {
            "data_layout": data_layout,
            "primary_input_column": primary_input_column,
            "reason": reason if data_layout == "unsupported" else None,
        }

        context.scratch["_layout_info"] = layout_info

        # Also update BuildContext directly
        context.data_layout = DataLayout(data_layout)
        context.primary_input_column = primary_input_column

        if data_layout == "unsupported":
            logger.warning(f"Layout registered as UNSUPPORTED: {reason}")
            return f"Layout registered as UNSUPPORTED: {reason}"
        else:
            logger.info(f"Layout registered: {data_layout}, primary_input_column={primary_input_column}")
            return f"Layout registered successfully: {data_layout}"

    return register_layout


def get_register_eda_report_tool(context: BuildContext):
    """
    Factory: Returns EDA report submission tool.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    # TODO: Consider using @dataclass_json dataclasses for input_description, target_description,
    # and feature_relationships to provide stronger typing and validation

    @tool
    @tool_span
    def save_eda_report(
        task_type: str,
        output_targets: list[str],
        input_description: dict[str, Any],
        target_description: dict[str, Any],
        data_challenges: list[str],
        preprocessing_recommendations: list[str],
        key_insights: list[str],
        recommended_split: dict[str, Any],
        feature_relationships: dict[str, Any] | None = None,
        group_column: str | None = None,
    ) -> str:
        """
        Submit your ML-focused data analysis results.

        Provide ML task definition and actionable insights for modeling.

        Args:
            task_type: Type of ML task (e.g., "binary_classification", "regression", "multiclass_classification")
            output_targets: List of target column name(s) to predict - ["col"] for single-output, ["col1", "col2"] for multi-output, [] for unsupervised
            input_description: Dict describing input structure
            target_description: Dict describing target distribution and characteristics
            data_challenges: List of ML-specific challenges (e.g., ["class imbalance", "path validity issues"])
            preprocessing_recommendations: List of high-level preprocessing guidance (e.g., ["Resize images to consistent dimensions", "Handle missing values"])
            key_insights: Important ML findings relevant to the task
            recommended_split: Split strategy dict with: ratios (dict), temporal_reasoning (str explaining if/why chronological split needed), stratification_reasoning (str explaining if/why stratified split needed)
            feature_relationships: Optional dict with feature-target relationships (only when computable, e.g., correlations for tabular data)
            group_column: Optional group/query ID column for ranking tasks (e.g., "session_id", "query_id", "user_id")

        Example (Tabular):
            save_eda_report(
                task_type="binary_classification",
                output_targets=["Transported"],
                input_description={"type": "tabular", "num_features": 12, "feature_columns": ["Age", "HomePlanet", ...]},
                target_description={"num_classes": 2, "distribution": {"class_0": 0.5, "class_1": 0.5}, "balanced": True},
                data_challenges=["18% missing values in Age", "Cabin has complex structure"],
                preprocessing_recommendations=["Impute missing values", "Encode categorical features"],
                key_insights=["CryoSleep is highly predictive based on correlation"],
                recommended_split={"ratios": {"train": 0.7, "val": 0.15, "test": 0.15}, "temporal_reasoning": "No chronological split needed - this is cross-sectional classification of records, not forecasting future events", "stratification_reasoning": "Stratified split recommended due to class imbalance to maintain balance across splits"},
                feature_relationships={"correlations_with_target": {"CryoSleep": 0.42, "Age": -0.15}}
            )

        Example (Image):
            save_eda_report(
                task_type="multiclass_classification",
                output_targets=["product_category"],
                input_description={"type": "single_column", "column_name": "image_path", "sample_paths": ["/data/img1.jpg", "/data/img2.png"]},
                target_description={"num_classes": 10, "distribution": {"category_0": 0.12, "category_1": 0.09, ...}, "balanced": False},
                data_challenges=["Class imbalance with category_0 at 12%", "Mixed image formats (jpg, png)"],
                preprocessing_recommendations=["Standardize image formats", "Resize to consistent dimensions", "Consider data augmentation for minority classes"],
                key_insights=["10 product categories with imbalanced distribution"],
                recommended_split={"train": 0.7, "val": 0.15, "test": 0.15},
                feature_relationships=None  # Can't correlate images with target
            )

        Example (Text):
            save_eda_report(
                task_type="binary_classification",
                output_targets=["sentiment"],
                input_description={"type": "single_column", "column_name": "review_text", "avg_length": 245, "length_range": [10, 1500]},
                target_description={"num_classes": 2, "distribution": {"positive": 0.65, "negative": 0.35}, "balanced": False},
                data_challenges=["High variance in text length", "Class imbalance (65% positive)"],
                preprocessing_recommendations=["Tokenize text with max_length handling", "Consider addressing class imbalance"],
                key_insights=["Reviews vary widely in length", "Positive sentiment dominant"],
                recommended_split={"train": 0.8, "val": 0.1, "test": 0.1},
                feature_relationships=None  # Can't correlate text with target directly
            )

        Example (Ranking):
            save_eda_report(
                task_type="learning_to_rank",
                output_targets=["relevance"],
                input_description={"type": "tabular", "num_features": 8, "items_per_query": 10, "feature_columns": ["price", "rating", "distance", ...]},
                target_description={"type": "ordinal", "range": [0, 4], "distribution": {"0": 0.47, "1": 0.47, "2": 0.06, "3": 0.0014}},
                data_challenges=["Imbalanced relevance scores", "Query-level dependencies"],
                preprocessing_recommendations=["Handle query grouping", "Consider query-level features", "Normalize scores within queries"],
                key_insights=["Price and rating are strong relevance indicators", "Distance shows non-linear relationship"],
                recommended_split={"train": 0.7, "val": 0.15, "test": 0.15, "stratification": "by query to preserve group structure"},
                feature_relationships={"correlations_with_relevance": {"rating": 0.35, "price": -0.28}},
                group_column="session_id"
            )

        Returns:
            Confirmation message
        """

        # Build structured report
        report = {
            "task_type": task_type,
            "output_targets": output_targets,
            "input_description": input_description,
            "target_description": target_description,
            "data_challenges": data_challenges,
            "preprocessing_recommendations": preprocessing_recommendations,
            "key_insights": key_insights,
            "recommended_split": recommended_split,
            "feature_relationships": feature_relationships,
            "group_column": group_column,
        }

        # Save to context
        context.scratch["_eda_report"] = report

        # Also save group_column directly to context for ranking tasks
        if group_column:
            context.group_column = group_column
            logger.info(f"Group column for ranking: {group_column}")

        logger.info(f"EDA report saved: {task_type} task with output_targets {output_targets}")
        return f"EDA report saved successfully: {task_type} task with targets {output_targets}"

    return save_eda_report


def get_save_split_uris_tool(context: BuildContext):
    """
    Factory: Returns split URI submission tool.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def save_split_uris(train_uri: str, val_uri: str, test_uri: str = None) -> str:
        """
        Submit the URIs of your train/val/test dataset splits.

        Call this after writing your split parquet files.

        Args:
            train_uri: Full path to train.parquet
            val_uri: Full path to val.parquet
            test_uri: Full path to test.parquet (optional - omit for 2-way splits)

        Example (3-way):
            save_split_uris(train_uri="/path/train.parquet", val_uri="/path/val.parquet", test_uri="/path/test.parquet")

        Example (2-way):
            save_split_uris(train_uri="/path/train.parquet", val_uri="/path/val.parquet")

        Returns:
            Confirmation message
        """

        # Save URIs to context scratch
        context.scratch["_train_uri"] = train_uri
        context.scratch["_val_uri"] = val_uri
        context.scratch["_test_uri"] = test_uri  # Can be None

        if test_uri:
            logger.info(f"Split URIs saved: train={train_uri}, val={val_uri}, test={test_uri}")
        else:
            logger.info(f"Split URIs saved: train={train_uri}, val={val_uri} (no test set)")

        return "Split URIs saved successfully"

    return save_split_uris


def get_save_sample_uris_tool(context: BuildContext):
    """
    Factory: Returns sample URIs submission tool.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def save_sample_uris(train_sample_uri: str, val_sample_uri: str) -> str:
        """
        Submit the URIs of your train and validation dataset samples.

        Call this after writing both sample parquet files.

        Args:
            train_sample_uri: Full path to train sample parquet file
            val_sample_uri: Full path to validation sample parquet file

        Example:
            save_sample_uris(
                train_sample_uri="workdir/.build/data/samples/train_sample.parquet",
                val_sample_uri="workdir/.build/data/samples/val_sample.parquet"
            )

        Returns:
            Confirmation message
        """

        # Save URIs to context scratch
        context.scratch["_train_sample_uri"] = train_sample_uri
        context.scratch["_val_sample_uri"] = val_sample_uri

        logger.info(f"Sample URIs saved: train={train_sample_uri}, val={val_sample_uri}")
        return "Sample URIs saved successfully"

    return save_sample_uris


def get_save_metric_implementation_fn(context: BuildContext):
    """
    Factory: Returns metric implementation submission function.

    Args:
        context: Build context for storing result

    Returns:
        Configured function
    """

    @tool_span
    @agentinspectable
    def save_metric_implementation(compute_metric_function: Any) -> str:
        """
        Submit your metric computation function.

        The function must have signature: compute_metric(y_true, y_pred) -> float

        Args:
            compute_metric_function: Callable function object

        Returns:
            Confirmation message

        Raises:
            ValueError: If function validation fails
        """
        from plexe.validation.validators import validate_metric_function_object

        # Validate function
        is_valid, error_msg = validate_metric_function_object(compute_metric_function)

        if not is_valid:
            logger.error(f"Metric function validation failed: {error_msg}")
            raise ValueError(f"Metric function validation failed: {error_msg}")

        # Save function to context
        context.compute_metric = compute_metric_function

        logger.info("Metric implementation function saved and validated")
        return "Metric implementation function saved successfully"

    return save_metric_implementation


def get_validate_baseline_predictor_tool(context: BuildContext, val_sample_df):
    """
    Factory: Returns baseline predictor validation tool.

    Args:
        context: Build context
        val_sample_df: Validation sample for testing

    Returns:
        Configured tool
    """

    @tool_span
    @agentinspectable
    def validate_baseline_predictor(predictor: Any, name: str, description: str) -> str:
        """
        Validate your baseline predictor and compute its performance metric.

        Tests that predictor has .predict(), works on validation data, AND
        computes the metric successfully. This ensures any evaluation errors
        (like deprecated sklearn API) are caught while the agent is still
        running so it can fix them.

        Args:
            predictor: HeuristicBaselinePredictor instance
            name: Name for this baseline
            description: Description of approach

        Returns:
            Success message with performance metric

        Raises:
            ValueError: If validation or metric computation fails
        """
        import numpy as np
        from plexe.helpers import compute_metric

        # Check class name matches template
        if type(predictor).__name__ != "HeuristicBaselinePredictor":
            error_msg = f"Predictor class must be named 'HeuristicBaselinePredictor', got '{type(predictor).__name__}' (use the template)"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Check interface
        if not hasattr(predictor, "predict") or not callable(predictor.predict):
            error_msg = "Predictor must have callable .predict() method"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Test on small validation sample
        try:
            X_test = val_sample_df.drop(columns=context.output_targets, errors="ignore").head(10)
            predictions = predictor.predict(X_test)

            if not isinstance(predictions, list | tuple | np.ndarray | pd.Series):
                error_msg = f"predict() must return array-like, got {type(predictions)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if len(predictions) != len(X_test):
                error_msg = f"predict() returned {len(predictions)} predictions for {len(X_test)} samples"
                logger.error(error_msg)
                raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Predictor test failed: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # CRITICAL: Compute metric HERE so agent sees any evaluation errors
        try:
            X_val = val_sample_df.drop(columns=context.output_targets, errors="ignore")
            y_val = val_sample_df[context.output_targets[0]]
            y_pred = predictor.predict(X_val)

            # This is where squared= errors would happen - agent can now see them!
            performance = compute_metric(y_true=y_val.values, y_pred=y_pred, metric_name=context.metric.name)

            logger.info(f"Baseline performance: {context.metric.name}={performance:.4f}")

        except Exception as e:
            # Agent will see this error and can fix it!
            error_msg = f"Metric computation failed: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Save to context (for final baseline object creation)
        context.baseline_predictor = predictor
        context.baseline_performance = float(performance)
        context.scratch["_baseline_name"] = name.lower().strip().replace(" ", "_").replace("-", "_")
        context.scratch["_baseline_description"] = description

        logger.info(f"Baseline predictor '{name}' validated successfully")
        return f"Baseline '{name}' validated: {context.metric.name}={performance:.4f}"

    return validate_baseline_predictor


def get_save_baseline_code_tool(context: BuildContext, val_sample_df):
    """
    Factory: Returns baseline code saving tool.

    Args:
        context: Build context
        val_sample_df: Validation sample for testing

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def save_baseline_code(code: str) -> str:
        """
        Save the baseline predictor code after validation. The code should contain the code
        for the entire module, including imports and the HeuristicBaselinePredictor class definition, as
        per the provided template.

        Args:
            code: Python code defining module containing HeuristicBaselinePredictor class

        Returns:
            Confirmation message

        Raises:
            ValueError: If code invalid
        """
        # Step 1: Execute code into module (allowed in tool, not in agent sandbox)
        predictor_module = types.ModuleType("baseline_predictor")
        try:
            exec(code, predictor_module.__dict__)
        except Exception as e:
            error_msg = f"Code execution failed: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Step 2: Get predictor class (known name from template)
        try:
            predictor_class = getattr(predictor_module, "HeuristicBaselinePredictor")
        except AttributeError:
            error_msg = "Code must define 'HeuristicBaselinePredictor' class (use template)"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Step 3: Instantiate and revalidate
        try:
            predictor = predictor_class()
        except Exception as e:
            error_msg = f"Failed to instantiate HeuristicBaselinePredictor(): {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        validate_tool = get_validate_baseline_predictor_tool(context, val_sample_df)
        name = context.scratch.get("_baseline_name", "baseline")
        description = context.scratch.get("_baseline_description", "")
        validate_tool(predictor, name, description)  # Raises if invalid

        # Step 4: Save code to file
        baseline_dir = context.work_dir / DirNames.BUILD_DIR / "search" / "baselines"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        code_path = baseline_dir / f"{name}.py"
        code_path.write_text(code)
        context.scratch["_baseline_code_path"] = str(code_path)

        logger.info(f"Baseline code validated and saved to {code_path}")
        return f"Baseline code validated and saved to {code_path}"

    return save_baseline_code


def get_evaluate_baseline_performance_tool(context: BuildContext, val_sample_df):
    """
    Factory: Returns baseline performance evaluation tool.

    Args:
        context: Build context with baseline predictor and metric
        val_sample_df: Validation sample DataFrame (pandas)

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def evaluate_baseline_performance() -> str:
        """
        Evaluate your baseline predictor on the validation sample.

        Runs prediction on val_sample_df and computes the metric.
        You must have already called register_baseline_predictor() before calling this.

        Returns:
            String with performance metric value
        """
        from plexe.helpers import compute_metric

        # Check prerequisites
        if context.baseline_predictor is None:
            raise ValueError("No baseline predictor registered. Call register_baseline_predictor() first.")

        if context.metric is None:
            raise ValueError("No metric selected. Cannot evaluate baseline.")

        # Separate features from target (and group_column for ranking)
        target_cols = context.output_targets
        exclude_cols = list(target_cols)
        if context.group_column and context.group_column in val_sample_df.columns:
            exclude_cols.append(context.group_column)

        feature_cols = [col for col in val_sample_df.columns if col not in exclude_cols]

        X_val = val_sample_df[feature_cols]
        y_val = val_sample_df[target_cols[0]]  # Single target assumed

        # Extract group_ids for ranking metrics
        group_ids = (
            val_sample_df[context.group_column].values
            if context.group_column and context.group_column in val_sample_df.columns
            else None
        )

        # Make predictions (standard array interface)
        y_pred = context.baseline_predictor.predict(X_val)

        # Compute metric (pass group_ids for ranking metrics)
        performance = compute_metric(
            y_true=y_val.values, y_pred=y_pred, metric_name=context.metric.name, group_ids=group_ids
        )

        # Save performance
        context.baseline_performance = float(performance)

        logger.info(f"Baseline performance: {context.metric.name}={performance:.4f}")
        return f"Baseline performance: {context.metric.name}={performance:.4f}"

    return evaluate_baseline_performance


# ============================================
# Hypothesis-Driven Search Tools
# ============================================


def get_save_hypothesis_tool(context: BuildContext, expand_node_id: int):
    """
    Factory: Returns hypothesis submission tool for HypothesiserAgent.

    Args:
        context: Build context for storing result
        expand_node_id: Which solution node to expand (injected from policy decision)

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def save_hypothesis(
        focus: str,
        vary: str,
        num_variants: int,
        rationale: str,
        keep_from_parent: list[str],
        expected_impact: str,
    ) -> str:
        """
        Submit your hypothesis for next exploration.

        Args:
            focus: What to vary - MUST be one of: "features", "model", or "both"
            vary: Specific aspect to change (e.g., "n_estimators", "scaling_strategy")
            num_variants: How many variations to try (1-3 recommended)
            rationale: Why this direction is promising
            keep_from_parent: What to keep unchanged - MUST be a list containing ONLY "features" and/or "model", or empty list []
                Examples: [], ["features"], ["model"], ["features", "model"]
            expected_impact: Predicted effect (e.g., "±3-5% performance swing expected")

        Returns:
            Confirmation message

        Raises:
            ValueError: If validation fails
        """

        # Validate focus
        valid_focus = ["features", "model", "both"]
        if focus not in valid_focus:
            raise ValueError(f"focus must be one of {valid_focus}, got: {focus}")

        # Validate keep_from_parent
        valid_keep_options = {"features", "model"}
        invalid_options = set(keep_from_parent) - valid_keep_options
        if invalid_options:
            raise ValueError(
                f"keep_from_parent can only contain 'features' and/or 'model', got invalid: {invalid_options}. "
                f"Valid examples: [], ['features'], ['model'], ['features', 'model']"
            )

        # Validate num_variants
        if num_variants < 1 or num_variants > 10:
            raise ValueError(f"num_variants should be between 1-10, got: {num_variants}")

        hypothesis = Hypothesis(
            expand_solution_id=expand_node_id,  # Injected from closure
            focus=focus,
            vary=vary,
            num_variants=num_variants,
            rationale=rationale,
            keep_from_parent=keep_from_parent,
            expected_impact=expected_impact,
        )

        context.scratch["_hypothesis"] = hypothesis

        logger.info(f"Hypothesis saved: expand solution {expand_node_id}, vary {vary}, {num_variants} variants")
        return f"Hypothesis saved: {num_variants} variants of {vary}"

    return save_hypothesis


def get_save_plan_tool(context: BuildContext, hypothesis: "Hypothesis", allowed_model_types: list[str] | None):
    """
    Factory: Returns plan submission tool for PlannerAgent.

    Args:
        context: Build context for storing result
        hypothesis: Hypothesis being implemented (injected for parent_node, rationale, expected_impact)
        allowed_model_types: Optional list of allowed model types (e.g., ["xgboost"] to restrict)

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def save_plan(
        variant_id: str,
        model_type: str,
        feature_strategy: str,
        feature_changes: dict,
        feature_rationale: str,
        model_directive: str,
        model_change_summary: str,
        model_rationale: str,
    ) -> str:
        """
        Submit a unified plan specification for a single variant.

        Args:
            variant_id: Variant identifier ("A", "B", "C", etc.)
            model_type: Model architecture type - MUST be one of: "xgboost", "catboost", or "keras"
            feature_strategy: MUST be one of: "reuse_parent", "new", or "modify_parent"
            feature_changes: Dict of specific feature changes (empty dict {} if reusing or creating new)
            feature_rationale: Why this feature approach
            model_directive: Natural language directive for model changes (e.g., "Increase n_estimators to around 250" for XGBoost or "Add dropout layer" for Keras)
            model_change_summary: Brief summary of model changes (e.g., "n_estimators: 100→250" or "architecture: 3 layers → 5 layers")
            model_rationale: Why this model change

        Returns:
            Confirmation message

        Raises:
            ValueError: If validation fails
        """
        from plexe.models import FeaturePlan, ModelPlan
        from plexe.config import ModelType

        # Validate feature_strategy
        valid_strategies = ["reuse_parent", "new", "modify_parent"]
        if feature_strategy not in valid_strategies:
            raise ValueError(f"feature_strategy must be one of {valid_strategies}, got: {feature_strategy}")

        # Validate variant_id is non-empty
        if not variant_id or not variant_id.strip():
            raise ValueError("variant_id cannot be empty")

        # Validate model_type
        valid_model_types = [
            ModelType.XGBOOST,
            ModelType.CATBOOST,
            ModelType.LIGHTGBM,
            ModelType.KERAS,
            ModelType.PYTORCH,
        ]
        if model_type not in valid_model_types:
            raise ValueError(f"model_type must be one of {valid_model_types}, got: {model_type}")

        # Check against allowed_model_types constraint if provided
        if allowed_model_types and model_type not in allowed_model_types:
            raise ValueError(f"model_type '{model_type}' is not allowed. Allowed types: {allowed_model_types}")

        plan = UnifiedPlan(
            variant_id=variant_id,
            parent_solution_id=hypothesis.expand_solution_id,  # Injected from hypothesis
            features=FeaturePlan(
                strategy=feature_strategy,
                parent_solution_id=hypothesis.expand_solution_id,  # Same as parent_solution_id
                changes=feature_changes,
                rationale=feature_rationale,
            ),
            model=ModelPlan(
                model_type=model_type,  # Provided by agent (per-variant decision)
                directive=model_directive,
                change_summary=model_change_summary,
                rationale=model_rationale,
            ),
            hypothesis_rationale=hypothesis.rationale,  # Injected from hypothesis
            expected_outcome=hypothesis.expected_impact,  # Injected from hypothesis
        )

        # Save to context (will be retrieved for execution)
        if "_plans" not in context.scratch:
            context.scratch["_plans"] = []
        context.scratch["_plans"].append(plan)

        logger.info(f"Plan {variant_id} saved: {model_type} - {model_change_summary}")
        return f"Plan {variant_id} saved successfully: {model_type}"

    return save_plan


def get_save_insight_tool(insight_store: InsightStore):
    """
    Factory: Returns insight submission tool for InsightExtractorAgent.

    Args:
        insight_store: InsightStore instance

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def save_insight(
        change: str,
        effect: str,
        applies_when: str,
        confidence: str,
        supporting_evidence: list[int],
    ) -> str:
        """
        Submit an insight extracted from experiments.

        Args:
            change: What was varied (e.g., "n_estimators: 100→250")
            effect: Observed outcome (e.g., "+5.8% improvement, peak at 250")
            applies_when: When/where this insight applies (e.g., "for datasets with ~8k rows, ~13 features")
            confidence: MUST be one of: "high", "medium", or "low"
            supporting_evidence: List of solution iteration IDs that support this insight

        Returns:
            Confirmation message

        Raises:
            ValueError: If confidence is not valid
        """
        # Validate confidence
        valid_confidence = ["high", "medium", "low"]
        if confidence not in valid_confidence:
            raise ValueError(f"confidence must be one of {valid_confidence}, got: {confidence}")

        insight = insight_store.add(
            change=change,
            effect=effect,
            context=applies_when,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
        )

        logger.info(f"Insight #{insight.id} saved: {change} → {effect}")
        return f"Insight #{insight.id} saved successfully"

    return save_insight


# ============================================
# Model Evaluation Tools
# ============================================


def get_register_core_metrics_tool(context: BuildContext):
    """
    Factory: Returns core metrics submission tool for ModelEvaluatorAgent.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def register_core_metrics_report(
        task_type: str,
        primary_metric_name: str,
        primary_metric_value: float,
        primary_metric_ci_lower: float | None,
        primary_metric_ci_upper: float | None,
        all_metrics: dict[str, float],
        statistical_notes: str,
        metric_confidence_intervals: dict[str, tuple[float, float]] | None = None,
        visualizations: dict[str, str] | None = None,
    ) -> str:
        """
        Submit your core metrics evaluation report.

        Args:
            task_type: Detected task type (e.g., "binary_classification", "regression")
            primary_metric_name: Name of primary optimization metric
            primary_metric_value: Value of primary metric on test set
            primary_metric_ci_lower: 95% CI lower bound (None if not computed)
            primary_metric_ci_upper: 95% CI upper bound (None if not computed)
            all_metrics: Dict of all computed metrics (e.g., {"accuracy": 0.85, "f1_score": 0.82})
            statistical_notes: Your interpretation of the results
            metric_confidence_intervals: Optional CIs for other metrics {metric: (lower, upper)}
            visualizations: Optional visualizations {plot_name: base64_png}

        Returns:
            Confirmation message
        """
        from plexe.models import CoreMetricsReport

        report = CoreMetricsReport(
            task_type=task_type,
            primary_metric_name=primary_metric_name,
            primary_metric_value=primary_metric_value,
            primary_metric_ci_lower=primary_metric_ci_lower,
            primary_metric_ci_upper=primary_metric_ci_upper,
            all_metrics=all_metrics,
            metric_confidence_intervals=metric_confidence_intervals,
            statistical_notes=statistical_notes,
            visualizations=visualizations,
        )

        context.scratch["_core_metrics_report"] = report
        logger.info(f"Core metrics report saved: {primary_metric_name}={primary_metric_value:.4f}")
        return f"Core metrics report saved: {primary_metric_name}={primary_metric_value:.4f}"

    return register_core_metrics_report


def get_register_diagnostic_report_tool(context: BuildContext):
    """
    Factory: Returns diagnostic report submission tool for ModelEvaluatorAgent.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def register_diagnostic_report(
        worst_predictions: list[dict],
        error_patterns: list[str],
        key_insights: list[str],
        error_distribution_summary: str,
        subgroup_analysis: dict[str, dict] | None = None,
    ) -> str:
        """
        Submit your diagnostic report (error analysis).

        Args:
            worst_predictions: Top 20-30 worst predictions with context
                Format: [{"index": 42, "true_value": 1.5, "predicted_value": 2.8, "error": 1.3, "features": {...}}, ...]
            error_patterns: Identified failure patterns (e.g., ["Fails on high Age values", "Poor for rare HomePlanet"])
            key_insights: WHY the model is failing (e.g., ["Model struggles with outliers"])
            error_distribution_summary: Summary of error characteristics
            subgroup_analysis: Optional subgroup analysis {subgroup_name: {metrics}}

        Returns:
            Confirmation message
        """
        from plexe.models import DiagnosticReport

        report = DiagnosticReport(
            worst_predictions=worst_predictions,
            error_patterns=error_patterns,
            subgroup_analysis=subgroup_analysis,
            key_insights=key_insights,
            error_distribution_summary=error_distribution_summary,
        )

        context.scratch["_diagnostic_report"] = report
        logger.info(f"Diagnostic report saved: {len(worst_predictions)} worst predictions analyzed")
        return "Diagnostic report saved successfully"

    return register_diagnostic_report


def get_register_robustness_report_tool(context: BuildContext):
    """
    Factory: Returns robustness report submission tool for ModelEvaluatorAgent.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def register_robustness_report(
        perturbation_tests: dict[str, dict],
        robustness_grade: str,
        concerns: list[str],
        recommendations: list[str],
        consistency_score: float | None = None,
    ) -> str:
        """
        Submit your robustness assessment report.

        Args:
            perturbation_tests: Test results by type (e.g., {"noise": {...}, "missing_values": {...}})
            robustness_grade: Overall grade - MUST be one of: "A", "B", "C", "D", "F"
            concerns: Identified risks (e.g., ["Sensitive to missing values"])
            recommendations: Mitigation suggestions (e.g., ["Add input validation"])
            consistency_score: Optional 0-1 score for consistency (same input → same output)

        Returns:
            Confirmation message

        Raises:
            ValueError: If grade is invalid
        """
        from plexe.models import RobustnessReport

        valid_grades = ["A", "B", "C", "D", "F"]
        if robustness_grade not in valid_grades:
            raise ValueError(f"robustness_grade must be one of {valid_grades}, got: {robustness_grade}")

        report = RobustnessReport(
            perturbation_tests=perturbation_tests,
            consistency_score=consistency_score,
            robustness_grade=robustness_grade,
            concerns=concerns,
            recommendations=recommendations,
        )

        context.scratch["_robustness_report"] = report
        logger.info(f"Robustness report saved: grade {robustness_grade}")
        return f"Robustness report saved: grade {robustness_grade}"

    return register_robustness_report


def get_register_explainability_report_tool(context: BuildContext):
    """
    Factory: Returns explainability report submission tool for ModelEvaluatorAgent.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def register_explainability_report(
        feature_importance: dict[str, float],
        method_used: str,
        top_features: list[str],
        interpretation: str,
        confidence_intervals: dict[str, tuple[float, float]] | None = None,
    ) -> str:
        """
        Submit your explainability analysis report.

        Args:
            feature_importance: Feature -> importance scores (e.g., {"Age": 0.25, "HomePlanet": 0.18})
            method_used: Method for computing importance (e.g., "permutation", "SHAP", "built-in")
            top_features: Most important features in order (e.g., ["Age", "HomePlanet", "CryoSleep"])
            interpretation: What these features mean and why they're important
            confidence_intervals: Optional CIs for feature importance {feature: (lower, upper)}

        Returns:
            Confirmation message
        """
        from plexe.models import ExplainabilityReport

        report = ExplainabilityReport(
            feature_importance=feature_importance,
            method_used=method_used,
            top_features=top_features,
            confidence_intervals=confidence_intervals,
            interpretation=interpretation,
        )

        context.scratch["_explainability_report"] = report
        logger.info(f"Explainability report saved: {len(feature_importance)} features analyzed")
        return "Explainability report saved successfully"

    return register_explainability_report


def get_register_baseline_comparison_tool(context: BuildContext):
    """
    Factory: Returns baseline comparison submission tool for ModelEvaluatorAgent.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def register_baseline_comparison_report(
        baseline_name: str,
        baseline_type: str,
        baseline_description: str,
        baseline_performance: dict[str, float],
        model_performance: dict[str, float],
        performance_delta: dict[str, float],
        interpretation: str,
        performance_delta_pct: dict[str, float] | None = None,
    ) -> str:
        """
        Submit your baseline comparison report.

        Args:
            baseline_name: Name of baseline (e.g., "heuristic_most_frequent")
            baseline_type: Type of baseline (e.g., "heuristic", "simple_model")
            baseline_description: Description of baseline approach
            baseline_performance: Baseline metrics (e.g., {"accuracy": 0.72})
            model_performance: Model metrics (e.g., {"accuracy": 0.85})
            performance_delta: Absolute differences (e.g., {"accuracy": 0.13})
            interpretation: Does improvement justify complexity?
            performance_delta_pct: Optional percentage changes (e.g., {"accuracy": 18.1})

        Returns:
            Confirmation message
        """
        from plexe.models import BaselineComparisonReport

        report = BaselineComparisonReport(
            baseline_name=baseline_name,
            baseline_type=baseline_type,
            baseline_description=baseline_description,
            baseline_performance=baseline_performance,
            model_performance=model_performance,
            performance_delta=performance_delta,
            performance_delta_pct=performance_delta_pct,
            interpretation=interpretation,
        )

        context.scratch["_baseline_comparison_report"] = report
        logger.info(f"Baseline comparison report saved: {baseline_name} vs model")
        return "Baseline comparison report saved successfully"

    return register_baseline_comparison_report


def get_register_final_evaluation_tool(context: BuildContext):
    """
    Factory: Returns final evaluation submission tool for ModelEvaluatorAgent.

    Args:
        context: Build context for storing result

    Returns:
        Configured tool
    """

    @tool
    @tool_span
    def register_final_evaluation_report(
        verdict: str,
        summary: str,
        deployment_ready: bool,
        key_concerns: list[str],
        recommendations: list[dict],
    ) -> str:
        """
        Submit your final evaluation report with verdict.

        Args:
            verdict: MUST be one of: "PASS", "CONDITIONAL_PASS", "FAIL"
            summary: 2-3 sentence executive summary
            deployment_ready: True if ready for production, False otherwise
            key_concerns: Critical issues (e.g., ["Sensitive to outliers"])
            recommendations: Prioritized actions
                Format: [{"priority": "HIGH", "action": "...", "rationale": "..."}, ...]
                Priority MUST be one of: "HIGH", "MEDIUM", "LOW"

        Returns:
            Confirmation message

        Raises:
            ValueError: If verdict or priorities are invalid
        """
        from plexe.models import EvaluationReport

        valid_verdicts = ["PASS", "CONDITIONAL_PASS", "FAIL"]
        if verdict not in valid_verdicts:
            raise ValueError(f"verdict must be one of {valid_verdicts}, got: {verdict}")

        # Validate recommendation priorities
        valid_priorities = ["HIGH", "MEDIUM", "LOW"]
        for rec in recommendations:
            if "priority" not in rec:
                raise ValueError("Each recommendation must have 'priority' field")
            if rec["priority"] not in valid_priorities:
                raise ValueError(f"Recommendation priority must be one of {valid_priorities}, got: {rec['priority']}")

        # Get component reports from scratch (required fields)
        core_metrics = context.scratch.get("_core_metrics_report")
        diagnostics = context.scratch.get("_diagnostic_report")
        robustness = context.scratch.get("_robustness_report")
        explainability = context.scratch.get("_explainability_report")  # Optional
        baseline_comparison = context.scratch.get("_baseline_comparison_report")

        # Validate required reports are present
        if not core_metrics:
            raise ValueError("core_metrics_report is required but not found in context")
        if not diagnostics:
            raise ValueError("diagnostic_report is required but not found in context")
        if not robustness:
            raise ValueError("robustness_report is required but not found in context")
        if not baseline_comparison:
            raise ValueError("baseline_comparison_report is required but not found in context")

        report = EvaluationReport(
            verdict=verdict,
            summary=summary,
            deployment_ready=deployment_ready,
            key_concerns=key_concerns,
            core_metrics=core_metrics,
            diagnostics=diagnostics,
            robustness=robustness,
            explainability=explainability,
            baseline_comparison=baseline_comparison,
            recommendations=recommendations,
        )

        context.scratch["_evaluation_report"] = report
        logger.info(f"Final evaluation report saved: verdict={verdict}, deployment_ready={deployment_ready}")
        return f"Final evaluation report saved: verdict={verdict}"

    return register_final_evaluation_report
