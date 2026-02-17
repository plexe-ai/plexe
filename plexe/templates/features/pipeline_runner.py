"""
Apply fitted sklearn Pipeline to full dataset via Spark.

Uses mapInPandas to distribute transformation across Spark partitions.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from textwrap import shorten
from typing import TYPE_CHECKING

import cloudpickle
import pandas as pd
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def transform_dataset_via_spark(
    spark: SparkSession,
    dataset_uri: str,
    fitted_pipeline: Pipeline,
    output_uri: str,
    target_columns: list[str],
    pipeline_code: str = "",
    group_column: str | None = None,
) -> str:
    """
    Apply fitted sklearn Pipeline to full dataset using Spark UDFs.

    This distributes the transformation across Spark workers, allowing
    processing of datasets larger than single-machine memory.

    IMPORTANT: Pipeline transforms features only. This function preserves
    target column(s) and group column in the output for training/evaluation.

    Args:
        spark: SparkSession
        dataset_uri: URI to raw dataset (with features + target + group)
        fitted_pipeline: Already-fitted sklearn Pipeline
        output_uri: Where to save transformed dataset
        target_columns: List of target column names to preserve
        pipeline_code: Python code string that defines custom transformers (optional)
        group_column: Optional group column for ranking (query_id, session_id) to preserve

    Returns:
        URI to transformed dataset
    """

    logger.info(f"Transforming dataset at {shorten(dataset_uri, 30)}...")

    # Step 1: Validate Pipeline is Fitted
    try:
        # Check if pipeline has been fitted by looking for fitted attributes
        if hasattr(fitted_pipeline, "steps") and len(fitted_pipeline.steps) > 0:
            first_step = fitted_pipeline.steps[0][1]
            # Most sklearn transformers have these when fitted
            if not any(hasattr(first_step, attr) for attr in ["n_features_in_", "feature_names_in_", "categories_"]):
                logger.warning("Pipeline may not be fitted - proceeding anyway")
    except Exception:
        pass  # Best effort check

    # Step 2: Load Dataset
    df = spark.read.parquet(dataset_uri)
    all_columns = df.columns

    # Identify columns to exclude from transformation
    columns_to_exclude = list(target_columns)
    if group_column and group_column in all_columns:
        columns_to_exclude.append(group_column)

    feature_columns = [col for col in all_columns if col not in columns_to_exclude]

    logger.info(f"Loaded dataset from {shorten(dataset_uri, 30)}")
    logger.info(f"Features: {len(feature_columns)}, Target: {target_columns}")
    if group_column:
        logger.info(f"Group column (preserved): {group_column}")

    # Step 3: Infer Output Schema from Sample
    sample_pdf = df.limit(10).toPandas()

    # Separate features from targets and group
    sample_features = sample_pdf[feature_columns]
    sample_targets = sample_pdf[target_columns]
    sample_group = sample_pdf[[group_column]] if group_column and group_column in sample_pdf.columns else None

    # Transform features only
    sample_transformed = fitted_pipeline.transform(sample_features)
    logger.info(f"Transformed {len(sample_transformed)} samples - pipeline works on pd.DataFrame")

    # Get feature names: if pipeline outputs DataFrame, use its columns; else use generic
    if isinstance(sample_transformed, pd.DataFrame):
        # Pipeline returned DataFrame with column names - use them!
        transformed_feature_cols = sample_transformed.columns.tolist()
        logger.info(f"Using meaningful feature names from pipeline output: {len(transformed_feature_cols)} features")
        sample_transformed_features = sample_transformed
    else:
        # Pipeline returned numpy array - need to assign column names
        try:
            # Try sklearn's get_feature_names_out() (sklearn 1.0+)
            if hasattr(fitted_pipeline, "get_feature_names_out"):
                transformed_feature_cols = fitted_pipeline.get_feature_names_out(
                    input_features=feature_columns
                ).tolist()
                logger.info("Using feature names from pipeline.get_feature_names_out()")
            else:
                raise AttributeError("No get_feature_names_out method")
        except Exception as e:
            # Fallback to generic names
            num_output_features = sample_transformed.shape[1]
            transformed_feature_cols = [f"feature_{i}" for i in range(num_output_features)]
            logger.warning(f"Using generic feature names ({e})")

        sample_transformed_features = pd.DataFrame(sample_transformed, columns=transformed_feature_cols)

    # Create full output (transformed features + group + target)
    output_parts = [sample_transformed_features]
    if sample_group is not None:
        output_parts.append(sample_group.reset_index(drop=True))
    output_parts.append(sample_targets.reset_index(drop=True))

    sample_output = pd.concat(output_parts, axis=1)

    # Let Spark infer schema from pandas (handles all dtypes correctly)
    output_schema = spark.createDataFrame(sample_output).schema

    group_info = f" + {len([group_column])} group" if group_column else ""
    logger.info(
        f"Output schema: {len(output_schema.fields)} columns ({len(transformed_feature_cols)} features{group_info} + {len(target_columns)} target)"
    )

    # Step 4: Bundle Pipeline Code + Serialized Pipeline
    # Custom transformers need code definitions available on workers
    # We bundle both the code string and pickled pipeline together
    logger.info("Bundling pipeline code with serialized pipeline for Spark distribution...")

    # Serialize pipeline with cloudpickle (handles custom classes and closures)
    pipeline_bytes = cloudpickle.dumps(fitted_pipeline)

    # Create bundle package
    package = {"code": pipeline_code, "pipeline_bytes": pipeline_bytes}
    package_bytes = cloudpickle.dumps(package)

    logger.info(f"Package serialized: {len(package_bytes)} bytes (code: {len(pipeline_code)} chars)")

    # Step 5: Define Transformation UDF
    def apply_pipeline_udf(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        """Apply fitted pipeline to features, preserve target."""
        # Unpack bundle in worker
        import cloudpickle

        package = cloudpickle.loads(package_bytes)

        # Execute custom code if present (defines custom transformer classes)
        if package["code"]:
            exec(package["code"], globals())

        # Deserialize pipeline (custom classes now available)
        pipeline = cloudpickle.loads(package["pipeline_bytes"])

        for pdf in iterator:
            # Separate features, targets, and group
            features = pdf[feature_columns]
            targets = pdf[target_columns]
            group = pdf[[group_column]] if group_column and group_column in pdf.columns else None

            # Transform features only
            transformed_features = pipeline.transform(features)

            # Handle both DataFrame (with column names) and numpy array output
            if isinstance(transformed_features, pd.DataFrame):
                # Pipeline returned DataFrame - preserve its column names
                transformed_df = transformed_features
            else:
                # Pipeline returned numpy array - use pre-defined column names
                transformed_df = pd.DataFrame(transformed_features, columns=transformed_feature_cols)

            # Re-attach group and target columns
            result_parts = [transformed_df]
            if group is not None:
                result_parts.append(group.reset_index(drop=True))
            result_parts.append(targets.reset_index(drop=True))

            result = pd.concat(result_parts, axis=1)

            yield result

    # Step 6: Apply Transformation via Spark
    logger.info("Applying transformation via Spark UDF (may take 10-30 minutes for large datasets)...")

    transformed_df = df.mapInPandas(apply_pipeline_udf, schema=output_schema)

    # Step 7: Write Transformed Dataset
    transformed_df.write.parquet(output_uri, mode="overwrite")

    logger.info(f"Transformed dataset saved to {shorten(output_uri, 30)}")

    return output_uri
