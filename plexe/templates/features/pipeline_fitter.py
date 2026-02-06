"""
Fit sklearn Pipeline on dataset.

Deterministic logic - no code generation.
"""

import logging
from textwrap import shorten

import pandas as pd
from sklearn.pipeline import Pipeline

from plexe.utils.s3 import download_s3_uri

logger = logging.getLogger(__name__)


def fit_pipeline(
    dataset_uri: str,
    pipeline: Pipeline,
    target_columns: list[str],
    group_column: str | None = None,
) -> Pipeline:
    """
    Fit sklearn Pipeline on the provided dataset.

    Workflow always passes sample URIs to this function (train_sample_uri),
    ensuring memory-efficient pipeline fitting. Sklearn transformers learn
    stable parameters from representative samples.

    CRITICAL: Pipeline is fitted on FEATURES ONLY (target + group columns removed).
    This ensures the pipeline never expects target/group columns during transform.

    Args:
        dataset_uri: URI to dataset sample (parquet) - expected to be ~30k rows
        pipeline: Unfitted sklearn Pipeline
        target_columns: List of target column names to exclude from fitting
        group_column: Optional group column for ranking (query_id, session_id) to exclude

    Returns:
        Fitted Pipeline (ready to transform)
    """

    logger.info(f"Fitting pipeline on dataset: {shorten(dataset_uri, 30)}")

    # Download from S3 if needed
    if dataset_uri.startswith("s3://"):
        dataset_uri = download_s3_uri(dataset_uri)

    # Load with pandas
    pdf = pd.read_parquet(dataset_uri)

    logger.info(f"Dataset loaded: {len(pdf)} rows, {len(pdf.columns)} columns")

    # Drop target columns and group column - pipeline should only see features
    columns_to_drop = list(target_columns)
    if group_column and group_column in pdf.columns:
        columns_to_drop.append(group_column)
        logger.info(f"Excluding group column '{group_column}' from pipeline fitting")

    pdf_features = pdf.drop(columns=columns_to_drop, errors="ignore")

    logger.info(f"Fitting pipeline on {len(pdf_features.columns)} feature columns (target/group columns excluded)")

    # Fit pipeline on features only
    pipeline.fit(pdf_features)

    logger.info("Pipeline fitted successfully")

    return pipeline
