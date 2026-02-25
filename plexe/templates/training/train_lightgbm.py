"""
Hardcoded robust LightGBM training loop.

Trains LightGBM directly with pandas data loading.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from textwrap import shorten

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.preprocessing import LabelEncoder

from plexe.utils.s3 import download_s3_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# ============================================
# Main Training Function
# ============================================


def train_lightgbm(
    untrained_model_path: Path,
    train_uri: str,
    val_uri: str,
    output_dir: Path,
    target_column: str,
    group_column: str | None = None,
) -> dict:
    """
    Train LightGBM model directly (no Spark).

    Args:
        untrained_model_path: Path to untrained LightGBM model (pkl)
        train_uri: Training data URI (already transformed parquet)
        val_uri: Validation data URI (already transformed parquet)
        output_dir: Where to save outputs
        target_column: Name of target column
        group_column: Optional group column for ranking (query_id, session_id)

    Returns:
        Training metadata
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Download from S3 if needed
    if train_uri.startswith("s3://"):
        train_uri = download_s3_uri(train_uri)
    if val_uri.startswith("s3://"):
        val_uri = download_s3_uri(val_uri)

    # Step 1: Load Untrained Model
    logger.info(f"Loading untrained model from {shorten(str(untrained_model_path), 30)}...")
    model = joblib.load(untrained_model_path)
    logger.info(f"Model type: {type(model).__name__}")
    logger.debug(f"Hyperparameters: {model.get_params()}")

    # Step 2: Load Training Data
    logger.info(f"Loading training data from {shorten(str(train_uri), 30)}...")
    train_df = pd.read_parquet(train_uri)
    logger.info(f"Training data shape: {train_df.shape}")

    # Separate features and target (and group column for ranking)
    columns_to_drop = [target_column]
    if group_column and group_column in train_df.columns:
        columns_to_drop.append(group_column)
        group_train = train_df[group_column]
    else:
        group_train = None

    X_train = train_df.drop(columns=columns_to_drop)
    y_train = train_df[target_column]

    # Step 3: Load Validation Data
    logger.info(f"Loading validation data from {shorten(str(val_uri), 30)}...")
    val_df = pd.read_parquet(val_uri)
    logger.info(f"Validation data shape: {val_df.shape}")

    # Separate features and target (and group column for ranking)
    columns_to_drop_val = [target_column]
    if group_column and group_column in val_df.columns:
        columns_to_drop_val.append(group_column)
        group_val = val_df[group_column]
    else:
        group_val = None

    X_val = val_df.drop(columns=columns_to_drop_val)
    y_val = val_df[target_column]

    # Step 3a: Handle label encoding for classification
    label_encoder = None
    if isinstance(model, LGBMClassifier):
        unique_labels = np.unique(y_train)
        expected_labels = np.arange(len(unique_labels))

        if not np.array_equal(unique_labels, expected_labels):
            logger.info(f"Non-contiguous classification labels detected: {unique_labels}")
            logger.info("Applying LabelEncoder for LightGBM compatibility")

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_val = label_encoder.transform(y_val)
            logger.info(f"Encoded labels: {np.unique(y_train)}")

    # Step 3b: Compute sample weights for class balancing (classification)
    sample_weight = None
    if isinstance(model, LGBMClassifier):
        from sklearn.utils.class_weight import compute_sample_weight

        sample_weight = compute_sample_weight("balanced", y=y_train)

        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        logger.info(f"Computed sample weights for {len(unique_classes)} classes")
        logger.info(f"Weight range: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")

    # Step 3c: Handle ranking-specific setup (LGBMRanker)
    group_sizes_train = None
    group_sizes_val = None
    if isinstance(model, LGBMRanker):
        if group_train is None or group_val is None:
            raise ValueError("LGBMRanker requires group_column parameter for query grouping")

        # Sort train data by group column (required for ranking)
        sort_indices_train = group_train.argsort()
        X_train = X_train.iloc[sort_indices_train].reset_index(drop=True)
        y_train = y_train.iloc[sort_indices_train].reset_index(drop=True)
        group_train_sorted = group_train.iloc[sort_indices_train].reset_index(drop=True)

        # Sort val data by group column
        sort_indices_val = group_val.argsort()
        X_val = X_val.iloc[sort_indices_val].reset_index(drop=True)
        y_val = y_val.iloc[sort_indices_val].reset_index(drop=True)
        group_val_sorted = group_val.iloc[sort_indices_val].reset_index(drop=True)

        # Compute group sizes (number of items per query)
        group_sizes_train = (
            group_train_sorted.groupby((group_train_sorted != group_train_sorted.shift()).cumsum()).size().values
        )
        group_sizes_val = (
            group_val_sorted.groupby((group_val_sorted != group_val_sorted.shift()).cumsum()).size().values
        )

        logger.info("Ranking task detected:")
        logger.info(f"  Train: {len(group_sizes_train)} queries, {len(X_train)} total items")
        logger.info(f"  Val: {len(group_sizes_val)} queries, {len(X_val)} total items")
        logger.info(f"  Avg items per query: {len(X_train) / len(group_sizes_train):.1f}")

    # Step 4: Train Model
    logger.info("Training LightGBM model...")

    import lightgbm as lgb

    callbacks = [lgb.early_stopping(50), lgb.log_evaluation(50)]

    if isinstance(model, LGBMRanker):
        # Ranking: pass group sizes
        model.fit(
            X_train,
            y_train,
            group=group_sizes_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_sizes_val],
            callbacks=callbacks,
        )
    else:
        # Classification/Regression: pass sample_weight (None for regression)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weight,
            callbacks=callbacks,
        )

    logger.info("Training complete!")

    # Step 5: Create artifacts/ subdirectory
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Step 6: Save Trained Model
    model_path = artifacts_dir / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Trained model saved to {shorten(str(model_path), 30)}")

    # Step 7: Save LabelEncoder (if used)
    if label_encoder is not None:
        encoder_path = artifacts_dir / "label_encoder.pkl"
        joblib.dump(label_encoder, encoder_path)
        logger.info(f"LabelEncoder saved to {shorten(str(encoder_path), 30)}")

    # Step 8: Save Metadata
    if isinstance(model, LGBMRanker):
        task_type = "ranking"
    elif isinstance(model, LGBMClassifier):
        task_type = "classification"
    else:
        task_type = "regression"

    metadata = {
        "model_type": "lightgbm",
        "training_mode": "direct",
        "hyperparameters": model.get_params(),
        "best_iteration": model.best_iteration_ if hasattr(model, "best_iteration_") else None,
        "n_features": X_train.shape[1],
        "target_column": target_column,
        "task_type": task_type,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }

    # Add ranking-specific metadata
    if isinstance(model, LGBMRanker):
        metadata["group_column"] = group_column
        metadata["n_train_queries"] = len(group_sizes_train) if group_sizes_train is not None else None
        metadata["n_val_queries"] = len(group_sizes_val) if group_sizes_val is not None else None

    metadata_path = artifacts_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {shorten(str(metadata_path), 30)}")

    return metadata


# ============================================
# CLI Entry Point
# ============================================


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--untrained-model", required=True, help="Path to untrained model PKL")
    parser.add_argument("--train-uri", required=True, help="Training data URI")
    parser.add_argument("--val-uri", required=True, help="Validation data URI")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument(
        "--group-column", required=False, default=None, help="Group column for ranking (query_id, session_id)"
    )
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    metadata = train_lightgbm(
        untrained_model_path=Path(args.untrained_model),
        train_uri=args.train_uri,
        val_uri=args.val_uri,
        output_dir=Path(args.output),
        target_column=args.target_column,
        group_column=args.group_column,
    )

    logger.info("Training complete!")
    logger.debug(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
