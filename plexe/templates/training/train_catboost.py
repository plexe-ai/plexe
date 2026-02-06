"""
Hardcoded robust CatBoost training loop.

Trains CatBoost directly with early stopping and native serialization.
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
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

from plexe.utils.s3 import download_s3_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# ============================================
# Main Training Function
# ============================================


def train_catboost(
    untrained_model_path: Path,
    train_uri: str,
    val_uri: str,
    output_dir: Path,
    target_column: str,
) -> dict:
    """
    Train CatBoost model directly (no Spark).

    Loads data with pandas, trains CatBoost with validation and early stopping.
    Uses CatBoost native serialization (.cbm format) for robust model persistence.

    Args:
        untrained_model_path: Path to untrained CatBoost model (.cbm file)
        train_uri: Training data URI (already transformed parquet)
        val_uri: Validation data URI (already transformed parquet)
        output_dir: Where to save outputs
        target_column: Name of target column

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
    # IMPORTANT: Untrained models are pickled (CatBoost doesn't support .save_model() on untrained models)
    logger.info(f"Loading untrained model from {shorten(str(untrained_model_path), 30)}...")
    model = joblib.load(untrained_model_path)

    # Determine if it's classification or regression
    is_classification = isinstance(model, CatBoostClassifier)
    logger.info(f"Model type: {'CatBoostClassifier' if is_classification else 'CatBoostRegressor'}")
    logger.debug(f"Hyperparameters: {model.get_params()}")

    # Step 2: Load Training Data
    logger.info(f"Loading training data from {shorten(str(train_uri), 30)}...")
    train_df = pd.read_parquet(train_uri)
    logger.info(f"Training data shape: {train_df.shape}")

    # Separate features and target
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Step 3: Load Validation Data
    logger.info(f"Loading validation data from {shorten(str(val_uri), 30)}...")
    val_df = pd.read_parquet(val_uri)
    logger.info(f"Validation data shape: {val_df.shape}")

    # Separate features and target
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    # Step 3a: Handle label encoding for classification
    label_encoder = None
    if is_classification:
        unique_labels = np.unique(y_train)
        expected_labels = np.arange(len(unique_labels))

        if not np.array_equal(unique_labels, expected_labels):
            logger.info(f"Non-contiguous classification labels detected: {unique_labels}")
            logger.info("Applying LabelEncoder for CatBoost compatibility")

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_val = label_encoder.transform(y_val)
            logger.info(f"Encoded labels: {np.unique(y_train)}")

    # Step 3b: Compute class weights for imbalanced classification
    class_weights_dict = None
    if is_classification:
        from sklearn.utils.class_weight import compute_class_weight

        # Compute balanced class weights
        unique_classes = np.unique(y_train)
        class_weights_array = compute_class_weight("balanced", classes=unique_classes, y=y_train)
        class_weights_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights_array)}

        # Log class distribution and weights
        unique_classes_count, class_counts = np.unique(y_train, return_counts=True)
        logger.info(f"Class distribution: {dict(zip(unique_classes_count, class_counts))}")
        logger.info(f"Computed class weights: {class_weights_dict}")

    # Step 4: Train Model
    logger.info("Training CatBoost model with validation and early stopping...")

    # Update model with class weights if computed (overrides auto_class_weights if set)
    if class_weights_dict is not None:
        model.set_params(class_weights=class_weights_dict)
        logger.info("Applied class weights to model")

    # Train with validation set for early stopping
    # CatBoost uses eval_set as a tuple (not list like XGBoost)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,  # Use best iteration, not last
        verbose=False,  # Reduce training output
    )

    logger.info("Training complete!")
    logger.info(f"Best iteration: {model.best_iteration_}")
    logger.info(f"Best score: {model.best_score_}")

    # Step 5: Create artifacts/ subdirectory
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Step 6: Save Trained Model (using CatBoost native format)
    # CRITICAL: Use .save_model() instead of pickle/joblib for CatBoost
    # pickle has known issues with CatBoost (best_iteration_ becomes None)
    model_path = artifacts_dir / "model.cbm"
    model.save_model(str(model_path))
    logger.info(f"Trained model saved to {shorten(str(model_path), 30)}")

    # Step 7: Save LabelEncoder (if used)
    if label_encoder is not None:
        encoder_path = artifacts_dir / "label_encoder.pkl"
        joblib.dump(label_encoder, encoder_path)
        logger.info(f"LabelEncoder saved to {shorten(str(encoder_path), 30)}")

    # Step 8: Save Metadata
    metadata = {
        "model_type": "catboost",
        "training_mode": "direct",
        "hyperparameters": model.get_params(),
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ is not None else None,
        "best_score": model.best_score_,  # Nested dict - save as-is (JSON handles this fine)
        "n_features": X_train.shape[1],
        "target_column": target_column,
        "task_type": "classification" if is_classification else "regression",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }

    metadata_path = artifacts_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {shorten(str(metadata_path), 30)}")

    return metadata


# ============================================
# CLI Entry Point
# ============================================


def main():
    parser = argparse.ArgumentParser(description="Train CatBoost model")
    parser.add_argument("--untrained-model", required=True, help="Path to untrained model CBM")
    parser.add_argument("--train-uri", required=True, help="Training data URI")
    parser.add_argument("--val-uri", required=True, help="Validation data URI")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()

    # Load inputs
    untrained_model_path = Path(args.untrained_model)

    # Train
    metadata = train_catboost(
        untrained_model_path=untrained_model_path,
        train_uri=args.train_uri,
        val_uri=args.val_uri,
        output_dir=Path(args.output),
        target_column=args.target_column,
    )

    logger.info("Training complete!")
    logger.debug(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
