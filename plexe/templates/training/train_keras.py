"""
Hardcoded robust Keras training loop.

Trains Keras 3 models directly with numpy arrays from parquet.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# CRITICAL: Set Keras backend BEFORE importing keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import pandas as pd

from plexe.utils.s3 import download_s3_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def train_keras(
    untrained_model_path: Path,
    train_uri: str,
    val_uri: str,
    output_dir: Path,
    target_column: str,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict:
    """
    Train Keras model directly.

    Args:
        untrained_model_path: Path to .keras model file
        train_uri: Training data parquet
        val_uri: Validation data parquet
        output_dir: Where to save outputs
        target_column: Target column name
        epochs: Number of epochs
        batch_size: Batch size

    Returns:
        Training metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {untrained_model_path}...")
    model = keras.models.load_model(untrained_model_path)
    logger.info(f"Model loaded: {type(model).__name__}")

    # Load optimizer/loss config
    config_path = untrained_model_path.parent / "training_config.json"
    logger.info(f"Loading training config from {config_path}...")
    with open(config_path) as f:
        training_config = json.load(f)

    # Recreate optimizer
    optimizer_class = getattr(keras.optimizers, training_config["optimizer_class"])
    optimizer = optimizer_class.from_config(training_config["optimizer_config"])
    logger.info(f"Optimizer: {type(optimizer).__name__}")

    # Recreate loss
    loss_class = getattr(keras.losses, training_config["loss_class"])
    loss = loss_class.from_config(training_config["loss_config"])
    logger.info(f"Loss: {type(loss).__name__}")

    # Compile (jit_compile=False to avoid deadlocks on some systems)
    logger.info("Compiling model...")
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"], jit_compile=False)
    logger.info("Model compiled successfully")

    # Download from S3 if needed
    if train_uri.startswith("s3://"):
        train_uri = download_s3_uri(train_uri)
    if val_uri.startswith("s3://"):
        val_uri = download_s3_uri(val_uri)

    # Load training data
    logger.info(f"Loading training data from {train_uri}...")
    train_df = pd.read_parquet(train_uri)
    logger.info(f"Training data shape: {train_df.shape}")

    X_train = train_df.drop(columns=[target_column]).values
    y_train = train_df[target_column].values

    # Load validation data
    logger.info(f"Loading validation data from {val_uri}...")
    val_df = pd.read_parquet(val_uri)
    logger.info(f"Validation data shape: {val_df.shape}")

    X_val = val_df.drop(columns=[target_column]).values
    y_val = val_df[target_column].values

    # Train
    logger.info(f"Training for {epochs} epochs, batch_size={batch_size}...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,  # Per-epoch logging (verbose=1 progress bars don't work well in pipes)
    )

    logger.info("Training complete!")

    # Create artifacts/ subdirectory (aligned with final packaging structure)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Save trained model
    model_path = artifacts_dir / "model.keras"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save training history
    history_path = artifacts_dir / "history.json"
    history_data = {key: [float(v) for v in values] for key, values in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    logger.info(f"History saved to {history_path}")

    # Save metadata
    metadata = {
        "model_type": "keras",
        "training_mode": "direct",
        "epochs": epochs,
        "batch_size": batch_size,
        "n_features": X_train.shape[1],
        "target_column": target_column,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }

    metadata_path = artifacts_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Keras model")
    parser.add_argument("--untrained-model", required=True, help="Path to untrained .keras model file")
    parser.add_argument("--train-uri", required=True, help="Training data URI")
    parser.add_argument("--val-uri", required=True, help="Validation data URI")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    train_keras(
        untrained_model_path=Path(args.untrained_model),
        train_uri=args.train_uri,
        val_uri=args.val_uri,
        output_dir=Path(args.output),
        target_column=args.target_column,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    logger.info("Script complete!")
