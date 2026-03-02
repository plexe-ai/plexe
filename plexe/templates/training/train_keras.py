"""
Keras training template with streaming data loading, multi-GPU (MirroredStrategy), and mixed precision.

Supports:
- Streaming parquet data via tf.data.Dataset + generator (handles 100GB+ datasets)
- Single GPU, multi-GPU (MirroredStrategy), and CPU training
- Mixed precision (FP16) for faster training and lower memory usage
- EarlyStopping with best model restoration
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
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf

from plexe.utils.parquet_dataset import (
    get_parquet_feature_count,
    get_parquet_row_count,
)
from plexe.utils.s3 import download_s3_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


_STREAMING_THRESHOLD_ROWS = 100_000
_STREAMING_THRESHOLD_BYTES = 1_000_000_000  # 1 GB


def _create_tf_dataset(
    uri: str,
    target_column: str,
    batch_size: int,
    n_features: int,
    total_rows: int,
    task_type: str | None = None,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from parquet files.

    For small datasets (< 100k rows and < 1GB): loads into memory via tensor slices (fast, no threads).
    For large datasets: streams row-by-row via from_generator (memory-efficient).
    """
    import pandas as pd

    from plexe.utils.parquet_dataset import get_dataset_size_bytes

    y_np_dtype = np.int32 if task_type == "multiclass_classification" else np.float32
    y_tf_dtype = tf.int32 if task_type == "multiclass_classification" else tf.float32

    dataset_bytes = get_dataset_size_bytes(uri)
    if total_rows < _STREAMING_THRESHOLD_ROWS and dataset_bytes < _STREAMING_THRESHOLD_BYTES:
        # Small dataset: load fully into memory (fast, avoids TF generator threading issues)
        df = pd.read_parquet(uri)
        feature_cols = [c for c in df.columns if c != target_column]
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_column].values.astype(y_np_dtype)
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        # Large dataset: stream from parquet to avoid OOM
        from plexe.utils.parquet_dataset import _resolve_parquet_files

        def row_generator():
            files = _resolve_parquet_files(uri)
            for file_path in files:
                parquet_file = pq.ParquetFile(file_path)
                columns = [c for c in parquet_file.schema_arrow.names if c != target_column]
                for batch in parquet_file.iter_batches(batch_size=4096, columns=columns + [target_column]):
                    batch_df = batch.to_pandas()
                    X_batch = batch_df[columns].values.astype(np.float32)
                    y_batch = batch_df[target_column].values.astype(y_np_dtype)
                    for i in range(len(X_batch)):
                        yield X_batch[i], y_batch[i]

        dataset = tf.data.Dataset.from_generator(
            row_generator,
            output_signature=(
                tf.TensorSpec(shape=(n_features,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=y_tf_dtype),
            ),
        )
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_keras(
    untrained_model_path: Path,
    train_uri: str,
    val_uri: str,
    output_dir: Path,
    target_column: str,
    epochs: int = 10,
    batch_size: int = 32,
    use_multi_gpu: bool = False,
    use_mixed_precision: bool = False,
    task_type: str | None = None,
) -> dict:
    """
    Train Keras model with streaming data, optional multi-GPU, and mixed precision.

    Args:
        untrained_model_path: Path to .keras model file
        train_uri: Training data parquet
        val_uri: Validation data parquet
        output_dir: Where to save outputs
        target_column: Target column name
        epochs: Number of epochs
        batch_size: Batch size
        use_multi_gpu: Whether to use MirroredStrategy for multi-GPU
        use_mixed_precision: Whether to use FP16 mixed precision
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================
    # Step 1: Setup GPU and distribution strategy
    # ============================================
    gpus = tf.config.list_physical_devices("GPU")
    gpu_count = len(gpus)

    if use_mixed_precision and gpu_count > 0:
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision (FP16) enabled")
    else:
        use_mixed_precision = False

    strategy = None
    if use_multi_gpu and gpu_count > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"MirroredStrategy: {gpu_count} GPUs")
    elif gpu_count > 0:
        logger.info(f"Single GPU training ({gpu_count} GPU(s) available)")
    else:
        logger.info("Training on CPU")

    # ============================================
    # Step 2: Load model and compile (inside strategy scope if multi-GPU)
    # ============================================
    config_path = untrained_model_path.parent / "training_config.json"
    logger.info(f"Loading training config from {config_path}...")
    with open(config_path) as f:
        training_config = json.load(f)

    def _load_and_compile():
        model = keras.models.load_model(untrained_model_path)
        logger.info(f"Model loaded: {type(model).__name__}")

        optimizer_class = getattr(keras.optimizers, training_config["optimizer_class"])
        optimizer = optimizer_class.from_config(training_config["optimizer_config"])
        loss_class = getattr(keras.losses, training_config["loss_class"])
        loss = loss_class.from_config(training_config["loss_config"])

        metrics = None if task_type == "regression" else ["accuracy"]
        logger.info(f"Optimizer: {type(optimizer).__name__}, Loss: {type(loss).__name__}, Metrics: {metrics}")
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False)
        return model

    if strategy is not None:
        with strategy.scope():
            model = _load_and_compile()
    else:
        model = _load_and_compile()

    logger.info("Model compiled successfully")

    # ============================================
    # Step 3: Download from S3 if needed
    # ============================================
    if train_uri.startswith("s3://"):
        train_uri = download_s3_uri(train_uri)
    if val_uri.startswith("s3://"):
        val_uri = download_s3_uri(val_uri)

    # ============================================
    # Step 4: Create streaming data pipelines
    # ============================================
    n_features = get_parquet_feature_count(train_uri, target_column)
    train_rows = get_parquet_row_count(train_uri)
    val_rows = get_parquet_row_count(val_uri)
    logger.info(f"Training data: {train_rows} rows, {n_features} features (streaming)")
    logger.info(f"Validation data: {val_rows} rows (streaming)")

    train_dataset = _create_tf_dataset(train_uri, target_column, batch_size, n_features, train_rows, task_type)
    val_dataset = _create_tf_dataset(val_uri, target_column, batch_size, n_features, val_rows, task_type)

    # ============================================
    # Step 5: Train with EarlyStopping
    # ============================================
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ]

    logger.info(f"Training for up to {epochs} epochs, batch_size={batch_size}...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Determine if early stopping occurred
    actual_epochs = len(history.history["loss"])
    early_stopped = actual_epochs < epochs

    logger.info(
        f"Training complete! Ran {actual_epochs}/{epochs} epochs" + (" (early stopped)" if early_stopped else "")
    )

    # ============================================
    # Step 6: Save artifacts
    # ============================================
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

    # Save metadata (including optimizer/loss config for faithful retraining)
    metadata = {
        "model_type": "keras",
        "training_mode": "direct",
        "task_type": task_type or "",
        "epochs": actual_epochs,
        "max_epochs": epochs,
        "batch_size": batch_size,
        "n_features": n_features,
        "target_column": target_column,
        "train_samples": train_rows,
        "val_samples": val_rows,
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "optimizer_class": training_config["optimizer_class"],
        "optimizer_config": training_config["optimizer_config"],
        "loss_class": training_config["loss_class"],
        "loss_config": training_config["loss_config"],
        "gpu_count": gpu_count,
        "mixed_precision": use_mixed_precision,
        "distributed": use_multi_gpu and strategy is not None,
        "early_stopped_epoch": actual_epochs if early_stopped else None,
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
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--multi-gpu", action="store_true", help="Enable MirroredStrategy for multi-GPU")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--task-type", required=False, default=None, help="Canonical task type")

    args = parser.parse_args()

    train_keras(
        untrained_model_path=Path(args.untrained_model),
        train_uri=args.train_uri,
        val_uri=args.val_uri,
        output_dir=Path(args.output),
        target_column=args.target_column,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_multi_gpu=args.multi_gpu,
        use_mixed_precision=args.mixed_precision,
        task_type=args.task_type,
    )

    logger.info("Script complete!")
