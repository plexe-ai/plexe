"""
Hardcoded robust PyTorch training loop.

Trains PyTorch models directly with DataLoader-based batching.
"""

import argparse
import copy
import inspect
import json
import logging
import sys
from pathlib import Path

import cloudpickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from plexe.utils.s3 import download_s3_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def train_pytorch(
    untrained_model_path: Path,
    train_uri: str,
    val_uri: str,
    output_dir: Path,
    target_column: str,
    epochs: int = 25,
    batch_size: int = 32,
) -> dict:
    """
    Train PyTorch model directly.

    Args:
        untrained_model_path: Path to untrained model (pkl via torch.save)
        train_uri: Training data parquet
        val_uri: Validation data parquet
        output_dir: Where to save outputs
        target_column: Target column name
        epochs: Number of training epochs
        batch_size: Batch size for DataLoader

    Returns:
        Training metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load untrained model
    logger.info(f"Loading untrained model from {untrained_model_path}...")
    model = torch.load(untrained_model_path, weights_only=False)
    logger.info(f"Model loaded: {type(model).__name__}")

    # Step 2: Load optimizer/loss config
    config_path = untrained_model_path.parent / "training_config.json"
    logger.info(f"Loading training config from {config_path}...")
    with open(config_path) as f:
        training_config = json.load(f)

    # Recreate optimizer (needs model.parameters())
    optimizer_class = getattr(torch.optim, training_config["optimizer_class"])
    optimizer_config = training_config.get("optimizer_config", {})
    if optimizer_config:
        signature = inspect.signature(optimizer_class.__init__)
        params = signature.parameters
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        if not accepts_kwargs:
            allowed = {name for name in params if name not in ("self", "params")}
            filtered_config = {key: value for key, value in optimizer_config.items() if key in allowed}
            dropped_keys = sorted(set(optimizer_config) - set(filtered_config))
            if dropped_keys:
                logger.warning(
                    "Dropping unsupported optimizer args for %s: %s",
                    optimizer_class.__name__,
                    dropped_keys,
                )
            optimizer_config = filtered_config
    optimizer = optimizer_class(model.parameters(), **optimizer_config)
    logger.info(f"Optimizer: {type(optimizer).__name__}")

    # Recreate loss
    loss_class = getattr(nn, training_config["loss_class"])
    loss_fn = loss_class()
    logger.info(f"Loss: {type(loss_fn).__name__}")

    # Step 3: Download from S3 if needed
    if train_uri.startswith("s3://"):
        train_uri = download_s3_uri(train_uri)
    if val_uri.startswith("s3://"):
        val_uri = download_s3_uri(val_uri)

    # Step 4: Load data and convert to tensors
    logger.info(f"Loading training data from {train_uri}...")
    train_df = pd.read_parquet(train_uri)
    logger.info(f"Training data shape: {train_df.shape}")

    X_train = torch.tensor(train_df.drop(columns=[target_column]).values, dtype=torch.float32)
    y_train_raw = train_df[target_column].values

    logger.info(f"Loading validation data from {val_uri}...")
    val_df = pd.read_parquet(val_uri)
    logger.info(f"Validation data shape: {val_df.shape}")

    X_val = torch.tensor(val_df.drop(columns=[target_column]).values, dtype=torch.float32)
    y_val_raw = val_df[target_column].values

    # Determine task type from loss function and target values
    is_classification = isinstance(loss_fn, nn.CrossEntropyLoss)
    is_binary = isinstance(loss_fn, nn.BCEWithLogitsLoss)

    if is_classification:
        # CrossEntropyLoss expects long targets
        y_train = torch.tensor(y_train_raw, dtype=torch.long)
        y_val = torch.tensor(y_val_raw, dtype=torch.long)
        task_type = "classification"
    elif is_binary:
        y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(y_val_raw, dtype=torch.float32).unsqueeze(1)
        task_type = "classification"
    else:
        # Regression
        y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
        y_val = torch.tensor(y_val_raw, dtype=torch.float32).unsqueeze(1)
        task_type = "regression"

    # Step 5: Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Step 6: Training loop
    logger.info(f"Training for {epochs} epochs, batch_size={batch_size}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Training on device: {device}")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = loss_fn(output, y_batch)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{epochs} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}"
            )

    logger.info("Training complete!")

    # Restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model (val_loss: {best_val_loss:.4f})")

    # Step 7: Save artifacts
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Save model state dict
    model_path = artifacts_dir / "model.pt"
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), model_path)
    logger.info(f"Model state dict saved to {model_path}")

    # Save model class definition via cloudpickle (needed to reconstruct at inference)
    model_class_path = artifacts_dir / "model_class.pkl"
    with open(model_class_path, "wb") as f:
        cloudpickle.dump(model_cpu, f)
    logger.info(f"Model class saved to {model_class_path}")

    # Save training history
    history_path = artifacts_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"History saved to {history_path}")

    # Save metadata (includes optimizer/loss config for faithful retraining)
    metadata = {
        "model_type": "pytorch",
        "training_mode": "direct",
        "epochs": epochs,
        "batch_size": batch_size,
        "best_val_loss": best_val_loss,
        "n_features": X_train.shape[1],
        "target_column": target_column,
        "task_type": task_type,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "optimizer_class": type(optimizer).__name__,
        "optimizer_config": {k: v for k, v in optimizer.defaults.items()},
        "loss_class": type(loss_fn).__name__,
    }

    metadata_path = artifacts_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch model")
    parser.add_argument("--untrained-model", required=True, help="Path to untrained model (pkl)")
    parser.add_argument("--train-uri", required=True, help="Training data URI")
    parser.add_argument("--val-uri", required=True, help="Validation data URI")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    train_pytorch(
        untrained_model_path=Path(args.untrained_model),
        train_uri=args.train_uri,
        val_uri=args.val_uri,
        output_dir=Path(args.output),
        target_column=args.target_column,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    logger.info("Script complete!")
