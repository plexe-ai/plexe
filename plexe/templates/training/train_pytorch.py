"""
PyTorch training template with streaming data loading, multi-GPU (DDP), and mixed precision.

Supports:
- Streaming parquet data via ParquetIterableDataset (handles 100GB+ datasets)
- Single GPU, multi-GPU (DDP via torchrun), and CPU training
- Mixed precision (FP16) for faster training and lower memory usage
- Best model checkpointing to disk (not memory)
"""

import argparse
import inspect
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import cloudpickle
import torch
import torch.distributed as dist
import torch.nn as nn

from plexe.utils.parquet_dataset import (
    ParquetIterableDataset,
    get_parquet_feature_count,
    get_parquet_row_count,
)
from plexe.utils.s3 import download_s3_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _infer_task_type(loss_fn: nn.Module) -> str:
    """Infer task type from loss function (legacy fallback when --task-type not provided)."""
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        return "multiclass_classification"
    elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
        return "binary_classification"
    return "regression"


def _is_rank0(use_ddp: bool) -> bool:
    """Check if this is rank 0 (or non-distributed)."""
    if not use_ddp:
        return True
    return dist.get_rank() == 0


def _resolve_num_workers(requested_workers: int) -> int:
    """Resolve safe DataLoader worker count for the current runtime."""
    if requested_workers <= 0:
        return 0

    start_method = mp.get_start_method(allow_none=True)
    if start_method is None:
        start_method = mp.get_context().get_start_method()

    if sys.platform == "darwin" and start_method == "spawn":
        logger.warning(
            "Falling back DataLoader workers from %s to 0 on platform=%s start_method=%s",
            requested_workers,
            sys.platform,
            start_method,
        )
        return 0

    return requested_workers


def train_pytorch(
    untrained_model_path: Path,
    train_uri: str,
    val_uri: str,
    output_dir: Path,
    target_column: str,
    epochs: int = 10,
    batch_size: int = 32,
    num_workers: int = 0,
    use_ddp: bool = False,
    use_mixed_precision: bool = False,
    task_type: str | None = None,
) -> dict:
    """
    Train PyTorch model with streaming data, optional DDP, and mixed precision.

    Args:
        untrained_model_path: Path to untrained model (pkl via torch.save)
        train_uri: Training data parquet (file or directory)
        val_uri: Validation data parquet (file or directory)
        output_dir: Where to save outputs
        target_column: Target column name
        epochs: Number of training epochs
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader worker processes
        use_ddp: Whether DDP is active (set by torchrun launcher)
        use_mixed_precision: Whether to use FP16 mixed precision
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================
    # Step 1: Setup device and distributed training
    # ============================================
    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA but no GPU is available. Remove --ddp to train on CPU.")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        gpu_count = dist.get_world_size()
        logger.info(f"DDP initialized: rank {dist.get_rank()}/{gpu_count}, device {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_count = torch.cuda.device_count()
        logger.info(f"Single GPU training on {device} ({gpu_count} GPU(s) available)")
    else:
        device = torch.device("cpu")
        gpu_count = 0
        use_mixed_precision = False  # AMP requires CUDA
        logger.info("Training on CPU")

    rank0 = _is_rank0(use_ddp)

    # ============================================
    # Step 2: Load untrained model
    # ============================================
    if rank0:
        logger.info(f"Loading untrained model from {untrained_model_path}...")
    model = torch.load(untrained_model_path, weights_only=False, map_location="cpu")
    model = model.to(device)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if rank0:
        logger.info(f"Model loaded: {type(model).__name__}")

    # ============================================
    # Step 3: Load optimizer/loss config
    # ============================================
    config_path = untrained_model_path.parent / "training_config.json"
    with open(config_path) as f:
        training_config = json.load(f)

    # Recreate optimizer (needs model.parameters() — works with DDP wrapper)
    optimizer_class = getattr(torch.optim, training_config["optimizer_class"])
    optimizer_config = training_config.get("optimizer_config", {})
    optimizer = optimizer_class(model.parameters(), **optimizer_config)

    # Recreate loss
    loss_class = getattr(nn, training_config["loss_class"])
    loss_fn = loss_class()

    if not task_type:
        task_type = _infer_task_type(loss_fn)
    if rank0:
        logger.info(f"Optimizer: {type(optimizer).__name__}, Loss: {type(loss_fn).__name__}, Task: {task_type}")

    # ============================================
    # Step 4: Download from S3 if needed
    # ============================================
    if train_uri.startswith("s3://"):
        train_uri = download_s3_uri(train_uri)
    if val_uri.startswith("s3://"):
        val_uri = download_s3_uri(val_uri)

    # ============================================
    # Step 5: Create streaming DataLoaders
    # ============================================
    train_dataset = ParquetIterableDataset(train_uri, target_column, task_type)
    val_dataset = ParquetIterableDataset(val_uri, target_column, task_type)

    effective_num_workers = _resolve_num_workers(num_workers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=effective_num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=effective_num_workers,
        pin_memory=device.type == "cuda",
    )

    n_features = get_parquet_feature_count(train_uri, target_column)
    train_rows = get_parquet_row_count(train_uri)
    val_rows = get_parquet_row_count(val_uri)

    if rank0:
        logger.info("Using ParquetIterableDataset for streaming data loading")
        logger.info(f"Training data: {train_rows} rows, {n_features} features (streaming)")
        logger.info(f"Validation data: {val_rows} rows (streaming)")
        logger.info(f"DataLoader workers: requested={num_workers}, effective={effective_num_workers}")

    # ============================================
    # Step 6: Setup mixed precision
    # ============================================
    scaler = torch.amp.GradScaler("cuda") if use_mixed_precision else None
    autocast_ctx = torch.amp.autocast("cuda") if use_mixed_precision else torch.amp.autocast("cpu", enabled=False)

    if rank0 and use_mixed_precision:
        logger.info("Mixed precision (FP16) enabled")

    # ============================================
    # Step 7: Training loop
    # ============================================
    if rank0:
        logger.info(f"Training for {epochs} epochs, batch_size={batch_size}...")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_checkpoint_path = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            with autocast_ctx:
                output = model(X_batch)
                loss = loss_fn(output, y_batch)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                with autocast_ctx:
                    output = model(X_batch)
                    loss = loss_fn(output, y_batch)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Track best model — save checkpoint to disk instead of memory.
        # Note: only rank 0 updates best_val_loss and saves checkpoints. Non-rank-0
        # processes retain best_val_loss=inf, which is intentional — only rank 0's
        # history and artifacts are used downstream.
        if avg_val_loss < best_val_loss and rank0:
            best_val_loss = avg_val_loss
            # Get the underlying model (unwrap DDP if needed)
            raw_model = model.module if use_ddp else model
            best_checkpoint_path = output_dir / "_best_checkpoint.pt"
            torch.save(raw_model.state_dict(), best_checkpoint_path)

        if rank0 and ((epoch + 1) % 5 == 0 or epoch == 0):
            logger.info(
                f"  Epoch {epoch + 1}/{epochs} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}"
            )

    if rank0:
        logger.info("Training complete!")

    # ============================================
    # Step 8: Save artifacts (rank 0 only)
    # ============================================
    metadata = {}
    if rank0:
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Restore best model weights
        raw_model = model.module if use_ddp else model
        if best_checkpoint_path and best_checkpoint_path.exists():
            raw_model.load_state_dict(torch.load(best_checkpoint_path, weights_only=True, map_location=device))
            best_checkpoint_path.unlink()  # Clean up temp checkpoint
            logger.info(f"Restored best model (val_loss: {best_val_loss:.4f})")

        # Save model state dict
        model_cpu = raw_model.to("cpu")
        model_path = artifacts_dir / "model.pt"
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

        init_params = set(inspect.signature(type(optimizer).__init__).parameters.keys())

        # Save metadata
        metadata = {
            "model_type": "pytorch",
            "training_mode": "direct",
            "epochs": epochs,
            "batch_size": batch_size,
            "best_val_loss": best_val_loss,
            "n_features": n_features,
            "target_column": target_column,
            "task_type": task_type,
            "train_samples": train_rows,
            "val_samples": val_rows,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "optimizer_class": type(optimizer).__name__,
            "optimizer_config": {k: v for k, v in optimizer.defaults.items() if k in init_params},
            "loss_class": type(loss_fn).__name__,
            "gpu_count": gpu_count,
            "mixed_precision": use_mixed_precision,
            "device": str(device),
            "distributed": use_ddp,
        }

        metadata_path = artifacts_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

    # ============================================
    # Step 9: Cleanup distributed training
    # ============================================
    if use_ddp:
        dist.destroy_process_group()

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch model")
    parser.add_argument("--untrained-model", required=True, help="Path to untrained model (pkl)")
    parser.add_argument("--train-uri", required=True, help="Training data URI")
    parser.add_argument("--val-uri", required=True, help="Validation data URI")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--ddp", action="store_true", help="Enable DDP (set by torchrun)")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--task-type", required=False, default=None, help="Canonical task type")

    args = parser.parse_args()

    train_pytorch(
        untrained_model_path=Path(args.untrained_model),
        train_uri=args.train_uri,
        val_uri=args.val_uri,
        output_dir=Path(args.output),
        target_column=args.target_column,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_ddp=args.ddp,
        use_mixed_precision=args.mixed_precision,
        task_type=args.task_type,
    )

    logger.info("Script complete!")
