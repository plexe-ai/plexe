#!/usr/bin/env python
"""
Integration test for Plexe GPU support.
This script tests GPU acceleration with XGBoost through Ray.
"""

import os
import time
import logging
import numpy as np
import ray
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check GPU availability
gpu_available = torch.cuda.is_available()
if gpu_available:
    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
else:
    logger.info("No GPU available, will run on CPU only")

# Initialize Ray with GPU if available
if not ray.is_initialized():
    if gpu_available:
        ray.init(num_gpus=1)
        logger.info("Ray initialized with GPU support")
        logger.info(f"Ray resources: {ray.cluster_resources()}")
    else:
        ray.init()
        logger.info("Ray initialized without GPU")


@ray.remote(num_gpus=1 if gpu_available else 0)
def train_xgboost_model():
    """Train an XGBoost model using GPU if available."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
    except ImportError:
        logger.error("XGBoost or scikit-learn not installed")
        return {"success": False, "error": "Required packages not installed"}

    logger.info("Training XGBoost model...")

    # Check GPU availability within the Ray task
    gpu_params = {}
    if torch.cuda.is_available():
        logger.info(f"GPU available for XGBoost: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory at start: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
        # Set XGBoost GPU parameters
        gpu_params = {"tree_method": "gpu_hist", "gpu_id": 0, "predictor": "gpu_predictor"}
        # Set environment variable for GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        logger.info("No GPU available for XGBoost, using CPU")

    # Create synthetic dataset
    n_samples = 50000
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = 2 + 3 * X[:, 0] + 0.5 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set XGBoost parameters
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Add GPU parameters if available
    params.update(gpu_params)

    logger.info(f"XGBoost parameters: {params}")

    # Train XGBoost model
    logger.info("Starting XGBoost training...")
    start_time = time.time()

    # Define watchlist for training
    watchlist = [(dtrain, "train"), (dtest, "test")]

    # Train model
    num_rounds = 100
    xgb_model = xgb.train(
        params, dtrain, num_rounds, evals=watchlist, verbose_eval=25  # Print evaluation every 25 iterations
    )

    train_time = time.time() - start_time

    # Make predictions
    y_pred = xgb_model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"Training completed in {train_time:.2f} seconds")
    logger.info(f"RMSE: {rmse:.6f}")

    # Get GPU memory usage if available
    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory = {
            "memory_end": torch.cuda.memory_allocated() / (1024**2),
            "memory_max": torch.cuda.max_memory_allocated() / (1024**2),
        }
        logger.info(f"GPU memory at end: {gpu_memory['memory_end']:.2f} MB")
        logger.info(f"GPU max memory: {gpu_memory['memory_max']:.2f} MB")

    # Return performance metrics
    return {
        "success": True,
        "training_time": train_time,
        "rmse": float(rmse),
        "parameters": params,
        "used_gpu": "gpu_hist" in params.get("tree_method", ""),
        "gpu_memory": gpu_memory,
    }


def main():
    """Run the XGBoost integration test."""
    logger.info("Starting GPU integration test")

    # Run XGBoost training using Ray
    logger.info("Launching XGBoost training task with Ray...")
    result = ray.get(train_xgboost_model.remote())

    if result.get("success", False):
        logger.info("Test successful!")
        logger.info(f"Training time: {result['training_time']:.2f} seconds")
        logger.info(f"RMSE: {result['rmse']:.6f}")
        logger.info(f"GPU used: {'Yes' if result['used_gpu'] else 'No'}")

        if result["used_gpu"] and result.get("gpu_memory"):
            logger.info(f"GPU memory used: {result['gpu_memory']['memory_max']:.2f} MB")

        # Calculate speedup compared to CPU (if we had CPU benchmarks)
        # Here we could compare to predetermined CPU times or run another test on CPU
    else:
        logger.error(f"Test failed: {result.get('error', 'Unknown error')}")

    # Clean up Ray
    ray.shutdown()
    logger.info("Ray shutdown complete")
    logger.info("Integration test completed")


if __name__ == "__main__":
    main()
