"""
This module provides utility functions for working with model descriptions and metadata.
"""

import importlib.util
from typing import Dict, Optional


def calculate_model_size(artifacts: list) -> Optional[int]:
    """
    Calculate the total size of the model artifacts in bytes.

    :param artifacts: List of artifacts with path attributes
    :return: The size in bytes or None if no artifacts exist
    """
    if not artifacts:
        return None

    total_size = 0
    for artifact in artifacts:
        if artifact.path and artifact.path.exists():
            total_size += artifact.path.stat().st_size

    return total_size if total_size > 0 else None


def format_code_snippet(code: Optional[str]) -> Optional[str]:
    """
    Format a code snippet for display, truncating if necessary.

    :param code: The source code as a string
    :return: A formatted code snippet or None if code doesn't exist
    """
    if not code:
        return None

    # Limit the size of code displayed, possibly add line numbers, etc.
    lines = code.splitlines()
    if len(lines) > 20:
        # Return first 10 and last 10 lines with a note in the middle
        return "\n".join(lines[:10] + ["# ... additional lines omitted ..."] + lines[-10:])
    return code


def is_package_available(package_name: str) -> bool:
    """
    Check if a Python package is installed and available.

    :param package_name: Name of the package to check
    :return: True if the package is available, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None


def is_gpu_available() -> bool:
    """
    Check if a GPU is available for training.

    :return: True if a GPU is available, False otherwise
    """
    # Check for PyTorch GPU
    if is_package_available("torch"):
        import torch

        if torch.cuda.is_available():
            return True

    # Check for TensorFlow GPU
    if is_package_available("tensorflow"):
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return True

    return False


def get_device() -> str:
    """
    Get the device to use for PyTorch ('cuda' or 'cpu').

    :return: 'cuda' if a GPU is available, 'cpu' otherwise
    """
    if is_package_available("torch") and is_gpu_available():
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def get_gpu_params(framework: str) -> Dict:
    """
    Get GPU-related parameters for different ML frameworks.

    :param framework: Name of the ML framework ('xgboost', 'lightgbm', 'catboost', 'pytorch', 'tensorflow')
    :return: Dictionary with appropriate GPU parameters for the specified framework
    """
    if not is_gpu_available():
        return {}

    framework = framework.lower()

    # XGBoost GPU parameters
    if framework == "xgboost":
        return {"tree_method": "gpu_hist", "gpu_id": 0, "predictor": "gpu_predictor"}
    # LightGBM GPU parameters
    elif framework == "lightgbm":
        return {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
    # CatBoost GPU parameters
    elif framework == "catboost":
        return {"task_type": "GPU", "devices": "0"}
    # PyTorch device
    elif framework == "pytorch" or framework == "torch":
        import torch

        return {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    # TensorFlow GPU config
    elif framework == "tensorflow" or framework == "tf":
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                # Only enable memory growth to avoid allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return {"gpu_enabled": True, "gpu_count": len(gpus)}
        except Exception:
            pass

    # Default empty dict for other frameworks
    return {}
