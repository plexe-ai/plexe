"""
Minimal GPU detection tool for Plexe agents.
"""

import subprocess
from smolagents import tool


@tool
def get_gpu_info() -> dict:
    """
    Get available GPU information for code generation.

    Returns:
        Dictionary with GPU availability and framework information.
        Example: {"available": True, "framework": "pytorch", "device": "cuda", "count": 1}
    """
    gpu_info = {"available": False}

    # Try PyTorch first
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info.update(
                {"available": True, "framework": "pytorch", "device": "cuda", "count": torch.cuda.device_count()}
            )
            return gpu_info
    except Exception:
        pass

    # Try TensorFlow next
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            gpu_info.update({"available": True, "framework": "tensorflow", "device": "gpu", "count": len(gpus)})
            return gpu_info
    except Exception:
        pass

    # Fallback to nvidia-smi if frameworks aren't available
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Count the number of GPUs listed
            gpu_count = len(result.stdout.strip().split("\n"))
            gpu_info.update({"available": True, "framework": "system", "device": "nvidia-gpu", "count": gpu_count})
            return gpu_info
    except Exception:
        pass

    # No GPU found
    return gpu_info


@tool
def get_ray_info() -> dict:
    """
    Get Ray cluster information including GPU availability.

    Returns:
        Dictionary with Ray cluster information.
        Example: {"available": True, "gpus": 4, "cpus": 8}
    """
    ray_info = {"available": False}

    try:
        import ray

        if not ray.is_initialized():
            return ray_info

        # Get cluster resources
        resources = ray.cluster_resources()
        ray_info.update(
            {
                "available": True,
                "gpus": resources.get("GPU", 0),
                "cpus": resources.get("CPU", 0),
                "nodes": sum(1 for k in resources.keys() if k.startswith("node:")),
            }
        )
    except ImportError:
        pass

    return ray_info
