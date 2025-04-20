"""
Minimal GPU detection tool for Plexe agents.
"""

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
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info.update({
                "available": True, 
                "framework": "pytorch",
                "device": "cuda",
                "count": torch.cuda.device_count()
            })
            return gpu_info
    except ImportError:
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
        ray_info.update({
            "available": True,
            "gpus": resources.get("GPU", 0),
            "cpus": resources.get("CPU", 0),
            "nodes": sum(1 for k in resources.keys() if k.startswith("node:"))
        })
    except ImportError:
        pass
    
    return ray_info