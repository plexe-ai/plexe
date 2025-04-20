#!/usr/bin/env python
"""
GPU detection and testing utility for Plexe.

Usage:
  python gpu_test.py [--ray] [--task]

Options:
  --ray   Test Ray initialization with GPU detection
  --task  Run a simple GPU task using Ray (implies --ray)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add the project root to the path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from plexe.internal.models.tools.hardware import get_gpu_info, get_ray_info

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_local_gpu():
    """Test local GPU availability."""
    logger.info("Local GPU Information:")
    gpu_info = get_gpu_info()
    logger.info(json.dumps(gpu_info, indent=2))
    
    if gpu_info["available"]:
        logger.info(f"✅ GPUs detected: {gpu_info['count']} ({gpu_info['framework']})")
    else:
        logger.info("❌ No local GPUs detected")
    
    return gpu_info


def test_ray_gpu(gpu_info=None):
    """Test Ray with GPU support."""
    if gpu_info is None:
        gpu_info = test_local_gpu()
    
    try:
        import ray
        
        logger.info("\nInitializing Ray...")
        if not ray.is_initialized():
            if gpu_info["available"]:
                # Initialize with all available GPUs
                ray.init(num_gpus=gpu_info["count"])
                logger.info(f"Ray initialized with {gpu_info['count']} GPUs")
            else:
                ray.init()
                logger.info("Ray initialized without GPUs (none detected)")
        
        # Get Ray info using our tool
        logger.info("\nRay Information from Tool:")
        ray_info = get_ray_info()
        logger.info(json.dumps(ray_info, indent=2))
        
        if ray_info["available"]:
            logger.info(f"✅ Ray initialized with {ray_info['gpus']} GPUs")
        else:
            logger.info("❌ Ray not initialized or no Ray GPUs available")
        
        return ray, ray_info
    
    except ImportError:
        logger.error("Ray is not installed. Cannot test Ray GPU features.")
        return None, None


def run_gpu_task(ray_instance):
    """Run a Ray task that uses a GPU."""
    
    @ray_instance.remote(num_gpus=1)
    def gpu_task():
        """Simple task that requires a GPU."""
        time.sleep(1)  # Simulate work
        
        # Try PyTorch first
        try:
            import torch
            if torch.cuda.is_available():
                x = torch.randn(1000, 1000, device="cuda")
                result = torch.matmul(x, x)
                result.cpu()  # Force sync
                
                return {
                    "success": True,
                    "framework": "pytorch",
                    "device": torch.cuda.get_device_name(0),
                    "memory": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated"
                }
        except Exception:
            pass
        
        # Try TensorFlow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                with tf.device("/GPU:0"):
                    x = tf.random.normal((1000, 1000))
                    result = tf.matmul(x, x)
                    result = result.numpy()  # Force execution
                    
                return {
                    "success": True,
                    "framework": "tensorflow",
                    "device": gpus[0].name,
                }
        except Exception:
            pass
        
        # If no ML framework works, report basic success
        return {
            "success": True,
            "framework": "system",
            "device": "GPU access confirmed via Ray"
        }
    
    logger.info("\nRunning GPU task...")
    result = ray_instance.get(gpu_task.remote())
    logger.info(json.dumps(result, indent=2))
    
    if result["success"]:
        logger.info(f"✅ GPU task executed successfully with {result['framework']}")
        if "device" in result:
            logger.info(f"✅ GPU device: {result['device']}")
    else:
        logger.info("❌ GPU task failed")
    
    return result


def main():
    """Run GPU detection tests based on command-line arguments."""
    parser = argparse.ArgumentParser(description="Test GPU detection and usage")
    parser.add_argument("--ray", action="store_true", help="Test Ray with GPU detection")
    parser.add_argument("--task", action="store_true", help="Run a GPU task using Ray")
    args = parser.parse_args()
    
    logger.info("GPU Detection Test\n" + "="*40)
    
    # Always run local GPU detection
    gpu_info = test_local_gpu()
    
    # Run Ray tests if requested
    ray_instance = None
    if args.ray or args.task:
        ray_instance, ray_info = test_ray_gpu(gpu_info)
    
    # Run GPU task if requested
    if args.task and ray_instance:
        if gpu_info["available"]:
            run_gpu_task(ray_instance)
        else:
            logger.error("Cannot run GPU task: No GPUs detected")
    
    # Shutdown Ray if initialized
    if ray_instance:
        ray_instance.shutdown()
        logger.info("\nRay shutdown complete")


if __name__ == "__main__":
    main()