#!/usr/bin/env python
"""
Script to test Ray GPU detection.
"""

import json
import logging
import sys
from pathlib import Path

# Add the project root to the path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import ray
from plexe.internal.models.tools.hardware import get_gpu_info, get_ray_info

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    """Test Ray with GPU support."""
    logger.info("Testing Ray with GPU support...\n")
    
    # Get local GPU info first
    logger.info("Local GPU Information:")
    gpu_info = get_gpu_info()
    logger.info(json.dumps(gpu_info, indent=2))
    
    # Initialize Ray with GPU support if available
    logger.info("\nInitializing Ray...")
    if not ray.is_initialized():
        if gpu_info["available"]:
            # Initialize with all available GPUs
            ray.init(num_gpus=gpu_info["count"])
            logger.info(f"Ray initialized with {gpu_info['count']} GPUs")
        else:
            ray.init()
            logger.info("Ray initialized without GPUs (none detected)")
    
    # Check Ray resources
    logger.info("\nRay Cluster Resources:")
    resources = ray.cluster_resources()
    logger.info(json.dumps(resources, indent=2))
    
    # Get Ray info using our tool
    logger.info("\nRay Information from Tool:")
    ray_info = get_ray_info()
    logger.info(json.dumps(ray_info, indent=2))
    
    # Overall summary
    logger.info("\nSummary:")
    if ray_info["available"]:
        logger.info(f"✅ Ray initialized with {ray_info['gpus']} GPUs")
    else:
        logger.info("❌ Ray not initialized or no Ray GPUs available")
    
    # Shutdown Ray
    ray.shutdown()
    logger.info("Ray shutdown complete")


if __name__ == "__main__":
    main()