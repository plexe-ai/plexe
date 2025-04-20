#!/usr/bin/env python
"""
Simple script to test GPU detection tools.
"""

import json
import logging
import sys
from pathlib import Path

# Add the project root to the path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from plexe.internal.models.tools.hardware import get_gpu_info, get_ray_info

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run GPU detection test."""
    logger.info("Testing GPU detection tools...\n")
    
    # Get GPU info
    logger.info("Local GPU Information:")
    gpu_info = get_gpu_info()
    logger.info(json.dumps(gpu_info, indent=2))
    
    # Get Ray info
    logger.info("\nRay Cluster Information:")
    ray_info = get_ray_info()
    logger.info(json.dumps(ray_info, indent=2))
    
    # Overall summary
    logger.info("\nSummary:")
    if gpu_info["available"]:
        logger.info(f"✅ GPUs detected: {gpu_info['count']} ({gpu_info['framework']})")
    else:
        logger.info("❌ No local GPUs detected")
    
    if ray_info["available"]:
        logger.info(f"✅ Ray initialized with {ray_info['gpus']} GPUs")
    else:
        logger.info("❌ Ray not initialized or no Ray GPUs available")


if __name__ == "__main__":
    main()