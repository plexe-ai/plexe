#!/usr/bin/env python
"""
Test script for Ray configuration and execution.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# Create a modified config module
import plexe.config as config_module

# Create a new RayConfig class that's not frozen
from dataclasses import dataclass


@dataclass
class TestRayConfig:
    address: str = None
    num_cpus: int = None
    num_gpus: int = None


# Store original
original_ray_config = config_module.config.ray

# Create a new test config
test_ray_config = TestRayConfig()
test_ray_config.num_cpus = 2

# Try to detect GPU availability
try:
    import torch
    gpu_available = torch.cuda.is_available()
except ImportError:
    gpu_available = False

# Set GPU count based on availability
test_ray_config.num_gpus = 1 if gpu_available else 0
print(f"Using num_gpus={test_ray_config.num_gpus} (GPU available: {gpu_available})")

# Patch the config
config_module.config.ray = test_ray_config

# Now import our code that uses the config
from plexe.internal.models.tools.execution import _get_executor_class
from plexe.internal.models.tools.hardware import get_ray_info

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Run tests
logger.info("Testing executor selection with Ray config:")
executor_class = _get_executor_class()
logger.info(f"Selected executor: {executor_class.__name__}")

# Initialize Ray if not already
try:
    import ray

    if not ray.is_initialized():
        logger.info("Initializing Ray with GPU support if available")
        ray.init(num_cpus=2, num_gpus=1 if gpu_available else 0)

    # Test Ray info
    logger.info("\nRay Cluster Information:")
    ray_info = get_ray_info()
    import json

    logger.info(json.dumps(ray_info, indent=2))

    # Shutdown Ray
    ray.shutdown()
    logger.info("Ray shutdown complete")
except ImportError:
    logger.info("Ray not available, skipping Ray tests")

# Restore original config
config_module.config.ray = original_ray_config
