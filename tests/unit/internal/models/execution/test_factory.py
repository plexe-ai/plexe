"""Test the executor factory."""

import pytest
from unittest.mock import patch, MagicMock

import importlib

from plexe.internal.models.tools.execution import _get_executor_class
from plexe.internal.models.execution.process_executor import ProcessExecutor


def test_get_executor_class_no_ray():
    """Test that ProcessExecutor is returned when Ray is not available."""
    # Mock ray not being available
    with patch.dict("sys.modules", {"ray": None}):
        executor_class = _get_executor_class()
        assert executor_class == ProcessExecutor


def test_get_executor_class_with_ray_initialized():
    """Test that RayExecutor is returned when Ray is initialized."""
    # Check if Ray is available
    ray_available = importlib.util.find_spec("ray") is not None

    if ray_available:
        import ray

        # Create mock for Ray
        ray_mock = MagicMock()
        ray_mock.is_initialized.return_value = True

        with patch.dict("sys.modules", {"ray": ray_mock}):
            executor_class = _get_executor_class()
            from plexe.internal.models.execution.ray_executor import RayExecutor

            assert executor_class == RayExecutor
    else:
        pytest.skip("Ray not available, skipping test")


def test_get_executor_class_with_ray_not_initialized():
    """Test that ProcessExecutor is returned when Ray is available but not initialized."""
    # Check if Ray is available
    ray_available = importlib.util.find_spec("ray") is not None

    if ray_available:
        import ray

        # Create mock for Ray with no configuration
        ray_mock = MagicMock()
        ray_mock.is_initialized.return_value = False

        # Mock config with no Ray configuration
        config_mock = MagicMock()
        config_mock.ray.address = None
        config_mock.ray.num_gpus = None
        config_mock.ray.num_cpus = None

        with patch.dict("sys.modules", {"ray": ray_mock}):
            with patch("plexe.config.config", config_mock):
                executor_class = _get_executor_class()
                assert executor_class == ProcessExecutor
    else:
        pytest.skip("Ray not available, skipping test")


def test_get_executor_class_with_ray_config_fallback():
    """Test that RayExecutor is returned from config if Ray is available but not initialized."""
    # Check if Ray is available
    ray_available = importlib.util.find_spec("ray") is not None

    if ray_available:
        import ray

        # Create mock for Ray with no initialization
        ray_mock = MagicMock()
        ray_mock.is_initialized.return_value = False

        # Mock config with Ray configuration
        config_mock = MagicMock()
        config_mock.ray.address = "ray://localhost:10001"

        with patch.dict("sys.modules", {"ray": ray_mock}):
            with patch("plexe.config.config", config_mock):
                executor_class = _get_executor_class()
                from plexe.internal.models.execution.ray_executor import RayExecutor

                assert executor_class == RayExecutor
    else:
        pytest.skip("Ray not available, skipping test")
