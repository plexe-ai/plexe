"""Test the executor factory."""

import pytest
from unittest.mock import patch

import importlib

from plexe.tools.execution import _get_executor_class
from plexe.internal.models.execution.process_executor import ProcessExecutor


def test_get_executor_class_non_distributed():
    """Test that ProcessExecutor is returned when distributed=False."""
    executor_class = _get_executor_class(distributed=False)
    assert executor_class == ProcessExecutor


def test_get_executor_class_distributed():
    """Test that RayExecutor is returned when distributed=True and Ray is available."""
    # Check if Ray is available
    ray_available = importlib.util.find_spec("ray") is not None

    if ray_available:
        executor_class = _get_executor_class(distributed=True)
        from plexe.internal.models.execution.ray_executor import RayExecutor

        assert executor_class == RayExecutor
    else:
        pytest.skip("Ray not available, skipping test")


def test_get_executor_class_distributed_ray_not_available():
    """Test that ProcessExecutor is returned as fallback when Ray is not available."""
    # Use a mock to simulate Ray not being available
    with patch(
        "builtins.__import__",
        side_effect=lambda name, *args, **kwargs: (
            ModuleNotFoundError("No module named 'ray'")
            if name == "plexe.internal.models.execution.ray_executor"
            else importlib.import_module(name)
        ),
    ):
        executor_class = _get_executor_class(distributed=True)
        assert executor_class == ProcessExecutor
