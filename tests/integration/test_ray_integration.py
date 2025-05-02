"""
Integration test for Ray-based distributed training.
"""

import pytest
import pandas as pd
import numpy as np
from plexe.models import Model


@pytest.fixture
def sample_dataset():
    """Create a simple synthetic dataset for testing."""
    # Create a sample regression dataset
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = 2 + 3 * X[:, 0] + 0.5 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1

    # Create a DataFrame with feature and target columns
    df = pd.DataFrame(data=np.column_stack([X, y]), columns=[f"feature_{i}" for i in range(5)] + ["target"])
    return df


def test_model_with_ray(sample_dataset):
    """Test building a model with Ray-based distributed execution."""
    # Skip this test if no API key is available
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not available")

    # Initialize Ray explicitly
    import ray

    # Ensure Ray is not already running
    if ray.is_initialized():
        ray.shutdown()

    # Initialize with specific resources - use GPU if available
    from plexe.internal.common.utils.model_utils import is_gpu_available

    # Check if GPU is available
    gpu_available = is_gpu_available()
    print(f"GPU available for testing: {gpu_available}")

    # Initialize Ray with GPU if available
    ray.init(num_cpus=2, num_gpus=1 if gpu_available else 0, ignore_reinit_error=True)

    # Log Ray resources
    resources = ray.cluster_resources()
    print(f"Ray resources: {resources}")

    # Import classes needed for assertions
    from plexe.internal.models.tools.execution import _get_executor_class
    from plexe.internal.models.execution.ray_executor import RayExecutor

    # Verify Ray is initialized
    assert ray.is_initialized(), "Ray should be initialized before the test"

    # Verify the factory correctly detects Ray
    executor_class = _get_executor_class()
    assert executor_class == RayExecutor, "Ray executor should be selected when Ray is initialized"

    try:
        # Create a model for testing (with shorter timeouts for testing)
        model = Model(intent="Predict the target variable given 5 numerical features")

        # Set a shorter timeout for testing
        model.build(
            datasets=[sample_dataset],
            provider="openai/gpt-4o-mini",
            timeout=300,  # 5 minutes max
            run_timeout=60,  # 1 minute per run
        )

        # Check if Ray is still initialized after model build
        if not ray.is_initialized():
            print("Warning: Ray was shut down during model building, trying to reinitialize")
            ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)

        # Test a prediction
        input_data = {f"feature_{i}": 0.5 for i in range(5)}
        prediction = model.predict(input_data)

        # Verify prediction worked
        assert prediction is not None
        assert "target" in prediction

        # Verify model built successfully
        assert model.metric is not None

        # Check if Ray was used (but don't fail the test if not)
        if hasattr(RayExecutor, "_ray_was_used"):
            print(f"Ray executor was used: {RayExecutor._ray_was_used}")

    finally:
        # Print Ray status before shutdown for debugging
        print(f"Ray status before shutdown: initialized={ray.is_initialized()}")

        # Clean up Ray resources
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown completed")
        else:
            print("Ray was already shut down")
