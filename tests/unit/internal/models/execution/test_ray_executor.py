from pathlib import Path
import pytest
import pandas as pd
import ray

from plexe.internal.models.execution.ray_executor import RayExecutor
from plexe.internal.common.datasets.tabular import TabularDataset


@pytest.fixture
def sample_code():
    return """
import pandas as pd
import time

# Simple training script that loads data, does minimal processing and outputs a result
print("Loading training dataset...")
train_df = pd.read_parquet("dataset_0_train.parquet")
print(f"Train dataset shape: {train_df.shape}")

# Simple model training
print("Training model...")
result = {"accuracy": 0.85}

# Create model directory and save a dummy file
import os
from pathlib import Path
model_dir = Path("model_files")
model_dir.mkdir(exist_ok=True)
with open(model_dir / "model.txt", "w") as f:
    f.write("dummy model file")

print(f"Final performance: {result['accuracy']}")
"""


@pytest.fixture
def sample_data():
    # Create a simple dataframe
    df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5], "target": [0, 1, 0, 1, 0]})
    return TabularDataset(df)


@pytest.fixture
def cleanup_ray():
    # Setup
    yield
    # Teardown - ensure Ray is shut down

    if ray.is_initialized():
        ray.shutdown()


def test_ray_executor(sample_code, sample_data, cleanup_ray, tmpdir):
    """Test that the RayExecutor correctly executes code and captures results."""
    # Setup
    working_dir = tmpdir
    datasets = {"dataset_0_train": sample_data}

    # Create and run the executor
    executor = RayExecutor(
        execution_id="test-execution", code=sample_code, working_dir=working_dir, datasets=datasets, timeout=30
    )
    result = executor.run()

    # Validate execution results
    assert result.exec_time > 0
    assert "Train dataset shape: (5, 3)" in result.term_out[0]
    assert result.performance == 0.85
    assert len(result.model_artifacts) > 0

    # Verify model artifacts
    model_files_path = None
    for artifact in result.model_artifacts:
        if "model_files" in str(artifact):
            model_files_path = artifact
            break

    assert model_files_path is not None
    assert Path(model_files_path, "model.txt").exists()


def test_ray_executor_timeout(sample_data, cleanup_ray, tmpdir):
    """Test that the RayExecutor correctly handles timeouts."""
    # Setup code with intentional infinite loop
    timeout_code = """
import time
print("Starting code that will timeout...")
time.sleep(100)  # This should trigger the timeout
"""

    working_dir = tmpdir
    datasets = {"dataset_0_train": sample_data}

    # Create and run the executor with a short timeout
    executor = RayExecutor(
        execution_id="test-timeout",
        code=timeout_code,
        working_dir=working_dir,
        datasets=datasets,
        timeout=1,  # Very short timeout to trigger
    )
    result = executor.run()

    # Validate timeout was triggered
    assert isinstance(result.exception, TimeoutError)
    assert "timeout" in str(result.exception).lower()
