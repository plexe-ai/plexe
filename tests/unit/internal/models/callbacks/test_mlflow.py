"""
Unit tests for the MLFlowCallback class.

These tests validate the functionality of the MLFlowCallback, which is responsible
for logging model building metrics, parameters, and artifacts to MLFlow.
"""

from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel

from plexe.callbacks import BuildStateInfo
from plexe.internal.models.callbacks.mlflow import MLFlowCallback
from plexe.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from plexe.internal.models.entities.node import Node


@pytest.fixture
def setup_env():
    """Set up common test environment."""
    # Create metric
    metric = Metric(name="accuracy", value=0.95, comparator=MetricComparator(ComparisonMethod.HIGHER_IS_BETTER))

    # Create node with the metric
    node = Node(
        solution_plan="Train a random forest model",
        performance=metric,
        execution_time=10.5,
        model_artifacts=["/path/to/artifact.pkl"],
    )

    # Create input/output schemas
    class InputSchema(BaseModel):
        feature1: float
        feature2: str

    class OutputSchema(BaseModel):
        prediction: float

    # Create a mock dataset
    mock_dataset = MagicMock()
    mock_dataset.to_pandas.return_value = MagicMock()

    # Create build info for different stages
    build_info = BuildStateInfo(
        intent="Predict house prices",
        provider="openai/gpt-4o-mini",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        run_timeout=300,
        max_iterations=10,
        timeout=3600,
        datasets={"train": mock_dataset},
    )

    # Create iteration info with node
    iteration_info = BuildStateInfo(
        intent="Predict house prices",
        provider="openai/gpt-4o-mini",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        run_timeout=300,
        max_iterations=10,
        timeout=3600,
        iteration=1,
        node=node,
        datasets={"train": mock_dataset},
    )

    return {
        "metric": metric,
        "node": node,
        "input_schema": InputSchema,
        "output_schema": OutputSchema,
        "mock_dataset": mock_dataset,
        "build_info": build_info,
        "iteration_info": iteration_info,
    }


def test_callback_initialization():
    """Test that the MLFlowCallback can be initialized properly."""
    with patch("mlflow.set_tracking_uri") as mock_set_tracking_uri:
        with patch("mlflow.active_run", return_value=None):  # No active run
            callback = MLFlowCallback(tracking_uri="http://localhost:5000", experiment_name="test-experiment")

            # Verify tracking URI was set
            mock_set_tracking_uri.assert_called_once_with("http://localhost:5000")

            # Verify default values
            assert callback.experiment_name == "test-experiment"
            assert callback.experiment_id is None


@patch("mlflow.set_tracking_uri")
@patch("mlflow.get_experiment_by_name")
@patch("mlflow.create_experiment")
def test_build_start(mock_create_experiment, mock_get_experiment, _, setup_env):
    """Test on_build_start callback."""
    # Set up mocks
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "test-experiment-id"
    mock_get_experiment.return_value = mock_experiment

    # Initialize callback with active_run patched
    with patch("mlflow.active_run", return_value=None):
        callback = MLFlowCallback(tracking_uri="http://localhost:5000", experiment_name="test-experiment")

    # Call on_build_start
    callback.on_build_start(setup_env["build_info"])

    # Verify experiment was retrieved
    mock_get_experiment.assert_called_once_with("test-experiment")

    # If experiment exists, we should not create a new one
    mock_create_experiment.assert_not_called()

    # Experiment ID should be set
    assert callback.experiment_id == "test-experiment-id"


@patch("mlflow.set_tracking_uri")
@patch("mlflow.get_experiment_by_name")
@patch("mlflow.create_experiment")
def test_build_start_new_experiment(mock_create_experiment, mock_get_experiment, _, setup_env):
    """Test on_build_start with a new experiment."""
    # Set up mocks for a new experiment
    mock_get_experiment.return_value = None
    mock_create_experiment.return_value = "new-experiment-id"

    # Initialize callback with active_run patched
    with patch("mlflow.active_run", return_value=None):
        callback = MLFlowCallback(tracking_uri="http://localhost:5000", experiment_name="new-experiment")

    # Call on_build_start
    callback.on_build_start(setup_env["build_info"])

    # Verify experiment was retrieved and created
    mock_get_experiment.assert_called_once_with("new-experiment")
    mock_create_experiment.assert_called_once_with("new-experiment")

    # Experiment ID should be set to the new ID
    assert callback.experiment_id == "new-experiment-id"


def test_build_end(setup_env):
    """Test on_build_end callback."""
    # Initialize callback with active_run patched (no existing run)
    with patch("mlflow.active_run", return_value=None):
        callback = MLFlowCallback(tracking_uri="http://localhost:5000", experiment_name="test-experiment")

    # Now patch active_run to return a run when the on_build_end method is called
    with patch("mlflow.active_run", return_value=MagicMock()):
        # And also patch end_run for verification
        with patch("mlflow.end_run") as mock_end_run:
            # Call on_build_end
            callback.on_build_end(setup_env["build_info"])

            # Verify end_run was called once
            mock_end_run.assert_called_once()


def test_log_metric(setup_env):
    """Test _log_metric helper method."""
    # Initialize callback
    with patch("mlflow.active_run", return_value=None):
        callback = MLFlowCallback(tracking_uri="http://localhost:5000", experiment_name="test-experiment")

    # Mock active_run to True when _log_metric is called
    with patch("mlflow.active_run", return_value=MagicMock()):
        # Mock the log_metric method
        with patch("mlflow.log_metric") as mock_log_metric:
            # Call _log_metric
            callback._log_metric(setup_env["metric"], prefix="test_", step=1)

            # Verify metric was logged
            mock_log_metric.assert_called_once()

            # Just check the first arg (metric name) and validate that other args exist
            args = mock_log_metric.call_args[0]
            assert "accuracy" in args[0]  # Name contains "accuracy"
            assert args[1] == 0.95  # Value is correct

            # Check that step was passed as a kwarg
            kwargs = mock_log_metric.call_args[1]
            assert kwargs.get("step") == 1


if __name__ == "__main__":
    pytest.main()
