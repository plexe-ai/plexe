"""Integration test for binary classification models using plexe.

This test covers:
1. Creating a binary classification model for heart disease prediction
2. Building the model with synthetic data
3. Making predictions with the model
4. Saving and loading the model
"""

import os
import pytest
from pathlib import Path
from pydantic import create_model
import plexe
from tests.utils.utils import generate_heart_data, verify_prediction, cleanup_files, verify_model_description


@pytest.fixture
def heart_data():
    """Generate synthetic heart disease data for testing."""
    return generate_heart_data(n_samples=30)


@pytest.fixture
def heart_input_schema():
    """Define the input schema for heart disease prediction."""
    return create_model(
        "HeartDiseaseInput",
        **{
            "age": int,
            "gender": int,
            "cp": int,
            "trtbps": int,
            "chol": int,
            "fbs": int,
            "restecg": int,
            "thalachh": int,
            "exng": int,
            "oldpeak": float,
            "slp": int,
            "caa": int,
            "thall": int,
        },
    )


@pytest.fixture
def heart_output_schema():
    """Define the output schema for heart disease prediction."""
    return create_model("HeartDiseaseOutput", **{"output": int})


@pytest.fixture
def model_dir(tmpdir):
    """Create and manage a temporary directory for model files."""
    model_path = Path(tmpdir) / "models"
    model_path.mkdir(exist_ok=True)
    return model_path


@pytest.fixture(autouse=True)
def run_around_tests(model_dir):
    """Set up and tear down for each test."""
    cleanup_files(model_dir)
    os.environ["MODEL_PATH"] = str(model_dir)
    yield
    # Teardown
    cleanup_files(model_dir)


def test_heart_disease_classification(heart_data, heart_input_schema, heart_output_schema):
    """Test binary classification for heart disease prediction."""
    # Create a model for heart disease prediction
    model = plexe.Model(
        intent="Predict whether a patient is likely to have heart disease based on their medical features",
        input_schema=heart_input_schema,
        output_schema=heart_output_schema,
    )

    # Build the model with minimal data and iterations for faster testing
    model.build(
        datasets=[heart_data],
        provider="openai/gpt-4o",
        max_iterations=3,  # Minimum iterations for reliable model generation
        timeout=300,  # 5 minute timeout
        run_timeout=150,
    )

    # Test a sample prediction
    test_input = {
        "age": 61,
        "gender": 1,
        "cp": 3,
        "trtbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalachh": 150,
        "exng": 0,
        "oldpeak": 2.3,
        "slp": 0,
        "caa": 0,
        "thall": 1,
    }
    # Try prediction with a dictionary directly
    prediction = model.predict(test_input)

    # Verify the prediction
    verify_prediction(prediction, heart_output_schema)
    assert prediction["output"] in [0, 1], "Binary classification output should be 0 or 1"

    # Verify model description
    description = model.describe()
    verify_model_description(description)

    # Test model saving
    model_path = plexe.save_model(model, "heart_disease_model.tar.gz")
    assert Path(model_path).exists(), f"Model file {model_path} not created"

    # Test model loading
    loaded_model = plexe.load_model(model_path)
    # Use dictionary for prediction with the loaded model
    loaded_prediction = loaded_model.predict(test_input)

    # Verify the loaded model's prediction
    verify_prediction(loaded_prediction, heart_output_schema)
    assert loaded_prediction["output"] in [0, 1], "Binary classification output should be 0 or 1"
    assert loaded_prediction == prediction, "Loaded model prediction should match original model prediction"
