"""Integration test for regression models using plexe.

This test covers:
1. Creating a regression model for house price prediction
2. Building the model with synthetic data
3. Making predictions with the model
4. Testing schema inference
"""

import os
import pytest
from pathlib import Path
from pydantic import create_model
import plexe
from tests.utils.utils import generate_house_prices_data, verify_prediction, cleanup_files, verify_model_description


@pytest.fixture
def house_data():
    """Generate synthetic house price data for testing."""
    return generate_house_prices_data(n_samples=30)


@pytest.fixture
def house_input_schema():
    """Define the input schema for house price prediction."""
    return create_model(
        "HousePriceInput",
        **{
            "area": int,
            "bedrooms": int,
            "bathrooms": int,
            "stories": int,
            "garage": int,
            "garden": int,
            "fenced": int,
            "age": int,
        },
    )


@pytest.fixture
def house_output_schema():
    """Define the output schema for house price prediction."""
    return create_model("HousePriceOutput", **{"price": float})


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


def test_house_price_regression(house_data, house_input_schema, house_output_schema):
    """Test regression for house price prediction."""
    # Create a model for house price prediction
    model = plexe.Model(
        intent="Predict the price of a house based on its features",
        input_schema=house_input_schema,
        output_schema=house_output_schema,
    )

    # Build the model with minimal data and iterations for faster testing
    model.build(
        datasets=[house_data],
        provider="openai/gpt-4o",
        max_iterations=3,  # Minimum iterations for reliable model generation
        timeout=300,  # 5 minute timeout
        run_timeout=150,
    )

    # Test a sample prediction
    test_input = {
        "area": 2500,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 2,
        "garage": 1,
        "garden": 1,
        "fenced": 1,
        "age": 5,
    }
    prediction = model.predict(test_input)

    # Verify the prediction
    verify_prediction(prediction, house_output_schema)
    assert isinstance(prediction["price"], (int, float)), "House price should be numeric"
    assert prediction["price"] > 0, "House price should be positive"

    # Verify model description
    description = model.describe()
    verify_model_description(description)

    # Test model saving and loading
    model_path = plexe.save_model(model, "house_price_model.tar.gz")
    loaded_model = plexe.load_model(model_path)
    loaded_prediction = loaded_model.predict(test_input)

    # Verify the loaded model's prediction
    verify_prediction(loaded_prediction, house_output_schema)
    assert isinstance(loaded_prediction["price"], (int, float)), "House price should be numeric"
    assert loaded_prediction["price"] > 0, "House price should be positive"
    assert loaded_prediction == prediction, "Loaded model prediction should match original model prediction"


def test_house_price_schema_inference(house_data):
    """Test regression with schema inference."""
    # Create a model with only intent
    model = plexe.Model(
        intent="""
        Predict the price of a house in thousands of dollars based on features like 
        area, number of bedrooms and bathrooms, etc.
        """,
    )

    # Build the model with minimal iterations for faster testing
    model.build(
        datasets=[house_data],
        provider="openai/gpt-4o",
        max_iterations=3,  # Minimum iterations for reliable model generation
        timeout=300,  # 5 minute timeout
        run_timeout=150,
    )

    # Test a sample prediction
    test_input = {
        "area": 2500,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 2,
        "garage": 1,
        "garden": 1,
        "fenced": 1,
        "age": 5,
    }

    prediction = model.predict(test_input)

    # Verify the prediction is a dictionary with at least one key
    assert isinstance(prediction, dict), "Prediction should be a dictionary"
    assert len(prediction) > 0, "Prediction should not be empty"

    # The first value should be a numeric price
    price_value = list(prediction.values())[0]
    assert isinstance(price_value, (int, float)), f"Price should be numeric, got {type(price_value)}"
    assert price_value > 0, "House price should be positive"

    # Verify model description
    description = model.describe()
    verify_model_description(description)
