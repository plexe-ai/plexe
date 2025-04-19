"""Integration test for time series forecasting models using plexe.

This test covers:
1. Creating a time series forecasting model for sales prediction
2. Building the model with synthetic time series data
3. Making predictions with the model for future time periods
"""

import os
import pytest
from pathlib import Path
from pydantic import create_model
import plexe
from tests.utils.utils import generate_time_series_data, verify_prediction, cleanup_files, verify_model_description


@pytest.fixture
def sales_data():
    """Generate synthetic time series data for testing."""
    return generate_time_series_data(n_samples=60)


@pytest.fixture
def sales_data_copy(sales_data):
    """Create a copy of the sales data to avoid mutation issues."""
    return sales_data.copy()


@pytest.fixture
def sales_input_schema():
    """Define the input schema for sales forecasting."""
    return create_model(
        "SalesInput",
        **{
            "date": str,
            "promo": int,
            "holiday": int,
            "day_of_week": int,
        },
    )


@pytest.fixture
def sales_output_schema():
    """Define the output schema for sales forecasting."""
    return create_model("SalesOutput", **{"sales": float})


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


def test_time_series_forecasting(sales_data_copy, sales_input_schema, sales_output_schema):
    """Test time series forecasting for sales prediction."""
    # Ensure date is in string format for the model input
    sales_data_copy["date"] = sales_data_copy["date"].dt.strftime("%Y-%m-%d")

    # Create a model for sales forecasting
    model = plexe.Model(
        intent="Predict daily sales based on the date, promotions, holidays, and day of the week",
        input_schema=sales_input_schema,
        output_schema=sales_output_schema,
    )

    # Build the model
    model.build(
        datasets=[sales_data_copy],
        provider="openai/gpt-4o",
        max_iterations=4,  # Minimum iterations for reliable model generation
        timeout=400,  # 5 minute timeout
        run_timeout=150,
    )

    # Test prediction for a future date
    future_date = "2023-03-01"  # A date after the training data
    test_input = {
        "date": future_date,
        "promo": 1,
        "holiday": 0,
        "day_of_week": 2,  # Wednesday
    }
    prediction = model.predict(test_input)

    # Verify the prediction
    verify_prediction(prediction, sales_output_schema)
    assert isinstance(prediction["sales"], (int, float)), "Sales should be numeric"
    assert prediction["sales"] > 0, "Sales should be positive"

    # Test another date with different features
    future_date_2 = "2023-03-04"  # Saturday
    test_input_2 = {
        "date": future_date_2,
        "promo": 0,
        "holiday": 1,
        "day_of_week": 5,  # Saturday
    }
    prediction_2 = model.predict(test_input_2)

    # Verify the prediction
    verify_prediction(prediction_2, sales_output_schema)
    assert isinstance(prediction_2["sales"], (int, float)), "Sales should be numeric"
    assert prediction_2["sales"] > 0, "Sales should be positive"

    # Verify model description
    description = model.describe()
    verify_model_description(description)
