"""Integration test for schema validation in plexe.

This test covers:
1. Creating models with fields that have validation requirements
2. Testing that invalid inputs fail properly 
3. Testing that valid inputs pass properly
"""

import os
import pytest
from pathlib import Path
from pydantic import create_model, Field
import plexe
from tests.utils.utils import generate_house_prices_data, verify_prediction, cleanup_files


@pytest.fixture
def house_data():
    """Generate synthetic house price data for testing."""
    return generate_house_prices_data(n_samples=30)


@pytest.fixture
def house_data_copy(house_data):
    """Create a copy of the house data to avoid mutation issues."""
    return house_data.copy()


@pytest.fixture
def validated_input_schema():
    """Define the input schema for house price prediction with validation."""
    return create_model(
        "ValidatedHouseInput",
        **{
            "area": (int, Field(description="Square feet (500-10000)")),
            "bedrooms": (int, Field(description="Number of bedrooms (1-10)")),
            "bathrooms": (int, Field(description="Number of bathrooms (1-7)")),
            "stories": (int, Field(description="Number of stories (1-4)")),
            "garage": (int, Field(description="Garage capacity in cars (0-3)")),
            "garden": (int, Field(description="Has garden (0=no, 1=yes)")),
            "fenced": (int, Field(description="Has fenced yard (0=no, 1=yes)")),
            "age": (int, Field(description="Age of house in years (0-100)")),
        },
    )


@pytest.fixture
def validated_output_schema():
    """Define the output schema for house price prediction with validation."""
    return create_model(
        "ValidatedHouseOutput",
        **{"price": (float, Field(ge=50, le=5000, description="House price in thousands of dollars (50-5000)"))},
    )


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


def test_input_validation(house_data_copy, validated_input_schema, validated_output_schema):
    """Test validation of input schema."""
    # Create a model with validated input schema
    model = plexe.Model(
        intent="Predict the price of a house based on its features, with input validation",
        input_schema=validated_input_schema,
        output_schema=validated_output_schema,
    )

    # Build the model
    model.build(
        datasets=[house_data_copy],
        provider="openai/gpt-4o",
        max_iterations=3,  # Minimum iterations for reliable model generation
        timeout=300,  # 5 minute timeout
        run_timeout=150,
    )

    # Valid input should work
    valid_input = {
        "area": 2500,
        "bedrooms": 4,
        "bathrooms": 2,
        "stories": 2,
        "garage": 1,
        "garden": 1,
        "fenced": 1,
        "age": 5,
    }
    prediction = model.predict(valid_input)
    verify_prediction(prediction, validated_output_schema)

    # Invalid inputs should raise validation errors
    invalid_inputs = [
        {
            "area": 300,
            "stories": 2,
            "garage": 1,
            "garden": 1,
            "fenced": 1,
            "age": 5,
        },  # Missing features
        {
            "area": "not-a-number",
            "bedrooms": None,
            "bathrooms": "two",
            "stories": False,
            "garage": 1,
            "garden": 1,
            "fenced": 1,
            "age": 5,
        },  # Invalid types
    ]

    for invalid_input in invalid_inputs:
        with pytest.raises(Exception):
            # This should raise a validation error when the model validates the input
            # against the schema before prediction
            model.predict(invalid_input, validate_input=True)


def test_output_validation(house_data_copy, validated_input_schema):
    """Test validation of output schema."""
    # Create an output schema with strict range validation
    strict_output_schema = create_model(
        "StrictHouseOutput",
        **{"price": (float, Field(ge=500, le=600, description="House price in thousands of dollars (500-600)"))},
    )

    # Create a model with standard input but strictly bounded output
    model = plexe.Model(
        intent="Predict the price of a house based on its features, ensuring predictions are between 500-600k",
        input_schema=validated_input_schema,
        output_schema=strict_output_schema,
    )

    # Build the model
    model.build(
        datasets=[house_data_copy],
        provider="openai/gpt-4o",
        max_iterations=3,  # Minimum iterations for reliable model generation
        timeout=300,
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

    # Verify the prediction meets the strict output schema
    verify_prediction(prediction, strict_output_schema)
