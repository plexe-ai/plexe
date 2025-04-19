"""Integration test for recommendation models using plexe.

This test covers:
1. Creating a recommendation model for product cross-selling
2. Building the model with synthetic data
3. Making predictions with the model
4. Validating the returned recommendations
"""

import os
import pytest
from pathlib import Path
from pydantic import create_model
import plexe
from tests.utils.utils import (
    generate_product_recommendation_data,
    verify_prediction,
    cleanup_files,
    verify_model_description,
)


@pytest.fixture
def product_data():
    """Generate synthetic product recommendation data for testing."""
    return generate_product_recommendation_data(n_samples=60)  # Gives ~20 orders with ~3 items each


@pytest.fixture
def recommendation_input_schema():
    """Define the input schema for product recommendation."""
    return create_model("ProductInput", **{"style": str})


@pytest.fixture
def recommendation_output_schema():
    """Define the output schema for product recommendation."""
    return create_model("ProductOutput", **{"recommended_styles": list})


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


def test_product_recommendation(product_data, recommendation_input_schema, recommendation_output_schema):
    """Test recommendation model for suggesting related products."""
    # Create a model for product recommendations
    model = plexe.Model(
        intent="""
        Given a product style code, recommend up to 3 other product styles that are frequently 
        purchased together with it based on transaction history. Use the order_id to identify 
        products purchased in the same transaction.
        """,
        input_schema=recommendation_input_schema,
        output_schema=recommendation_output_schema,
    )

    # Build the model with minimal iterations for faster testing
    model.build(
        datasets=[product_data],
        provider="openai/gpt-4o",
        max_iterations=3,  # Minimum iterations for reliable model generation
        timeout=300,  # 5 minute timeout
        run_timeout=150,
    )

    # Get a sample style to test with (first style in the dataset)
    test_style = product_data["style"].iloc[0]

    # Test a sample prediction
    test_input = {"style": test_style}
    prediction = model.predict(test_input)

    # Verify the prediction
    verify_prediction(prediction, recommendation_output_schema)

    # Check that the recommendations are a list
    assert isinstance(prediction["recommended_styles"], list), "Recommendations should be a list"

    # Check that we have at most 3 recommendations
    assert len(prediction["recommended_styles"]) <= 3, "Should have at most 3 recommendations"

    # Check that the recommended items are not the input item
    for style in prediction["recommended_styles"]:
        assert style != test_style, "Recommendations should not include the input item"

    # Verify model description
    description = model.describe()
    verify_model_description(description)

    # Test model saving
    model_path = plexe.save_model(model, "recommendation_model.tar.gz")
    assert Path(model_path).exists(), f"Model file {model_path} not created"

    # Test model loading
    loaded_model = plexe.load_model(model_path)
    loaded_prediction = loaded_model.predict(test_input)

    # Verify the loaded model's prediction
    verify_prediction(loaded_prediction, recommendation_output_schema)
    assert isinstance(loaded_prediction["recommended_styles"], list), "Recommendations should be a list"
