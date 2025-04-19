"""Integration test for multiclass classification models using plexe.

This test covers:
1. Creating a multiclass classification model for sentiment analysis
2. Building the model with synthetic data
3. Making predictions with the model
4. Testing dataset generation capabilities
"""

import os
import pytest
from pathlib import Path
from pydantic import create_model
import plexe
from tests.utils.utils import verify_prediction, cleanup_files, verify_model_description, generate_sentiment_data


@pytest.fixture
def sentiment_data():
    """Generate synthetic sentiment data for testing."""
    return generate_sentiment_data(n_samples=30)


@pytest.fixture
def sentiment_input_schema():
    """Define the input schema for sentiment analysis."""
    return create_model("SentimentInput", **{"text": str})


@pytest.fixture
def sentiment_output_schema():
    """Define the output schema for sentiment analysis."""
    return create_model("SentimentOutput", **{"sentiment": str})


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


def test_multiclass_classification(sentiment_data, sentiment_input_schema, sentiment_output_schema):
    """Test multiclass classification for sentiment analysis."""
    # Create a model for sentiment analysis
    model = plexe.Model(
        intent="Classify text sentiment into positive, negative, or neutral categories",
        input_schema=sentiment_input_schema,
        output_schema=sentiment_output_schema,
    )

    # Build the model with minimal data and iterations for faster testing
    model.build(
        datasets=[sentiment_data],
        provider="openai/gpt-4o",
        max_iterations=3,  # Minimum iterations for reliable model generation
        timeout=300,  # 5 minute timeout
        run_timeout=150,
    )

    # Test sample predictions
    test_inputs = [
        {"text": "This product exceeded my expectations! The quality is amazing."},
        {"text": "Very disappointed with this purchase. Would not recommend."},
        {"text": "The product is okay, nothing special but works as expected."},
    ]

    for test_input in test_inputs:
        prediction = model.predict(test_input)

        # Verify the prediction
        verify_prediction(prediction, sentiment_output_schema)
        assert prediction["sentiment"] in [
            "positive",
            "negative",
            "neutral",
        ], f"Prediction should be one of ['positive', 'negative', 'neutral'], got {prediction['sentiment']}"

    # Verify model description
    description = model.describe()
    verify_model_description(description)


def test_dataset_generation_and_multiclass(sentiment_input_schema, sentiment_output_schema):
    """Test multiclass classification with dataset generation."""
    # Create a dataset generator for sentiment data
    dataset = plexe.DatasetGenerator(
        description="A dataset of text reviews with positive, negative, and neutral sentiment labels",
        provider="openai/gpt-4o",
        schema=create_model("SentimentData", **{"text": str, "sentiment": str}),
    )

    # Generate a small synthetic dataset
    dataset.generate(15)  # Generate 15 samples

    # Create a model using the generated dataset
    model = plexe.Model(
        intent="Classify text sentiment into positive, negative, or neutral categories",
        input_schema=sentiment_input_schema,
        output_schema=sentiment_output_schema,
    )

    # Build the model with minimal iterations for faster testing
    model.build(
        datasets=[dataset],
        provider="openai/gpt-4o",
        max_iterations=2,  # Minimum iterations for reliable model generation
        timeout=300,  # 5 minute timeout
        run_timeout=150,
    )

    # Test a sample prediction
    test_input = {"text": "This product is quite good, but has some minor flaws"}

    prediction = model.predict(test_input)

    # Verify the prediction
    verify_prediction(prediction, sentiment_output_schema)
    assert prediction["sentiment"] in [
        "positive",
        "negative",
        "neutral",
    ], f"Prediction should be one of ['positive', 'negative', 'neutral'], got {prediction['sentiment']}"
