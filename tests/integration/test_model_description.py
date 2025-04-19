"""Integration test for model description functionality in plexe.

This test covers:
1. Creating a simple model for iris flower classification
2. Building the model with synthetic data
3. Generating model descriptions in different formats (dict, json, text, markdown)
4. Verifying the content of the model descriptions
"""

import json
import os
import pytest
from pathlib import Path
from pydantic import create_model

import pandas as pd
import numpy as np
import plexe
from tests.utils.utils import cleanup_files


@pytest.fixture
def iris_data():
    """Generate synthetic iris data for testing."""
    # Create a simple dataset similar to the iris dataset
    np.random.seed(42)
    n_samples = 30

    data = {
        "sepal_length": np.random.uniform(4.5, 7.5, n_samples),
        "sepal_width": np.random.uniform(2.0, 4.5, n_samples),
        "petal_length": np.random.uniform(1.0, 6.5, n_samples),
        "petal_width": np.random.uniform(0.1, 2.5, n_samples),
        "species": ["setosa"] * n_samples,
    }

    # Generate target based on petal length and width
    # Simplistic rule: if petal_length > 3.0, it's likely virginica or versicolor

    for i in range(n_samples):
        if data["petal_length"][i] < 2.0:
            data["species"][i] = "setosa"
        elif data["petal_length"][i] < 5.0:
            data["species"][i] = "versicolor"
        else:
            data["species"][i] = "virginica"

    return pd.DataFrame(data)


@pytest.fixture
def iris_input_schema():
    """Define the input schema for iris classification."""
    return create_model(
        "IrisInput",
        **{
            "sepal_length": float,
            "sepal_width": float,
            "petal_length": float,
            "petal_width": float,
        },
    )


@pytest.fixture
def iris_output_schema():
    """Define the output schema for iris classification."""
    return create_model("IrisOutput", **{"species": str})


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


def verify_description_format(description, format_type):
    """Verify that a description has the expected format and content."""
    if format_type == "dict":
        assert isinstance(description, dict)
        assert "id" in description
        assert "intent" in description
        assert "schemas" in description
        assert "implementation" in description
        assert "performance" in description
        assert "code" in description
    elif format_type == "json":
        assert isinstance(description, str)
        # Try to parse the JSON string
        try:
            json_dict = json.loads(description)
            assert isinstance(json_dict, dict)
            assert "id" in json_dict
            assert "intent" in json_dict
        except json.JSONDecodeError:
            pytest.fail("Description is not valid JSON")
    elif format_type == "text":
        assert isinstance(description, str)
        assert "Model:" in description
        assert "Intent:" in description
        assert "Input Schema:" in description
        assert "Output Schema:" in description
    elif format_type == "markdown":
        assert isinstance(description, str)
        assert "# Model:" in description
        assert "**Intent:**" in description
        assert "## Input Schema" in description
        assert "## Output Schema" in description


def test_model_description(iris_data, iris_input_schema, iris_output_schema):
    """Test model description generation in various formats and content verification."""

    # Create a model for iris species classification
    model = plexe.Model(
        intent="Classify iris flowers into species based on their sepal and petal measurements",
        input_schema=iris_input_schema,
        output_schema=iris_output_schema,
    )

    # Build the model with minimal iterations for faster testing
    model.build(
        datasets=[iris_data],
        provider="openai/gpt-4o",
        max_iterations=2,  # Minimum iterations for quick testing
        timeout=180,  # 3 minute timeout
        run_timeout=150,
    )

    # Test that the model is in the ready state
    assert model.state.value == "ready", "Model should be in ready state after building"

    # PART 1: Test description object and its format methods

    # Get the model description object
    desc = model.describe()

    # Test the object has the expected attributes and methods
    assert hasattr(desc, "id")
    assert hasattr(desc, "intent")
    assert hasattr(desc, "schemas")
    assert hasattr(desc, "to_dict")
    assert hasattr(desc, "to_json")
    assert hasattr(desc, "as_text")
    assert hasattr(desc, "as_markdown")

    # Test dictionary format
    dict_desc = desc.to_dict()
    verify_description_format(dict_desc, "dict")

    # Test JSON format
    json_desc = desc.to_json()
    verify_description_format(json_desc, "json")

    # Test text format
    text_desc = desc.as_text()
    verify_description_format(text_desc, "text")

    # Test markdown format
    md_desc = desc.as_markdown()
    verify_description_format(md_desc, "markdown")

    # Ensure output is always visible, even when tests pass
    print("\n\n=== MODEL DESCRIPTION IN JSON FORMAT ===\n")
    print(json_desc)

    print("\n\n=== MODEL DESCRIPTION IN TEXT FORMAT ===\n")
    print(text_desc)

    print("\n\n=== MODEL DESCRIPTION IN MARKDOWN FORMAT ===\n")
    print(md_desc)

    # PART 2: Verify description content details

    # Verify basic content
    assert "id" in dict_desc
    assert dict_desc["intent"] == "Classify iris flowers into species based on their sepal and petal measurements"

    # Verify schema information
    assert "schemas" in dict_desc
    assert "input" in dict_desc["schemas"]
    assert "output" in dict_desc["schemas"]

    # Verify input schema has the expected fields
    input_schema = dict_desc["schemas"]["input"]
    for field in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        assert field in input_schema

    # Verify output schema has species field
    output_schema = dict_desc["schemas"]["output"]
    assert any(field.lower() == "species" for field in output_schema.keys())

    # Verify implementation info
    assert "implementation" in dict_desc
    assert "artifacts" in dict_desc["implementation"]

    # Verify code sections
    assert "code" in dict_desc
    assert "prediction" in dict_desc["code"]

    # Verify performance metrics
    assert "performance" in dict_desc
    assert "metrics" in dict_desc["performance"]
