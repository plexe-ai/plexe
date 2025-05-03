"""Integration test for customer churn prediction models using plexe.

This test covers:
1. Creating a model for predicting customer churn
2. Building the model with synthetic data
3. Making predictions with the model
4. Validating predictions with probability output
5. Using schema inference
"""

import os
from pathlib import Path

import pytest
from pydantic import create_model, Field

import plexe
from tests.utils.utils import generate_customer_churn_data, verify_prediction, cleanup_files, verify_model_description


@pytest.fixture
def churn_data():
    """Generate synthetic customer churn data for testing."""
    return generate_customer_churn_data(n_samples=30)


@pytest.fixture
def churn_input_schema():
    """Define the input schema for churn prediction with field validations."""
    return create_model(
        "ChurnInput",
        **{
            "tenure": (int, Field(ge=0, description="Number of months the customer has been with the company")),
            "monthly_charges": (float, Field(ge=0, description="Monthly charges in dollars")),
            "total_charges": (float, Field(ge=0, description="Total charges in dollars")),
            "contract_type": (int, Field(ge=0, le=2, description="0=month-to-month, 1=one year, 2=two year")),
            "payment_method": (
                int,
                Field(ge=0, le=3, description="0=electronic check, 1=mailed check, 2=bank transfer, 3=credit card"),
            ),
            "tech_support": (int, Field(ge=0, le=1, description="0=no, 1=yes")),
            "online_backup": (int, Field(ge=0, le=1, description="0=no, 1=yes")),
            "online_security": (int, Field(ge=0, le=1, description="0=no, 1=yes")),
        },
    )


@pytest.fixture
def churn_output_schema():
    """Define the output schema for churn prediction with probability."""
    return create_model(
        "ChurnOutput",
        **{
            "churn_probability": (float, Field(ge=0, le=1, description="Probability of customer churning (0-1)")),
            "churn": (int, Field(ge=0, le=1, description="Binary churn prediction (0=no, 1=yes)")),
        },
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


def test_customer_churn_prediction(churn_data, churn_input_schema, churn_output_schema):
    """Test customer churn prediction with probability output."""
    # Create a model for churn prediction
    model = plexe.Model(
        intent="""
        Predict the probability that a customer will churn (leave the company) based on their 
        service usage and contract details. Return both the probability of churning (0-1) and 
        a binary prediction (0=will not churn, 1=will churn) using a threshold of 0.5.
        """,
        input_schema=churn_input_schema,
        output_schema=churn_output_schema,
    )

    # Build the model with minimal iterations for faster testing
    model.build(
        datasets=[churn_data],
        provider="openai/gpt-4o",
        max_iterations=4,  # Minimum iterations for reliable model generation
        timeout=300,  # 5 minute timeout
        run_timeout=150,
    )

    # Test sample predictions
    test_inputs = [
        # High risk customer (month-to-month contract, high monthly charge, low tenure)
        {
            "tenure": 3,
            "monthly_charges": 120.00,
            "total_charges": 360.00,
            "contract_type": 0,
            "payment_method": 0,
            "tech_support": 0,
            "online_backup": 0,
            "online_security": 0,
        },
        # Low risk customer (two-year contract, moderate monthly charge, high tenure)
        {
            "tenure": 60,
            "monthly_charges": 90.00,
            "total_charges": 5400.00,
            "contract_type": 2,
            "payment_method": 3,
            "tech_support": 1,
            "online_backup": 1,
            "online_security": 1,
        },
    ]

    for test_input in test_inputs:
        prediction = model.predict(test_input)

        # Verify the prediction structure
        verify_prediction(prediction, churn_output_schema)

        # Verify probability constraints
        assert 0 <= prediction["churn_probability"] <= 1, "Churn probability should be between 0 and 1"

        # Verify binary prediction
        assert prediction["churn"] in [0, 1], "Binary churn prediction should be 0 or 1"

        # Verify consistency between probability and binary prediction
        if prediction["churn_probability"] > 0.5:
            assert prediction["churn"] == 1, "Binary prediction should be 1 if probability > 0.5"
        else:
            assert prediction["churn"] == 0, "Binary prediction should be 0 if probability <= 0.5"

    # Verify model description
    description = model.describe()
    verify_model_description(description)

    # Test model saving and loading
    model_path = plexe.save_model(model, "churn_model.tar.gz")
    loaded_model = plexe.load_model(model_path)

    # Verify loaded model predictions
    for test_input in test_inputs:
        loaded_prediction = loaded_model.predict(test_input)
        verify_prediction(loaded_prediction, churn_output_schema)
