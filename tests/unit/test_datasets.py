from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from plexe import DatasetGenerator


@pytest.fixture
def sample_schema():
    """Test schema for house price prediction"""
    return {
        "input_schema": {"square_feet": float, "bedrooms": int, "location": str},
        "output_schema": {"price": float},
    }


@pytest.fixture
def mock_generated_data():
    """Mock data generation output"""
    return pd.DataFrame(
        {
            "square_feet": np.random.uniform(1000, 3000, 50),
            "bedrooms": np.random.randint(2, 6, 50),
            "location": np.random.choice(["suburban", "urban", "rural"], 50),
            "price": np.random.uniform(200000, 600000, 50),
        }
    )


class TestDataGeneration:
    """Test suite for data generation with comprehensive mocking"""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Setup all required mocks for the test class"""
        # Mock the data generation function
        self.mock_generate_data = patch("plexe.datasets.DataGenerator.generate", return_value=pd.DataFrame()).start()

        yield

        # Stop all mocks after the test
        patch.stopall()

    def test_basic_generation(self, sample_schema, mock_generated_data):
        """Test basic data generation"""
        self.mock_generate_data.return_value = mock_generated_data

        dataset = DatasetGenerator(
            description="House features and prices, each row is a house",
            schema={**sample_schema["input_schema"], **sample_schema["output_schema"]},
            provider="openai/gpt-4o",
        )
        dataset.generate(50)

        # Verify generate_data was called with correct parameters
        self.mock_generate_data.assert_called_once()
        call_args = self.mock_generate_data.call_args[0]
        assert isinstance(call_args[0], int)
        assert call_args[0] == 50

        # Verify generated data was added to the dataset
        assert dataset._data is not None
        assert len(dataset._data) == 50

    def test_data_augmentation(self, sample_schema, mock_generated_data):
        """Test data augmentation with existing dataset"""
        self.mock_generate_data.return_value = mock_generated_data

        existing_data = pd.DataFrame(
            {
                "square_feet": [1000, 1500, 2000],
                "bedrooms": [2, 3, 4],
                "location": ["A", "B", "C"],
                "price": [200000, 300000, 400000],
            }
        )

        dataset = DatasetGenerator(
            description="House features and prices, each row is a house",
            schema={**sample_schema["input_schema"], **sample_schema["output_schema"]},
            provider="openai/gpt-4o",
            data=existing_data.copy(),
        )
        dataset.generate(50)

        # Verify generate_data was called with correct parameters
        self.mock_generate_data.assert_called_once()
        call_args = self.mock_generate_data.call_args[0]
        assert isinstance(call_args[0], int)
        assert call_args[0] == 50

        # Verify final dataset includes both original and synthetic data
        assert len(dataset._data) == len(existing_data) + 50
