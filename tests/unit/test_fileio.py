"""
Unit tests for plexe.fileio module, including backwards compatibility testing.
"""

from pathlib import Path
from typing import Any

import pytest

import plexe.fileio as fileio


def _validate_model(model: Any) -> None:
    """Helper function to validate the loaded model."""
    # Basic validation - model should load successfully
    assert model is not None, "Model should not be None"
    assert hasattr(model, "intent"), "Model should have an 'intent' attribute"
    assert hasattr(model, "input_schema"), "Model should have an 'input_schema' attribute"
    assert hasattr(model, "output_schema"), "Model should have an 'output_schema' attribute"
    assert hasattr(model, "predictor"), "Model should have a 'predictor' attribute"
    assert isinstance(model.intent, str), "Intent should be of type str"
    assert hasattr(model, "predict"), "Model should have a 'predict' method"
    assert callable(model.predict), "Model's 'predict' should be callable"
    # Schema validation
    assert model.input_schema is not None
    assert model.output_schema is not None
    # Model should be in READY state if it was saved as a complete model
    from plexe.core.state import ModelState

    assert model.state == ModelState.READY


class TestFileIO:
    """Test cases for fileio module functionality."""

    def test_load_model_backwards_compatibility_v0_18_3(self):
        """Test loading a model bundle from v0.18.3 for backwards compatibility."""
        fixture_path = Path(__file__).parent.parent / "fixtures/legacy_models/model_v0_18_3.tar.gz"

        if not fixture_path.exists():
            pytest.skip(f"Legacy model fixture not found: {fixture_path}")

        # Load the legacy model
        model = fileio.load_model(fixture_path)

        _validate_model(model)

    def test_load_model_backwards_compatibility_v0_23_2(self):
        """Test loading a model bundle from v0.23.2 for backwards compatibility."""
        fixture_path = Path(__file__).parent.parent / "fixtures/legacy_models/model_v0_23_2.tar.gz"

        if not fixture_path.exists():
            pytest.skip(f"Legacy model fixture not found: {fixture_path}")

        # Load the legacy model
        model = fileio.load_model(fixture_path)

        _validate_model(model)

    def test_load_model_file_not_found(self):
        """Test that load_model raises appropriate error for missing files."""
        non_existent_path = Path("non_existent_model.tar.gz")

        with pytest.raises(ValueError, match="Failed to load model"):
            fileio.load_model(non_existent_path)
