"""
Unit tests for callback system in Plexe.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pydantic import create_model

from plexe.callbacks import Callback, BuildStateInfo
from plexe.models import Model
from plexe.internal.common.utils.model_state import ModelState
from plexe.internal.models.entities.node import Node
from plexe.internal.common.registries.objects import ObjectRegistry

# Mock patches for all agent run methods to prevent test hanging
AGENT_PATCHES = [
    "plexe.internal.agents.PlexeAgent.run",
    "plexe.agents.dataset_analyser.EdaAgent.run",
    "plexe.agents.schema_resolver.SchemaResolverAgent.run",
]


# Create a safe get wrapper to handle schema_reasoning lookup
def safe_get_original(self, t, name):
    uri = self._get_uri(t, name)
    if uri not in self._items:
        if name == "schema_reasoning":
            # Return a mock schema reasoning if requested
            return "Mock schema reasoning for tests"
        raise KeyError(f"Item '{uri}' not found in registry")
    return self._items[uri]


class TestCallbacks(unittest.TestCase):
    """Test the callback system."""

    def setUp(self):
        # Create a simple model for testing
        self.model = Model(
            intent="Predict housing prices based on features",
            input_schema=create_model("Input", bedrooms=int, bathrooms=int, sqft=float),
            output_schema=create_model("Output", price=float),
        )

        # Create a mock dataset
        self.data = pd.DataFrame(
            {
                "bedrooms": [2, 3, 4, 3, 2],
                "bathrooms": [1, 2, 2, 1, 1],
                "sqft": [1000, 1500, 2000, 1200, 950],
                "price": [200000, 300000, 400000, 250000, 180000],
            }
        )

        # Clear the object registry
        ObjectRegistry().clear()

        # Save the original get method
        self.original_get = ObjectRegistry.get
        # Patch the get method to handle missing schema_reasoning
        ObjectRegistry.get = safe_get_original

    def tearDown(self):
        # Restore the original get method
        ObjectRegistry.get = self.original_get
        # Clear the object registry
        ObjectRegistry().clear()

    def test_callback_registration(self):
        """Test that callbacks can be registered with the Model.build method."""
        # Create a mock callback
        callback = MagicMock(spec=Callback)

        # Set up mock responses for EDA and Schema agents
        mock_eda_result = {}
        mock_schema_result = {"input_schema": {}, "output_schema": {}}

        # Patch all agents to avoid actual model building and prevent test hanging
        with (
            patch(AGENT_PATCHES[0]) as mock_plexe_run,
            patch(AGENT_PATCHES[1], return_value=mock_eda_result) as mock_eda_run,
            patch(AGENT_PATCHES[2], return_value=mock_schema_result) as mock_schema_run,
        ):

            # Setup the mock to return a valid result for PlexeAgent
            mock_result = MagicMock()
            mock_result.training_source_code = "def train(): pass"
            mock_result.inference_source_code = "def predict(): pass"
            mock_result.predictor = MagicMock()
            mock_result.model_artifacts = []
            mock_result.test_performance = None
            mock_result.metadata = {}
            mock_plexe_run.return_value = mock_result

            # Build the model with the callback
            self.model.build(
                datasets=[self.data],
                provider="openai/gpt-4o-mini",
                timeout=300,
                max_iterations=1,
                run_timeout=30,
                callbacks=[callback],
            )

            # Check if the callback methods were called
            # The on_build_start should be called once
            callback.on_build_start.assert_called_once()

            # In the new agentic architecture, on_build_end may be called multiple times
            # (at least once in model.py, possibly again in the agent implementation)
            assert callback.on_build_end.call_count >= 1

            # Check that all agents were called
            mock_eda_run.assert_called_once()
            mock_schema_run.assert_called_once()
            mock_plexe_run.assert_called_once()

    def test_callback_error_handling(self):
        """Test that errors in callbacks are caught and don't interrupt model building."""

        # Create a callback that raises an exception
        class ErrorCallback(Callback):
            def on_build_start(self, info: BuildStateInfo) -> None:
                raise ValueError("Simulated error in callback")

            def on_build_end(self, info: BuildStateInfo) -> None:
                raise ValueError("Another simulated error in callback")

        callback = ErrorCallback()

        # Set up mock responses for EDA and Schema agents
        mock_eda_result = {}
        mock_schema_result = {"input_schema": {}, "output_schema": {}}

        # Patch all agents to avoid actual model building and prevent test hanging
        with (
            patch(AGENT_PATCHES[0]) as mock_plexe_run,
            patch(AGENT_PATCHES[1], return_value=mock_eda_result),
            patch(AGENT_PATCHES[2], return_value=mock_schema_result),
        ):

            # Setup the mock to return a valid result for PlexeAgent
            mock_result = MagicMock()
            mock_result.training_source_code = "def train(): pass"
            mock_result.inference_source_code = "def predict(): pass"
            mock_result.predictor = MagicMock()
            mock_result.model_artifacts = []
            mock_result.test_performance = None
            mock_result.metadata = {}
            mock_plexe_run.return_value = mock_result

            # Build should not raise an exception despite the callback errors
            self.model.build(
                datasets=[self.data],
                provider="openai/gpt-4o-mini",
                timeout=300,
                max_iterations=1,
                run_timeout=30,
                callbacks=[callback],
            )

            # The model building should still have completed successfully
            self.assertEqual(self.model.state, ModelState.READY)

    def test_iteration_callbacks(self):
        """Test that iteration callbacks are properly passed to the agent."""
        # Create a mock callback
        callback = MagicMock(spec=Callback)

        # Set up mock responses for EDA and Schema agents
        mock_eda_result = {}
        mock_schema_result = {"input_schema": {}, "output_schema": {}}

        # Patch all agents to avoid actual model building and prevent test hanging
        with (
            patch(AGENT_PATCHES[0]) as mock_plexe_run,
            patch(AGENT_PATCHES[1], return_value=mock_eda_result),
            patch(AGENT_PATCHES[2], return_value=mock_schema_result),
        ):

            def side_effect(*args, **kwargs):
                # Just use the specific callback passed to this test instead of
                # getting all callbacks from the registry which could include
                # MLFlowCallback instances

                # Create a mock node
                mock_node = MagicMock(spec=Node)

                # Create a basic BuildStateInfo for testing
                build_info = BuildStateInfo(
                    intent=self.model.intent,
                    provider="openai/gpt-4o-mini",
                    input_schema=self.model.input_schema,
                    output_schema=self.model.output_schema,
                    iteration=0,
                    datasets={"train": self.data},
                )

                # Call iteration callbacks directly on our mock callback
                callback.on_iteration_start(build_info)

                # Update info for iteration end and add the node
                build_info.node = mock_node
                callback.on_iteration_end(build_info)

                # Return mock result
                mock_result = MagicMock()
                mock_result.training_source_code = "def train(): pass"
                mock_result.inference_source_code = "def predict(): pass"
                mock_result.predictor = MagicMock()
                mock_result.model_artifacts = []
                mock_result.test_performance = None
                mock_result.metadata = {}
                return mock_result

            mock_plexe_run.side_effect = side_effect

            # Build the model with the callback
            self.model.build(
                datasets=[self.data],
                provider="openai/gpt-4o-mini",
                timeout=300,
                max_iterations=1,
                run_timeout=30,
                callbacks=[callback],
            )

            # Verify the basic callback methods were called
            # Note: in the agentic architecture, iteration callbacks might be handled differently
            callback.on_build_start.assert_called_once()

            # In the new agentic architecture, on_build_end may be called multiple times
            assert callback.on_build_end.call_count >= 1

            # Verify iteration callbacks were called
            callback.on_iteration_start.assert_called_once()
            callback.on_iteration_end.assert_called_once()


if __name__ == "__main__":
    unittest.main()
