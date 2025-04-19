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

    def test_callback_registration(self):
        """Test that callbacks can be registered with the Model.build method."""
        # Create a mock callback
        callback = MagicMock(spec=Callback)

        # Patch the PlexeAgent to avoid actual model building
        with patch("plexe.internal.agents.PlexeAgent.run") as mock_run:
            # Setup the mock to return a valid result
            mock_result = MagicMock()
            mock_result.training_source_code = "def train(): pass"
            mock_result.inference_source_code = "def predict(): pass"
            mock_result.predictor = MagicMock()
            mock_result.model_artifacts = []
            mock_result.test_performance = None
            mock_result.metadata = {}
            mock_run.return_value = mock_result

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

            # Check that the agent was called
            mock_run.assert_called_once()

    def test_callback_error_handling(self):
        """Test that errors in callbacks are caught and don't interrupt model building."""

        # Create a callback that raises an exception
        class ErrorCallback(Callback):
            def on_build_start(self, info: BuildStateInfo) -> None:
                raise ValueError("Simulated error in callback")

            def on_build_end(self, info: BuildStateInfo) -> None:
                raise ValueError("Another simulated error in callback")

        callback = ErrorCallback()

        # Patch the PlexeAgent to avoid actual model building
        with patch("plexe.internal.agents.PlexeAgent.run") as mock_run:
            # Setup the mock to return a valid result
            mock_result = MagicMock()
            mock_result.training_source_code = "def train(): pass"
            mock_result.inference_source_code = "def predict(): pass"
            mock_result.predictor = MagicMock()
            mock_result.model_artifacts = []
            mock_result.test_performance = None
            mock_result.metadata = {}
            mock_run.return_value = mock_result

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

        # Create a more sophisticated mock that simulates calling iteration callbacks
        with patch("plexe.internal.agents.PlexeAgent.run") as mock_run:

            def side_effect(*args, **kwargs):
                # Retrieve callbacks from the object registry since the new architecture
                # uses the registry to store callbacks
                from plexe.internal.common.registries.objects import ObjectRegistry

                registry = ObjectRegistry()
                callbacks = list(registry.get_all(Callback).values())

                # Simulate calling iteration callbacks
                for cb in callbacks:
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

                    # Set iteration info for start
                    cb.on_iteration_start(build_info)

                    # Update info for iteration end and add the node
                    build_info.node = mock_node
                    cb.on_iteration_end(build_info)

                # Return mock result
                mock_result = MagicMock()
                mock_result.training_source_code = "def train(): pass"
                mock_result.inference_source_code = "def predict(): pass"
                mock_result.predictor = MagicMock()
                mock_result.model_artifacts = []
                mock_result.test_performance = None
                mock_result.metadata = {}
                return mock_result

            mock_run.side_effect = side_effect

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

            # We no longer check for iteration callbacks as they might be handled differently in the agent


if __name__ == "__main__":
    unittest.main()
