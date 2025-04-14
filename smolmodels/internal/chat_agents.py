"""
This module provides a conversational wrapper around the core ML model building functionality.

It defines a chat agent that collects information from users and delegates to the Model class
for the actual model building process.
"""

import yaml
import importlib
import logging
from typing import Dict, Any, List

from smolagents import LiteLLMModel, tool, ToolCallingAgent
import pandas as pd
from pydantic import create_model

from smolmodels.models import Model

logger = logging.getLogger(__name__)


@tool
def build_model(
    intent: str,
    dataset_paths: List[str],
    input_schema: Dict[str, str] = None,
    output_schema: Dict[str, str] = None,
    provider: str = "openai/gpt-4o-mini",
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Build a machine learning model based on the provided information.

    Args:
        intent: Natural language description of what the model should do
        dataset_paths: Paths to CSV files containing training data
        input_schema: Dictionary mapping field names to types (as strings)
        output_schema: Dictionary mapping field names to types (as strings)
        provider: LLM provider to use for model generation
        max_iterations: Maximum number of iterations for model building

    Returns:
        Information about the built model
    """
    try:
        # Map string type names to actual types
        type_map = {
            "string": str,
            "str": str,
            "integer": int,
            "int": int,
            "float": float,
            "number": float,
            "boolean": bool,
            "bool": bool,
        }

        # Convert schemas to proper types
        input_types = {k: type_map.get(v.lower(), str) for k, v in input_schema.items()}
        output_types = {k: type_map.get(v.lower(), str) for k, v in output_schema.items()}

        # Create pydantic schemas
        input_model = create_model("InputSchema", **input_types)
        output_model = create_model("OutputSchema", **output_types)

        # Load datasets
        datasets = [pd.read_csv(path) for path in dataset_paths]

        # Create and build the model
        model = Model(intent=intent, input_schema=input_model, output_schema=output_model)

        model.build(
            datasets=datasets,
            provider=provider,
            max_iterations=max_iterations,
            timeout=1800,  # 30 minutes default timeout
        )

        # Return success information
        return {
            "success": True,
            "model_type": model.metadata.get("model_type", "Unknown"),
            "framework": model.metadata.get("framework", "Unknown"),
            "metrics": model.get_metrics(),
            "description": model.describe().to_dict(),
        }

    except Exception as e:
        logger.error(f"Error building model: {str(e)}")
        return {"success": False, "error": str(e)}


class ChatSmolmodelsAgent:
    """
    A conversational agent that helps users build ML models through natural conversation.
    """

    def __init__(self, model_id: str, verbose: bool = False):
        """
        Initialize the conversational ML agent.

        Args:
            model_id: Model ID for the LLM powering the agent
            verbose: Whether to display detailed logs
        """
        p_templates: dict = yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        predictor_custom_templates: dict = yaml.safe_load(
            importlib.resources.files("smolmodels")
            .joinpath("templates/prompts/agent")
            .joinpath("chat-system-prompt.yaml")
            .read_text()
        )
        # Merge the base and custom templates
        p_templates["system_prompt"] = predictor_custom_templates["system_prompt"]

        # Create the conversation agent with description instead of system_prompt
        # Based on the error, CodeAgent doesn't accept system_prompt
        self.agent = ToolCallingAgent(
            model=LiteLLMModel(model_id=model_id),
            tools=[build_model],
            verbosity_level=2 if verbose else 0,
            max_steps=30,
            prompt_templates=p_templates,
        )
