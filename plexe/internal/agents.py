"""
This module defines a multi-agent ML engineering system for building machine learning models.

This implementation can be used both programmatically through the library interface
and interactively through a Gradio UI.
"""

import types
import logging
from typing import List, Dict
from dataclasses import dataclass, field

from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent

from plexe.config import config
from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.entities.code import Code
from plexe.internal.models.tools.evaluation import review_finalised_model
from plexe.internal.models.tools.execution import execute_training_code
from plexe.internal.models.tools.code_generation import (
    generate_inference_code,
    fix_inference_code,
    generate_training_code,
    fix_training_code,
)
from plexe.internal.models.tools.metrics import select_target_metric
from plexe.internal.models.tools.validation import validate_inference_code, validate_training_code
from plexe.internal.models.tools.datasets import split_datasets, create_input_sample
from plexe.internal.models.tools.response_formatting import (
    format_final_orchestrator_agent_response,
    format_final_mle_agent_response,
    format_final_mlops_agent_response,
)
from plexe.internal.models.interfaces.predictor import Predictor
from plexe.internal.models.entities.metric import Metric
from plexe.internal.common.registries.objects import ObjectRegistry
from plexe.internal.models.entities.metric import MetricComparator, ComparisonMethod
from plexe.internal.common.utils.agents import get_prompt_templates


logger = logging.getLogger(__name__)


@dataclass
class ModelGenerationResult:
    training_source_code: str
    inference_source_code: str
    predictor: Predictor
    model_artifacts: List[Artifact]
    performance: Metric  # Validation performance
    test_performance: Metric = None  # Test set performance
    metadata: Dict[str, str] = field(default_factory=dict)  # Model metadata


class PlexeAgent:
    """
    Multi-agent ML engineering system for building machine learning models.

    This class creates and manages a system of specialized agents that work together
    to analyze data, plan solutions, train models, and generate inference code.
    """

    def __init__(
        self,
        orchestrator_model_id: str = "anthropic/claude-3-7-sonnet-20250219",
        ml_researcher_model_id: str = "openai/gpt-4o",
        ml_engineer_model_id: str = "anthropic/claude-3-7-sonnet-20250219",
        ml_ops_engineer_model_id: str = "anthropic/claude-3-7-sonnet-20250219",
        verbose: bool = False,
        max_steps: int = 30,
    ):
        """
        Initialize the multi-agent ML engineering system.

        Args:
            orchestrator_model_id: Model ID for the orchestrator agent
            verbose: Whether to display detailed agent logs
            max_steps: Maximum number of steps for the orchestrator agent
        """
        self.orchestrator_model_id = orchestrator_model_id
        self.ml_researcher_model_id = ml_researcher_model_id
        self.ml_engineer_model_id = ml_engineer_model_id
        self.ml_ops_engineer_model_id = ml_ops_engineer_model_id
        self.verbose = verbose
        self.max_steps = max_steps

        # Set verbosity levels
        self.orchestrator_verbosity = 2 if verbose else 1
        self.specialist_verbosity = 1 if verbose else 1

        # Create solution planner agent - plans ML approaches
        self.ml_research_agent = ToolCallingAgent(
            name="MLResearchScientist",
            description=(
                "Expert ML researcher that develops detailed solution ideas and plans for ML use cases. "
                "To work effectively, as part of the 'task' prompt the agent STRICTLY requires:"
                "- the ML task definition (i.e. 'intent')"
                "- input schema for the model"
                "- output schema for the model"
                "- the name and comparison method of the metric to optimise"
                "- the identifier of the LLM that should be used for plan generation"
            ),
            model=LiteLLMModel(model_id=self.ml_researcher_model_id),
            tools=[],
            add_base_tools=False,
            verbosity_level=self.specialist_verbosity,
            prompt_templates=get_prompt_templates("toolcalling_agent.yaml", "mls_prompt_templates.yaml"),
        )

        # Create model trainer agent - implements training code
        self.mle_agent = ToolCallingAgent(
            name="MLEngineer",
            description=(
                "Expert ML engineer that implements, trains and validates ML models based on provided plans. "
                "To work effectively, as part of the 'task' prompt the agent STRICTLY requires:"
                "- the ML task definition (i.e. 'intent')"
                "- input schema for the model"
                "- output schema for the model"
                "- the name and comparison method of the metric to optimise"
                "- the full solution plan that outlines how to solve this problem"
                "- the split train/validation dataset names"
                "- the working directory to use for model execution"
                "- the identifier of the LLM that should be used for code generation"
            ),
            model=LiteLLMModel(model_id=self.ml_engineer_model_id),
            tools=[
                generate_training_code,
                validate_training_code,
                fix_training_code,
                execute_training_code,
                format_final_mle_agent_response,
            ],
            add_base_tools=False,
            verbosity_level=self.specialist_verbosity,
            prompt_templates=get_prompt_templates("toolcalling_agent.yaml", "mle_prompt_templates.yaml"),
        )

        # Create predictor builder agent - creates inference code
        self.mlops_engineer = ToolCallingAgent(
            name="MLOperationsEngineer",
            description=(
                "Expert ML operations engineer that writes inference code for ML models to be used in production. "
                "To work effectively, as part of the 'task' prompt the agent STRICTLY requires:"
                "- input schema for the model"
                "- output schema for the model"
                "- the 'training code id' of the training code produced by the MLEngineer agent"
                "- the identifier of the LLM that should be used for code generation"
            ),
            model=LiteLLMModel(model_id=self.ml_ops_engineer_model_id),
            tools=[
                split_datasets,
                generate_inference_code,
                validate_inference_code,
                fix_inference_code,
                format_final_mlops_agent_response,
            ],
            add_base_tools=False,
            verbosity_level=self.specialist_verbosity,
            prompt_templates=get_prompt_templates("toolcalling_agent.yaml", "mlops_prompt_templates.yaml"),
            planning_interval=8,
        )

        # Create orchestrator agent - coordinates the workflow
        self.manager_agent = CodeAgent(
            model=LiteLLMModel(model_id=self.orchestrator_model_id),
            tools=[
                select_target_metric,
                review_finalised_model,
                split_datasets,
                create_input_sample,
                format_final_orchestrator_agent_response,
            ],
            managed_agents=[self.ml_research_agent, self.mle_agent, self.mlops_engineer],
            add_base_tools=False,
            verbosity_level=self.orchestrator_verbosity,
            additional_authorized_imports=config.code_generation.authorized_agent_imports,
            max_steps=self.max_steps,
            prompt_templates=get_prompt_templates("code_agent.yaml", "manager_prompt_templates.yaml"),
            planning_interval=7,
        )

    def run(self, task, additional_args: dict) -> ModelGenerationResult:
        """
        Run the orchestrator agent to generate a machine learning model.

        Returns:
            ModelGenerationResult: The result of the model generation process.
        """
        object_registry = ObjectRegistry()
        result = self.manager_agent.run(task=task, additional_args=additional_args)

        try:
            # Only log the full result when in verbose mode
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Agent result: %s", result)

            # Extract data from the agent result
            training_code_id = result.get("training_code_id", "")
            inference_code_id = result.get("inference_code_id", "")
            training_code = object_registry.get(Code, training_code_id).code
            inference_code = object_registry.get(Code, inference_code_id).code

            # Extract performance metrics
            if "performance" in result and isinstance(result["performance"], dict):
                metrics = result["performance"]
            else:
                metrics = {}

            metric_name = metrics.get("name", "unknown")
            metric_value = metrics.get("value", 0.0)
            comparison_str = metrics.get("comparison_method", "")
            comparison_method_map = {
                "HIGHER_IS_BETTER": ComparisonMethod.HIGHER_IS_BETTER,
                "LOWER_IS_BETTER": ComparisonMethod.LOWER_IS_BETTER,
                "TARGET_IS_BETTER": ComparisonMethod.TARGET_IS_BETTER,
            }
            comparison_method = ComparisonMethod.HIGHER_IS_BETTER  # Default to higher is better
            for key, method in comparison_method_map.items():
                if key in comparison_str:
                    comparison_method = method

            comparator = MetricComparator(comparison_method)
            performance = Metric(
                name=metric_name,
                value=metric_value,
                comparator=comparator,
            )

            # Get model artifacts from registry or result
            artifact_names = result.get("model_artifact_names", [])

            # Model metadata
            metadata = result.get("metadata", {"model_type": "unknown", "framework": "unknown"})

            # Compile the inference code into a module
            inference_module: types.ModuleType = types.ModuleType("predictor")
            exec(inference_code, inference_module.__dict__)
            # Instantiate the predictor class from the loaded module
            predictor_class = getattr(inference_module, "PredictorImplementation")
            predictor = predictor_class(object_registry.get_all(Artifact).values())

            return ModelGenerationResult(
                training_source_code=training_code,
                inference_source_code=inference_code,
                predictor=predictor,
                model_artifacts=list(object_registry.get_multiple(Artifact, artifact_names).values()),
                performance=performance,
                test_performance=performance,  # Using the same performance for now
                metadata=metadata,
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to process agent result: {str(e)}") from e
