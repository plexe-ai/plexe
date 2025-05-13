"""
This module defines a multi-agent ML engineering system for building machine learning models.
"""

import logging
import types
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent

from plexe.config import config
from plexe.internal.common.registries.objects import ObjectRegistry
from plexe.internal.common.utils.agents import get_prompt_templates
from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.entities.code import Code
from plexe.internal.models.entities.metric import Metric
from plexe.internal.models.entities.metric import MetricComparator, ComparisonMethod
from plexe.internal.models.interfaces.predictor import Predictor
from plexe.internal.models.tools.training import (
    get_generate_training_code,
    get_fix_training_code,
)
from plexe.internal.models.tools.evaluation import get_review_finalised_model
from plexe.internal.models.tools.metrics import get_select_target_metric
from plexe.internal.models.tools.datasets import split_datasets, create_input_sample, get_dataset_preview
from plexe.internal.models.tools.schemas import get_raw_dataset_schema
from plexe.internal.models.tools.execution import get_executor_tool
from plexe.internal.models.tools.response_formatting import (
    format_final_orchestrator_agent_response,
    format_final_mle_agent_response,
    format_final_mlops_agent_response,
)
from plexe.internal.models.tools.context import get_inference_context_tool
from plexe.internal.models.tools.validation import validate_training_code, validate_inference_code

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
        tool_model_id: str = "openai/gpt-4o",
        verbose: bool = False,
        max_steps: int = 30,
        distributed: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
    ):
        """
        Initialize the multi-agent ML engineering system.

        Args:
            orchestrator_model_id: Model ID for the orchestrator agent
            ml_researcher_model_id: Model ID for the ML researcher agent
            ml_engineer_model_id: Model ID for the ML engineer agent
            ml_ops_engineer_model_id: Model ID for the ML ops engineer agent
            tool_model_id: Model ID for the model used inside tool calls
            verbose: Whether to display detailed agent logs
            max_steps: Maximum number of steps for the orchestrator agent
            distributed: Whether to run the agents in a distributed environment
            chain_of_thought_callable: Optional callable for chain of thought logging
        """
        self.orchestrator_model_id = orchestrator_model_id
        self.ml_researcher_model_id = ml_researcher_model_id
        self.ml_engineer_model_id = ml_engineer_model_id
        self.ml_ops_engineer_model_id = ml_ops_engineer_model_id
        self.tool_model_id = tool_model_id
        self.verbose = verbose
        self.max_steps = max_steps
        self.distributed = distributed
        self.chain_of_thought_callable = chain_of_thought_callable

        # Set verbosity levels
        self.orchestrator_verbosity = 2 if verbose else 0
        self.specialist_verbosity = 1 if verbose else 0

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
            ),
            model=LiteLLMModel(model_id=self.ml_researcher_model_id),
            tools=[get_dataset_preview],
            add_base_tools=False,
            verbosity_level=self.specialist_verbosity,
            prompt_templates=get_prompt_templates("toolcalling_agent.yaml", "mls_prompt_templates.yaml"),
            step_callbacks=[self.chain_of_thought_callable],
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
            ),
            model=LiteLLMModel(model_id=self.ml_engineer_model_id),
            tools=[
                get_generate_training_code(self.tool_model_id),
                validate_training_code,
                get_fix_training_code(self.tool_model_id),
                get_executor_tool(distributed),
                get_dataset_preview,
                format_final_mle_agent_response,
            ],
            add_base_tools=False,
            verbosity_level=self.specialist_verbosity,
            prompt_templates=get_prompt_templates("toolcalling_agent.yaml", "mle_prompt_templates.yaml"),
            step_callbacks=[self.chain_of_thought_callable],
        )

        # Create predictor builder agent - creates inference code
        self.mlops_engineer = CodeAgent(
            name="MLOperationsEngineer",
            description=(
                "Expert ML operations engineer that analyzes training code and creates high-quality production-ready "
                "inference code for ML models. To work effectively, as part of the 'task' prompt the agent STRICTLY requires:"
                "- input schema for the model"
                "- output schema for the model"
                "- the 'training code id' of the training code produced by the MLEngineer agent"
            ),
            model=LiteLLMModel(model_id=self.ml_ops_engineer_model_id),
            tools=[
                get_inference_context_tool(self.tool_model_id),
                validate_inference_code,
                format_final_mlops_agent_response,
            ],
            add_base_tools=False,
            verbosity_level=self.specialist_verbosity,
            additional_authorized_imports=config.code_generation.authorized_agent_imports,
            prompt_templates=get_prompt_templates("code_agent.yaml", "mlops_prompt_templates.yaml"),
            planning_interval=8,
            step_callbacks=[self.chain_of_thought_callable],
        )

        # Create orchestrator agent - coordinates the workflow
        self.manager_agent = CodeAgent(
            name="Orchestrator",
            model=LiteLLMModel(model_id=self.orchestrator_model_id),
            tools=[
                get_select_target_metric(self.tool_model_id),
                get_review_finalised_model(self.tool_model_id),
                split_datasets,
                create_input_sample,
                get_dataset_preview,
                get_raw_dataset_schema,
                format_final_orchestrator_agent_response,
            ],
            managed_agents=[self.ml_research_agent, self.mle_agent, self.mlops_engineer],
            add_base_tools=False,
            verbosity_level=self.orchestrator_verbosity,
            additional_authorized_imports=config.code_generation.authorized_agent_imports,
            max_steps=self.max_steps,
            prompt_templates=get_prompt_templates("code_agent.yaml", "manager_prompt_templates.yaml"),
            planning_interval=7,
            step_callbacks=[self.chain_of_thought_callable],
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
