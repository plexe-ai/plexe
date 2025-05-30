"""
This module defines a multi-agent ML engineering system for building machine learning models.
"""

import json
import logging
import types
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

from smolagents import CodeAgent, LiteLLMModel, AgentText

from plexe.agents.dataset_analyser import EdaAgent
from plexe.agents.dataset_splitter import DatasetSplitterAgent
from plexe.agents.feature_engineer import FeatureEngineeringAgent
from plexe.agents.model_packager import ModelPackagerAgent
from plexe.agents.model_planner import ModelPlannerAgent
from plexe.agents.model_tester import ModelTesterAgent
from plexe.agents.model_trainer import ModelTrainerAgent
from plexe.agents.schema_resolver import SchemaResolverAgent
from plexe.config import config
from plexe.core.object_registry import ObjectRegistry
from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.entities.code import Code
from plexe.internal.models.entities.metric import Metric
from plexe.internal.models.entities.metric import MetricComparator, ComparisonMethod
from plexe.core.entities.solution import Solution
from plexe.core.interfaces.predictor import Predictor
from plexe.tools.datasets import get_latest_datasets
from plexe.tools.evaluation import get_review_finalised_model, get_solution_performances
from plexe.tools.metrics import get_select_target_metric
from plexe.tools.response_formatting import (
    format_final_orchestrator_agent_response,
)
from plexe.tools.training import register_best_solution

logger = logging.getLogger(__name__)


@dataclass
class ModelGenerationResult:
    training_source_code: str
    inference_source_code: str
    feature_transformer_source_code: str
    dataset_split_code: str
    predictor: Predictor
    model_artifacts: List[Artifact]
    performance: Metric  # Validation performance
    test_performance: Metric = None  # Test set performance
    testing_source_code: str = None  # Testing code from model tester agent
    evaluation_report: Dict = None  # Evaluation report from model tester agent
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
        max_steps: int = 50,
        distributed: bool = False,
        chain_of_thought_callable: Optional[Callable] = None,
        max_solutions: int = 1,
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
        self.orchestrator_verbosity = 1 if verbose else 0
        self.specialist_verbosity = 1 if verbose else 0

        # Create solution planner agent - plans ML approaches
        self.ml_research_agent = ModelPlannerAgent(
            model_id=self.ml_researcher_model_id,
            verbose=verbose,
            chain_of_thought_callable=chain_of_thought_callable,
            max_solutions=max_solutions,
        ).agent

        # Create and run the schema resolver agent
        self.schema_resolver_agent = SchemaResolverAgent(
            model_id=self.orchestrator_model_id,
            verbose=verbose,
            chain_of_thought_callable=chain_of_thought_callable,
        ).agent

        # Create the EDA agent to analyze the dataset
        self.eda_agent = EdaAgent(
            model_id=self.orchestrator_model_id,
            verbose=verbose,
            chain_of_thought_callable=chain_of_thought_callable,
        ).agent

        # Create feature engineering agent - transforms raw datasets for better model performance
        self.feature_engineering_agent = FeatureEngineeringAgent(
            model_id=self.ml_engineer_model_id,
            verbose=verbose,
            chain_of_thought_callable=self.chain_of_thought_callable,
        ).agent

        # Create dataset splitter agent - intelligently splits datasets
        self.dataset_splitter_agent = DatasetSplitterAgent(
            model_id=self.orchestrator_model_id,
            verbose=verbose,
            chain_of_thought_callable=self.chain_of_thought_callable,
        ).agent

        # Create model trainer agent - implements training code
        self.mle_agent = ModelTrainerAgent(
            ml_engineer_model_id=self.ml_engineer_model_id,
            tool_model_id=self.tool_model_id,
            distributed=self.distributed,
            verbose=verbose,
            chain_of_thought_callable=self.chain_of_thought_callable,
            schema_resolver_agent=self.schema_resolver_agent,
        ).agent

        # Create predictor builder agent - creates inference code
        self.mlops_engineer = ModelPackagerAgent(
            model_id=self.ml_ops_engineer_model_id,
            tool_model_id=self.tool_model_id,
            verbose=verbose,
            chain_of_thought_callable=self.chain_of_thought_callable,
            schema_resolver_agent=self.schema_resolver_agent,
        ).agent

        # Create model tester agent - tests and evaluates the finalized model
        self.model_tester_agent = ModelTesterAgent(
            model_id=self.ml_engineer_model_id,
            verbose=verbose,
            chain_of_thought_callable=self.chain_of_thought_callable,
        ).agent

        # Create orchestrator agent - coordinates the workflow
        self.manager_agent = CodeAgent(
            name="Orchestrator",
            model=LiteLLMModel(model_id=self.orchestrator_model_id),
            tools=[
                get_select_target_metric(self.tool_model_id),
                get_review_finalised_model(self.tool_model_id),
                get_latest_datasets,
                get_solution_performances,
                register_best_solution,
                format_final_orchestrator_agent_response,
            ],
            managed_agents=[
                self.eda_agent,
                self.schema_resolver_agent,
                self.feature_engineering_agent,
                self.ml_research_agent,
                self.dataset_splitter_agent,
                self.mle_agent,
                self.mlops_engineer,
                self.model_tester_agent,
            ],
            add_base_tools=False,
            verbosity_level=self.orchestrator_verbosity,
            additional_authorized_imports=config.code_generation.authorized_agent_imports,
            max_steps=self.max_steps,
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

        print(f"Registry contents:\n\n" f"{json.dumps(sorted(object_registry.list()), indent=4)}" f"\n\n")

        try:
            # Only log the full result when in verbose mode
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Agent result: %s", result)

            if isinstance(result, AgentText):
                result = json.loads(str(result))

            # Extract data from the agent result
            best_solution = object_registry.get(Solution, "best_performing_solution")
            training_code = best_solution.training_code
            inference_code = best_solution.inference_code

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

            # Model metadata
            metadata = result.get("metadata", {"model_type": "unknown", "framework": "unknown"})

            # Compile the inference code into a module
            inference_module: types.ModuleType = types.ModuleType("predictor")
            exec(inference_code, inference_module.__dict__)
            # Instantiate the predictor class from the loaded module
            predictor_class = getattr(inference_module, "PredictorImplementation")
            predictor = predictor_class(best_solution.model_artifacts)

            # Get feature transformer code if available
            feature_transformer_code = None
            try:
                feature_code = object_registry.get(Code, "feature_transformations")
                if feature_code:
                    feature_transformer_code = feature_code.code
            except KeyError:
                # No feature transformations code found, that's ok
                pass

            # Get dataset split code if available
            dataset_split_code = None
            try:
                dataset_split_code = object_registry.get(Code, "dataset_splitting_code")
                if dataset_split_code:
                    dataset_split_code = dataset_split_code.code
            except KeyError:
                # No dataset split code found, that's ok
                pass

            # Get testing code if available
            testing_code = None
            try:
                testing_code = best_solution.testing_code
            except Exception:
                # No testing code found, that's ok
                pass

            # Get evaluation report if available
            evaluation_report = None
            try:
                evaluation_report = best_solution.model_evaluation_report
            except Exception:
                # No evaluation report found, that's ok
                pass

            return ModelGenerationResult(
                training_source_code=training_code,
                inference_source_code=inference_code,
                feature_transformer_source_code=feature_transformer_code,
                dataset_split_code=dataset_split_code,
                predictor=predictor,
                model_artifacts=best_solution.model_artifacts,
                performance=performance,
                test_performance=performance,  # Using the same performance for now
                testing_source_code=testing_code,
                evaluation_report=evaluation_report,
                metadata=metadata,
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to process agent result: {str(e)}") from e
