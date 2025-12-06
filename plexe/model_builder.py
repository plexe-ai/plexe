"""
ModelBuilder for creating ML models through agentic workflows.

This module provides the ModelBuilder class that handles the orchestration of 
the multi-agent system to build machine learning models.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Type, Optional

import pandas as pd
from pydantic import BaseModel, ValidationError

from plexe.config import prompt_templates
from plexe.datasets import DatasetGenerator
from plexe.callbacks import Callback, BuildStateInfo, ChainOfThoughtModelCallback, ModelCheckpointCallback
from plexe.internal.common.utils.chain_of_thought.emitters import ConsoleEmitter
from plexe.agents.agents import PlexeAgent
from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.internal.common.provider import ProviderConfig, Provider
from plexe.core.object_registry import ObjectRegistry
from plexe.internal.common.utils.pydantic_utils import map_to_basemodel, format_schema
from plexe.internal.common.utils.markdown_utils import format_eda_report_markdown
from plexe.core.state import ModelState
from plexe.tools.schemas import get_solution_schemas

logger = logging.getLogger(__name__)


class ModelBuilder:
    """Factory for creating ML models through agentic workflows."""

    def __init__(
        self,
        provider: str | ProviderConfig = "openai/gpt-4o-mini",
        verbose: bool = False,
        distributed: bool = False,
        working_dir: Optional[str] = None,
    ):
        """
        Initialize the model builder.

        Args:
            provider: LLM provider configuration
            verbose: Whether to display detailed agent logs
            distributed: Whether to use distributed training with Ray
            working_dir: Optional custom working directory
        """
        self.provider_config = ProviderConfig(default_provider=provider) if isinstance(provider, str) else provider
        self.verbose = verbose
        self.distributed = distributed
        self.working_dir = working_dir or self._create_working_dir()

    @staticmethod
    def _create_working_dir() -> str:
        """Create unique working directory for this build."""
        run_id = f"run-{datetime.now().isoformat()}".replace(":", "-").replace(".", "-")
        working_dir = f"./workdir/{run_id}/"
        os.makedirs(working_dir, exist_ok=True)
        return working_dir

    def _validate_intent(self, intent: str) -> None:
        """
        Validate the intent string using basic checks and LLM validation.
        
        Performs:
        1. Basic validation (non-empty, reasonable length)
        2. LLM-based validation to check if intent is clear and actionable
        
        :param intent: The intent string to validate
        :raises ValueError: If intent fails validation
        """
        # Basic validation first (no LLM call needed)
        if not intent or not intent.strip():
            raise ValueError("Intent cannot be empty or whitespace only")
        
        if len(intent.strip()) < 10:
            raise ValueError("Intent is too short. Please provide a more detailed description (at least 10 characters)")
        
        if len(intent) > 2000:
            raise ValueError("Intent is too long. Please keep it under 2000 characters")
        
        # LLM-based validation for quality and clarity
        try:
            class IntentValidationResponse(BaseModel):
                """Response model for intent validation."""
                is_valid: bool
                feedback: str  # Optional feedback on how to improve the intent
            
            provider = Provider(self.provider_config.tool_provider)
            
            validation_prompt = f"""Analyze the following machine learning model intent and determine if it is clear, actionable, and well-defined for building an ML model.

Intent: "{intent}"

Check if the intent:
1. Clearly describes what the model should predict or classify
2. Is specific enough to build a model (not too vague)
3. Describes a valid ML task (not something that requires non-ML solutions)
4. Has sufficient detail for an ML engineer to understand the problem

Return whether the intent is valid and provide brief feedback if improvements are needed."""

            response_str = provider.query(
                system_message="You are an expert ML consultant that validates machine learning problem definitions. Analyze intents for clarity, specificity, and feasibility.",
                user_message=validation_prompt,
                response_format=IntentValidationResponse,
            )
            
            # Parse and validate response
            try:
                response_dict = json.loads(response_str) if isinstance(response_str, str) else response_str
                validation = IntentValidationResponse(**response_dict)
            except (json.JSONDecodeError, ValidationError) as parse_error:
                logger.warning(f"Failed to parse LLM validation response: {str(parse_error)}")
                # Don't block - basic validation passed, LLM parsing failed
                return
            
            if not validation.is_valid:
                error_msg = f"Intent validation failed: {validation.feedback}"
                logger.warning(f"Intent validation failed: {validation.feedback}")
                raise ValueError(error_msg)
            
            logger.debug(f"Intent validation passed: {validation.feedback if validation.feedback else 'Intent is clear and actionable'}")
            
        except ValueError:
            # Re-raise ValueError (validation failed) - this should block the build
            raise
        except Exception as e:
            # If LLM validation fails for other reasons, log warning but don't block (graceful degradation)
            # Basic validation already passed, so we allow it through
            logger.warning(f"LLM intent validation failed, proceeding with basic validation only: {str(e)}")
            # Don't raise - basic validation passed, LLM validation is a nice-to-have

    def build(
        self,
        intent: str,
        datasets: List[pd.DataFrame | DatasetGenerator],
        input_schema: Type[BaseModel] | Dict[str, type] = None,
        output_schema: Type[BaseModel] | Dict[str, type] = None,
        timeout: int = None,
        max_iterations: int = None,
        run_timeout: int = 1800,
        callbacks: List[Callback] = None,
        enable_checkpointing: bool = False,
    ):
        """
        Build a complete ML model using the agentic workflow.

        Args:
            intent: Natural language description of the model's purpose
            datasets: Training datasets
            input_schema: Optional input schema (inferred if not provided)
            output_schema: Optional output schema (inferred if not provided)
            timeout: Maximum total time for building
            max_iterations: Maximum number of iterations
            run_timeout: Maximum time per training run
            callbacks: Optional callbacks for monitoring
            enable_checkpointing: Whether to enable checkpointing

        Returns:
            Completed Model instance
        """
        # Clear and use singleton object registry for this build
        object_registry = ObjectRegistry()
        object_registry.clear()

        # Validate parameters
        if timeout is None and max_iterations is None:
            raise ValueError("At least one of 'timeout' or 'max_iterations' must be set")
        if run_timeout is not None and timeout is not None and run_timeout > timeout:
            raise ValueError(f"Run timeout ({run_timeout}s) cannot exceed total timeout ({timeout}s)")

        # Validate intent using LLM
        self._validate_intent(intent)

        # Process schemas
        input_schema = map_to_basemodel("in", input_schema) if input_schema else None
        output_schema = map_to_basemodel("out", output_schema) if output_schema else None

        object_registry.register(bool, "input_schema_is_locked", input_schema is not None, immutable=True)
        object_registry.register(bool, "output_schema_is_locked", output_schema is not None, immutable=True)

        # Initialize callbacks
        callbacks = callbacks or []
        if enable_checkpointing and not any(isinstance(cb, ModelCheckpointCallback) for cb in callbacks):
            callbacks.append(ModelCheckpointCallback())

        cot_callback = ChainOfThoughtModelCallback(emitter=ConsoleEmitter())
        callbacks.append(cot_callback)
        cot_callable = cot_callback.get_chain_of_thought_callable()

        # Register callbacks
        object_registry.register_multiple(Callback, {f"{i}": c for i, c in enumerate(callbacks)})

        try:
            # Register datasets
            training_data = {
                f"dataset_{i}": DatasetAdapter.coerce((data.data if isinstance(data, DatasetGenerator) else data))
                for i, data in enumerate(datasets)
            }
            object_registry.register_multiple(TabularConvertible, training_data, immutable=True)

            # Register schemas if provided
            if input_schema:
                object_registry.register(dict, "input_schema", format_schema(input_schema), immutable=True)
            if output_schema:
                object_registry.register(dict, "output_schema", format_schema(output_schema), immutable=True)

            # Generate unique model identifier
            model_identifier = f"model-{datetime.now().isoformat()}".replace(":", "-").replace(".", "-")

            # Notify callbacks of build start
            self._notify_callbacks(
                callbacks,
                "build_start",
                intent,
                input_schema,
                output_schema,
                training_data,
                model_identifier=model_identifier,
                model_state="BUILDING",
            )

            # Create and run agent
            agent = PlexeAgent(
                orchestrator_model_id=self.provider_config.orchestrator_provider,
                ml_researcher_model_id=self.provider_config.research_provider,
                ml_engineer_model_id=self.provider_config.engineer_provider,
                ml_ops_engineer_model_id=self.provider_config.ops_provider,
                tool_model_id=self.provider_config.tool_provider,
                verbose=self.verbose,
                max_steps=30,
                distributed=self.distributed,
                chain_of_thought_callable=cot_callable,
                max_solutions=max_iterations,
            )

            agent_prompt = prompt_templates.agent_builder_prompt(
                intent=intent,
                input_schema=json.dumps(format_schema(input_schema), indent=4),
                output_schema=json.dumps(format_schema(output_schema), indent=4),
                datasets=list(training_data.keys()),
                working_dir=self.working_dir,
                max_iterations=max_iterations,
                resume=False,
            )

            additional_args = {
                "intent": intent,
                "working_dir": self.working_dir,
                "input_schema": format_schema(input_schema),
                "output_schema": format_schema(output_schema),
                "max_iterations": max_iterations,
                "timeout": timeout,
                "run_timeout": run_timeout,
            }

            generated = agent.run(agent_prompt, additional_args=additional_args)

            # Extract final schemas (may have been inferred)
            schemas = get_solution_schemas("best_performing_solution")
            final_input_schema = map_to_basemodel("InputSchema", schemas["input"])
            final_output_schema = map_to_basemodel("OutputSchema", schemas["output"])

            # Build metadata
            metadata = {
                "provider": str(self.provider_config.default_provider),
                "orchestrator_provider": str(self.provider_config.orchestrator_provider),
                "research_provider": str(self.provider_config.research_provider),
                "engineer_provider": str(self.provider_config.engineer_provider),
                "ops_provider": str(self.provider_config.ops_provider),
                "tool_provider": str(self.provider_config.tool_provider),
            }
            metadata.update(generated.metadata)

            # Extract EDA reports
            eda_reports = {}
            eda_markdown_reports = {}
            for name in training_data.keys():
                try:
                    eda_report = object_registry.get(dict, f"eda_report_{name}")
                    eda_reports[name] = eda_report
                    eda_markdown_reports[name] = format_eda_report_markdown(eda_report)
                except KeyError:
                    logger.debug(f"No EDA report found for dataset '{name}'")

            metadata["eda_reports"] = eda_reports
            metadata["eda_markdown_reports"] = eda_markdown_reports

            # Import here to avoid circular dependency
            from plexe.models import Model

            # Create model and populate it with results
            model = Model(intent=intent, input_schema=final_input_schema, output_schema=final_output_schema)
            model.identifier = model_identifier
            model.predictor = generated.predictor
            model.trainer_source = generated.training_source_code
            model.predictor_source = generated.inference_source_code
            model.feature_transformer_source = generated.feature_transformer_source_code
            model.dataset_splitter_source = generated.dataset_split_code
            model.testing_source = generated.testing_source_code
            model.artifacts = generated.model_artifacts
            model.metric = generated.test_performance
            model.evaluation_report = generated.evaluation_report
            model.metadata.update(metadata)
            model.training_data = training_data  # Store actual training data
            model.state = ModelState.READY

            # Notify callbacks of build end
            self._notify_callbacks(
                callbacks,
                "build_end",
                intent,
                final_input_schema,
                final_output_schema,
                training_data,
                model_identifier=model_identifier,
                model_state="READY",
                final_metric=generated.test_performance,
                final_artifacts=generated.model_artifacts,
                trainer_source=generated.training_source_code,
                predictor_source=generated.inference_source_code,
            )

            return model

        except Exception as e:
            logger.error(f"Error during model building: {str(e)}")
            raise e

    def _notify_callbacks(
        self,
        callbacks,
        event,
        intent,
        input_schema,
        output_schema,
        training_data,
        model_identifier=None,
        model_state=None,
        final_metric=None,
        final_artifacts=None,
        trainer_source=None,
        predictor_source=None,
    ):
        """Helper to notify callbacks with consistent error handling."""
        for callback in callbacks:
            try:
                method_name = f"on_{event}"
                if hasattr(callback, method_name):
                    getattr(callback, method_name)(
                        BuildStateInfo(
                            intent=intent,
                            input_schema=input_schema,
                            output_schema=output_schema,
                            provider=self.provider_config.tool_provider,
                            datasets=training_data,
                            model_identifier=model_identifier,
                            model_state=model_state,
                            final_metric=final_metric,
                            final_artifacts=final_artifacts,
                            trainer_source=trainer_source,
                            predictor_source=predictor_source,
                        )
                    )
            except Exception as e:
                logger.warning(f"Error in callback {callback.__class__.__name__}.{method_name}: {str(e)[:50]}")
