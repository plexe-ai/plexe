"""
This module defines the `Model` class, which represents a machine learning model.

A `Model` is characterized by a natural language description of its intent, structured input and output schemas.
This class provides methods for building the model, making predictions, and inspecting its state, metadata, and metrics.

Key Features:
- Intent: A natural language description of the model's purpose.
- Input/Output Schema: Defines the structure and types of inputs and outputs.
- Mutable State: Tracks the model's lifecycle, training metrics, and metadata.
- Build Process: Integrates solution generation with callbacks.
- Chain of Thought: Captures the reasoning steps of the model building process.

Example:
>>>    model = Model(
>>>        intent="Given a dataset of house features, predict the house price.",
>>>        output_schema=create_model("output", **{"price": float}),
>>>        input_schema=create_model("input", **{
>>>            "bedrooms": int,
>>>            "bathrooms": int,
>>>            "square_footage": float
>>>        })
>>>    )
>>>
>>>    model.build(
>>>        datasets=[pd.read_csv("houses.csv")], 
>>>        provider="openai:gpt-4o-mini", 
>>>        max_iterations=10,
>>>        chain_of_thought=True  # Enable chain of thought logging
>>>    )
>>>
>>>    prediction = model.predict({"bedrooms": 3, "bathrooms": 2, "square_footage": 1500.0})
>>>    print(prediction)
"""

import logging
import os
import uuid
import warnings
from datetime import datetime
from typing import Dict, List, Type, Any
from deprecated import deprecated

import pandas as pd
from pydantic import BaseModel

from plexe.callbacks import Callback
from plexe.core.interfaces.predictor import Predictor
from plexe.core.object_registry import ObjectRegistry
from plexe.core.state import ModelState  # Import from core package
from plexe.datasets import DatasetGenerator
from plexe.internal.common.datasets.interface import Dataset
from plexe.internal.common.provider import ProviderConfig
from plexe.internal.common.utils.model_utils import calculate_model_size, format_code_snippet
from plexe.internal.common.utils.pydantic_utils import map_to_basemodel, format_schema
from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.entities.description import (
    ModelDescription,
    SchemaInfo,
    ImplementationInfo,
    PerformanceInfo,
    CodeInfo,
)
from plexe.internal.models.entities.metric import Metric

logger = logging.getLogger(__name__)


class Model:
    """
    Represents a model that transforms inputs to outputs according to a specified intent.

    A `Model` is defined by a human-readable description of its expected intent, as well as structured
    definitions of its input schema and output schema.

    Attributes:
        intent (str): A human-readable, natural language description of the model's expected intent.
        output_schema (dict): A mapping of output key names to their types.
        input_schema (dict): A mapping of input key names to their types.

    Example:
        model = Model(
            intent="Given a dataset of house features, predict the house price.",
            output_schema=create_model("output_schema", **{"price": float}),
            input_schema=create_model("input_schema", **{
                "bedrooms": int,
                "bathrooms": int,
                "square_footage": float,
            })
        )
    """

    def __init__(
        self,
        intent: str,
        input_schema: Type[BaseModel] | Dict[str, type] = None,
        output_schema: Type[BaseModel] | Dict[str, type] = None,
        distributed: bool = False,
    ):
        """
        Initialise a model with a natural language description of its intent, as well as
        structured definitions of its input schema and output schema.

        :param intent: A human-readable, natural language description of the model's expected intent.
        :param input_schema: a pydantic model or dictionary defining the input schema
        :param output_schema: a pydantic model or dictionary defining the output schema
        :param distributed: Whether to use distributed training with Ray if available.
        """
        # todo: analyse natural language inputs and raise errors where applicable

        # The model's identity is defined by these fields
        self.intent: str = intent
        self.input_schema: Type[BaseModel] = map_to_basemodel("in", input_schema) if input_schema else None
        self.output_schema: Type[BaseModel] = map_to_basemodel("out", output_schema) if output_schema else None
        self.training_data: Dict[str, Dataset] = dict()
        self.distributed: bool = distributed

        # The model's mutable state is defined by these fields
        self.state: ModelState = ModelState.DRAFT
        self.predictor: Predictor | None = None
        self.trainer_source: str | None = None
        self.predictor_source: str | None = None
        self.feature_transformer_source: str | None = None
        self.dataset_splitter_source: str | None = None
        self.testing_source: str | None = None
        self.evaluation_report: Dict | None = None
        self.artifacts: List[Artifact] = []
        self.metric: Metric | None = None
        self.metadata: Dict[str, Any] = dict()  # todo: initialise metadata, etc

        # Registries used to make datasets, artifacts and other objects available across the system
        self.object_registry = ObjectRegistry()

        # Setup the working directory and unique identifiers
        self.identifier: str = f"model-{abs(hash(self.intent))}-{str(uuid.uuid4())}"
        self.run_id = f"run-{datetime.now().isoformat()}".replace(":", "-").replace(".", "-")
        self.working_dir = f"./workdir/{self.run_id}/"
        os.makedirs(self.working_dir, exist_ok=True)

    @deprecated(reason="Use ModelBuilder.build() instead", version="0.23.0")
    def build(
        self,
        datasets: List[pd.DataFrame | DatasetGenerator],
        provider: str | ProviderConfig = "openai/gpt-4o-mini",
        timeout: int = None,
        max_iterations: int = None,
        run_timeout: int = 1800,
        callbacks: List[Callback] = None,
        verbose: bool = False,
        # resume: bool = False,
        enable_checkpointing: bool = False,
    ) -> None:
        """
        Build the model using the provided dataset and optional data generation configuration.

        DEPRECATED: This interface is deprecated. Use ModelBuilder.build() instead:

            from plexe import ModelBuilder
            builder = ModelBuilder(provider=provider, verbose=verbose, distributed=distributed)
            model = builder.build(intent=intent, datasets=datasets, ...)

        :param datasets: the datasets to use for training the model
        :param provider: the provider to use for model building, either a string or a ProviderConfig
                         for granular control of which models to use for different agent roles
        :param timeout: maximum total time in seconds to spend building the model (all iterations combined)
        :param max_iterations: maximum number of iterations to spend building the model
        :param run_timeout: maximum time in seconds for each individual model training run
        :param callbacks: list of callbacks to notify during the model building process
        :param verbose: whether to display detailed agent logs during model building (default: False)
        :param enable_checkpointing: whether to enable automatic checkpointing (default: True)
        :return:
        """
        warnings.warn(
            "Model.build() is deprecated. Use ModelBuilder.build() instead:\n\n"
            "    from plexe import ModelBuilder\n"
            "    builder = ModelBuilder(provider=provider, verbose=verbose, distributed=distributed)\n"
            "    model = builder.build(intent=intent, datasets=datasets, ...)\n",
            DeprecationWarning,
            stacklevel=2,
        )

        # Import here to avoid circular dependency
        from plexe.model_builder import ModelBuilder

        # Create builder and delegate to it
        builder = ModelBuilder(
            provider=provider,
            verbose=verbose,
            distributed=self.distributed,
            working_dir=self.working_dir,
        )

        # Build the model using ModelBuilder
        built_model = builder.build(
            intent=self.intent,
            datasets=datasets,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            timeout=timeout,
            max_iterations=max_iterations,
            run_timeout=run_timeout,
            callbacks=callbacks,
            enable_checkpointing=enable_checkpointing,
        )

        # Copy all results back to self to maintain backwards compatibility
        for attr in [
            "identifier",
            "input_schema",
            "output_schema",
            "predictor",
            "trainer_source",
            "predictor_source",
            "feature_transformer_source",
            "dataset_splitter_source",
            "testing_source",
            "evaluation_report",
            "artifacts",
            "metric",
            "training_data",
        ]:
            setattr(self, attr, getattr(built_model, attr))

        self.metadata.update(built_model.metadata)
        self.state = ModelState.READY

    def predict(self, x: Dict[str, Any], validate_input: bool = False, validate_output: bool = False) -> Dict[str, Any]:
        """
        Call the model with input x and return the output.
        :param x: input to the model
        :param validate_input: whether to validate the input against the input schema
        :param validate_output: whether to validate the output against the output schema
        :return: output of the model
        """
        if self.state != ModelState.READY:
            raise RuntimeError("The model is not ready for predictions.")
        try:
            if validate_input:
                self.input_schema.model_validate(x)
            y = self.predictor.predict(x)
            if validate_output:
                self.output_schema.model_validate(y)
            return y
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}") from e

    def get_state(self) -> ModelState:
        """
        Return the current state of the model.
        :return: the current state of the model
        """
        return self.state

    def get_metadata(self) -> dict:
        """
        Return metadata about the model.
        :return: metadata about the model
        """
        return self.metadata

    def get_metrics(self) -> dict:
        """
        Return metrics about the model.
        :return: metrics about the model
        """
        return None if self.metric is None else {self.metric.name: self.metric.value}

    def describe(self) -> ModelDescription:
        """
        Return a structured description of the model.

        :return: A ModelDescription object with various methods like to_dict(), as_text(),
                as_markdown(), to_json() for different output formats
        """
        # Create schema info
        schemas = SchemaInfo(
            input=format_schema(self.input_schema),
            output=format_schema(self.output_schema),
        )

        # Create implementation info
        implementation = ImplementationInfo(
            framework=self.metadata.get("framework", "Unknown"),
            model_type=self.metadata.get("model_type", "Unknown"),
            artifacts=[a.name for a in self.artifacts],
            size=calculate_model_size(self.artifacts),
        )

        # Create performance info
        # Convert Metric objects to string representation for JSON serialization
        metrics_dict = {}
        if hasattr(self.metric, "value") and hasattr(self.metric, "name"):  # Check if it's a Metric object
            metrics_dict[self.metric.name] = str(self.metric.value)

        performance = PerformanceInfo(
            metrics=metrics_dict,
            training_data_info={
                name: {
                    "modality": data.structure.modality,
                    "features": data.structure.features,
                    "structure": data.structure.details,
                }
                for name, data in self.training_data.items()
            },
        )

        # Create code info
        code = CodeInfo(
            training=format_code_snippet(self.trainer_source),
            prediction=format_code_snippet(self.predictor_source),
            feature_transformations=format_code_snippet(self.feature_transformer_source),
        )

        # Assemble and return the complete model description
        return ModelDescription(
            id=self.identifier,
            state=self.state.value,
            intent=self.intent,
            schemas=schemas,
            implementation=implementation,
            performance=performance,
            code=code,
            training_date=self.metadata.get("creation_date", "Unknown"),
            rationale=self.metadata.get("selection_rationale", "Unknown"),
            provider=self.metadata.get("provider", "Unknown"),
            task_type=self.metadata.get("task_type", "Unknown"),
            domain=self.metadata.get("domain", "Unknown"),
            behavior=self.metadata.get("behavior", "Unknown"),
            preprocessing_summary=self.metadata.get("preprocessing_summary", "Unknown"),
            architecture_summary=self.metadata.get("architecture_summary", "Unknown"),
            training_procedure=self.metadata.get("training_procedure", "Unknown"),
            evaluation_metric=self.metadata.get("evaluation_metric", "Unknown"),
            inference_behavior=self.metadata.get("inference_behavior", "Unknown"),
            strengths=self.metadata.get("strengths", "Unknown"),
            limitations=self.metadata.get("limitations", "Unknown"),
        )
