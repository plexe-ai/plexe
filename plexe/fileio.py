"""
This module provides file I/O utilities for saving and loading models to and from archive files.

It serves as a user-facing API that delegates implementation details to core.storage.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

# Import core implementations
from plexe.core.storage import (
    _save_model_to_tar,
    _load_model_data_from_tar,
    _save_checkpoint_to_tar,
    _load_checkpoint_data_from_tar,
    _list_checkpoint_files,
    _delete_checkpoint_file,
    _clear_checkpoint_files,
)

# Import other required types
from plexe.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from plexe.internal.models.entities.artifact import Artifact

logger = logging.getLogger(__name__)


def save_model(model: Any, path: str | Path) -> str:
    """
    Save a model to a tar archive.

    Args:
        model: The model to save
        path: Path where to save the model

    Returns:
        Path where the model was saved

    Raises:
        ValueError: If the model cannot be saved or the path is invalid
    """
    try:
        return _save_model_to_tar(model, path)
    except Exception as e:
        logger.error(f"Error saving model to {path}: {e}")
        raise ValueError(f"Failed to save model: {str(e)}") from e


def load_model(path: str | Path):
    """
    Load a model from a tar archive.

    Args:
        path: Path to the model archive

    Returns:
        A Model instance

    Raises:
        ValueError: If the model file doesn't exist or is invalid
    """
    # Import here to avoid circular imports
    from plexe.models import Model

    try:
        # Load model data using the core implementation
        model_data = _load_model_data_from_tar(path)

        # Create model instance with schemas already processed by core
        model = Model(
            intent=model_data["intent"],
            input_schema=model_data["input_schema"],
            output_schema=model_data["output_schema"],
        )

        # Set model state and properties
        model.state = model_data["state"]
        model.metadata = model_data["metadata"]
        model.identifier = model_data["identifier"]
        model.trainer_source = model_data["trainer_source"]
        model.predictor_source = model_data["predictor_source"]
        # Set additional properties if available; these are optional for backward compatibility
        model.feature_transformer_source = model_data.get("feature_transformer_source", None)
        model.dataset_splitter_source = model_data.get("dataset_splitter_source", None)
        model.testing_source = model_data.get("testing_source", None)
        model.evaluation_report = model_data.get("evaluation_report", None)

        # Process metrics data if available
        metrics_data = model_data["metrics_data"]
        if metrics_data:
            comparator = MetricComparator(
                comparison_method=ComparisonMethod(metrics_data["comparison_method"]), target=metrics_data["target"]
            )
            model.metric = Metric(name=metrics_data["name"], value=metrics_data["value"], comparator=comparator)

        # Add EDA reports to metadata if found
        if "eda_markdown_reports" in model_data:
            model.metadata["eda_markdown_reports"] = model_data["eda_markdown_reports"]

        # Process artifacts
        artifact_handles = []
        for artifact_item in model_data["artifact_data"]:
            artifact_handles.append(Artifact.from_data(artifact_item["name"], artifact_item["data"]))

        model.artifacts = artifact_handles

        # Load predictor if source code is available
        if model.predictor_source:
            import types

            predictor_module = types.ModuleType("predictor")
            exec(model.predictor_source, predictor_module.__dict__)
            model.predictor = predictor_module.PredictorImplementation(artifact_handles)

        logger.debug(f"Model successfully loaded from {path}")
        return model

    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}")
        raise ValueError(f"Failed to load model: {str(e)}") from e


def save_checkpoint(model: Any, iteration: int, path: Optional[str | Path] = None) -> str:
    """
    Save a model checkpoint to a tar archive.

    Args:
        model: The model to checkpoint
        iteration: Current iteration number
        path: Optional custom path

    Returns:
        Path where the checkpoint was saved

    Raises:
        ValueError: If the checkpoint cannot be saved or the path is invalid
    """
    try:
        return _save_checkpoint_to_tar(model, iteration, path)
    except Exception as e:
        logger.error(f"Error saving checkpoint (iteration {iteration}): {e}")
        raise ValueError(f"Failed to save checkpoint: {str(e)}") from e


def load_checkpoint(
    checkpoint_path: Optional[str | Path] = None, model_id: Optional[str] = None, latest: bool = False
) -> Any:
    """
    Load a model from a checkpoint.

    This function loads a model from a checkpoint file, which can then be used
    to resume a previously interrupted build process.

    Args:
        checkpoint_path: Direct path to a checkpoint file
        model_id: Model identifier to find checkpoints for
        latest: If True and model_id is provided, loads the latest checkpoint for that model

    Returns:
        Model instance initialized from the checkpoint

    Raises:
        ValueError: If no checkpoint could be found or if the parameters are invalid
    """
    # Import here to avoid circular imports
    from plexe.models import Model

    try:
        # Parameter validation
        if checkpoint_path is None and model_id is None:
            raise ValueError("Either checkpoint_path or model_id must be provided")

        if checkpoint_path is not None and model_id is not None:
            raise ValueError("Only one of checkpoint_path or model_id should be provided")

        # If model_id is provided, find relevant checkpoints
        if model_id is not None:
            checkpoints = list_checkpoints(model_id)
            if not checkpoints:
                raise ValueError(f"No checkpoints found for model_id '{model_id}'")

            if latest:
                # Sort by timestamp (descending) and take the first one
                checkpoint_path = sorted(checkpoints, reverse=True)[0]
            else:
                # If not latest, we need an exact path
                raise ValueError("When using model_id without latest=True, please specify a specific checkpoint_path")

        # Load the checkpoint data from tar archive (schemas are already processed)
        checkpoint_data = _load_checkpoint_data_from_tar(checkpoint_path)

        # Create a new model with the basic information from the checkpoint
        model = Model(
            intent=checkpoint_data["intent"],
            input_schema=checkpoint_data["input_schema"],
            output_schema=checkpoint_data["output_schema"],
        )

        # Update model with checkpoint data
        model.identifier = checkpoint_data["identifier"]
        model.state = checkpoint_data["state"]
        model.metadata.update(checkpoint_data["metadata"])

        # Store the checkpoint data for resumption
        model._checkpoint_data = checkpoint_data

        logger.info(f"Model loaded from checkpoint {checkpoint_path}")
        return model

    except Exception as e:
        if isinstance(e, ValueError):
            # Preserve specific ValueError messages
            raise
        logger.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
        raise ValueError(f"Failed to load checkpoint: {str(e)}") from e


def list_checkpoints(model_id: Optional[str] = None) -> List[str]:
    """
    List available checkpoints.

    Args:
        model_id: Optional model identifier to filter checkpoints

    Returns:
        List of checkpoint paths

    Raises:
        ValueError: If there's an issue accessing the checkpoint directory
    """
    try:
        return _list_checkpoint_files(model_id)
    except Exception as e:
        logger.error(f"Error listing checkpoints: {e}")
        raise ValueError(f"Failed to list checkpoints: {str(e)}") from e


def delete_checkpoint(path: str | Path) -> bool:
    """
    Delete a specific checkpoint.

    Args:
        path: Path to the checkpoint to delete

    Returns:
        True if deletion was successful, False otherwise

    Raises:
        ValueError: If there's an issue accessing the file
    """
    try:
        return _delete_checkpoint_file(path)
    except Exception as e:
        logger.error(f"Error deleting checkpoint {path}: {e}")
        raise ValueError(f"Failed to delete checkpoint: {str(e)}") from e


def clear_checkpoints(model_id: Optional[str] = None, older_than_days: Optional[int] = None) -> int:
    """
    Clear checkpoints based on filter criteria.

    Args:
        model_id: Optional model identifier to filter checkpoints
        older_than_days: Optional age in days to filter checkpoints

    Returns:
        Number of checkpoints deleted

    Raises:
        ValueError: If there's an issue accessing or deleting the files
    """
    try:
        return _clear_checkpoint_files(model_id, older_than_days)
    except Exception as e:
        logger.error(f"Error clearing checkpoints: {e}")
        raise ValueError(f"Failed to clear checkpoints: {str(e)}") from e
