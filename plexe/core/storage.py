"""
Core storage functions for model and checkpoint persistence.

This module contains implementation logic for saving and loading models and checkpoints,
without direct dependencies on the Model class. It provides a foundation for the
fileio module, breaking the circular dependency between models.py and fileio.py.
"""

import io
import json
import yaml
import logging
import tarfile
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Type
import numpy as np

from plexe.core.state import ModelState
from plexe.config import config
from plexe.internal.common.utils.pydantic_utils import map_to_basemodel

logger = logging.getLogger(__name__)

# Type variable for generic model type
M = TypeVar("M")


class FallbackNoneLoader(yaml.SafeLoader):
    pass


def fallback_to_none(loader, tag_suffix, node):
    return None


FallbackNoneLoader.add_multi_constructor("", fallback_to_none)


def _convert_to_native_types(obj):
    """Recursively convert numpy types and other non-native types to Python native types.

    Falls back to string representation for any type that can't be converted.
    """
    try:
        # Handle NumPy types specifically
        if isinstance(obj, np.generic):
            return _convert_to_native_types(obj.item())
        elif isinstance(obj, np.ndarray):
            return _convert_to_native_types(obj.tolist())
        # Handle common collections
        elif isinstance(obj, dict):
            return {k: _convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_convert_to_native_types(item) for item in obj)
        # Native types are already safe
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        # For other types, raise error to trigger fallback
        else:
            raise TypeError(f"Unsupported type: {type(obj).__name__}")
    except Exception as e:
        # Fallback: if anything goes wrong, use JSON serialization with sanitization
        logger.warning(
            f"Failed to convert object of type {type(obj).__name__} for serialization: {e}. Using string representation."
        )
        return json.loads(json.dumps(obj, skipkeys=True, default=str))


def _load_yaml_or_json_from_tar(tar, yaml_path: str, json_path: str):
    """Load from YAML if available, fallback to JSON for backward compatibility."""
    members = [m.name for m in tar.getmembers()]
    if yaml_path in members:
        content = tar.extractfile(yaml_path).read().decode("utf-8")
        return yaml.load(content, Loader=FallbackNoneLoader)
    elif json_path in members:
        content = tar.extractfile(json_path).read().decode("utf-8")
        return json.loads(content)
    else:
        raise FileNotFoundError(f"Neither {yaml_path} nor {json_path} found in archive")


def _save_model_to_tar(model: Any, path: str | Path) -> str:
    """
    Core implementation of saving a model to a tar archive.

    Args:
        model: The model to save (any object with required attributes)
        path: Path where to save the model

    Returns:
        Path where the model was saved
    """
    # Ensure .tar.gz extension
    if not str(path).endswith(".tar.gz"):
        raise ValueError("Path must end with .tar.gz")

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(path, "w:gz") as tar:
            # Get metrics data if available
            metrics_data = {}
            if hasattr(model, "metric") and model.metric:
                metrics_data = {
                    "name": model.metric.name,
                    "value": _convert_to_native_types(model.metric.value),
                    "comparison_method": model.metric.comparator.comparison_method.value,
                    "target": _convert_to_native_types(model.metric.comparator.target),
                }

            # Gather metadata
            metadata = {
                "intent": model.intent,
                "state": model.state.value,
                "metrics": metrics_data,
                "metadata": model.metadata,
                "identifier": model.identifier,
            }

            # Save each metadata item separately
            for key, value in metadata.items():
                if key in ["metrics", "metadata"]:
                    info = tarfile.TarInfo(f"metadata/{key}.yaml")
                    # Ensure all values are serializable
                    safe_value = _convert_to_native_types(value)
                    content = yaml.safe_dump(safe_value, default_flow_style=False).encode("utf-8")
                else:
                    info = tarfile.TarInfo(f"metadata/{key}.txt")
                    content = str(value).encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save schemas
            for name, schema in [("input_schema", model.input_schema), ("output_schema", model.output_schema)]:
                schema_dict = {name: field.annotation.__name__ for name, field in schema.model_fields.items()}
                info = tarfile.TarInfo(f"schemas/{name}.yaml")
                content = yaml.safe_dump(schema_dict, default_flow_style=False).encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save trainer source if available
            if hasattr(model, "trainer_source") and model.trainer_source:
                info = tarfile.TarInfo("code/trainer.py")
                content = model.trainer_source.encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save predictor source if available
            if hasattr(model, "predictor_source") and model.predictor_source:
                info = tarfile.TarInfo("code/predictor.py")
                content = model.predictor_source.encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save feature transformer source if available
            if hasattr(model, "feature_transformer_source") and model.feature_transformer_source:
                info = tarfile.TarInfo("code/feature_transformer.py")
                content = model.feature_transformer_source.encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save dataset splitter source if available
            if hasattr(model, "dataset_splitter_source") and model.dataset_splitter_source:
                info = tarfile.TarInfo("code/dataset_splitter.py")
                content = model.dataset_splitter_source.encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save testing source if available
            if hasattr(model, "testing_source") and model.testing_source:
                info = tarfile.TarInfo("code/testing.py")
                content = model.testing_source.encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save evaluation report if available
            if hasattr(model, "evaluation_report") and model.evaluation_report:
                info = tarfile.TarInfo("metadata/evaluation_report.yaml")
                # Convert numpy types to native Python types before serialization
                evaluation_report_native = _convert_to_native_types(model.evaluation_report)
                content = yaml.safe_dump(evaluation_report_native, default_flow_style=False).encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save artifacts
            if hasattr(model, "artifacts"):
                for artifact in model.artifacts:
                    arc_name = f"artifacts/{Path(artifact.name).as_posix()}"
                    info = tarfile.TarInfo(arc_name)

                    if artifact.is_path():
                        with open(artifact.path, "rb") as f:
                            content = f.read()
                    elif artifact.is_handle():
                        content = artifact.handle.read()
                    else:
                        content = artifact.data

                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))

            # Save EDA markdown reports if available
            if (
                hasattr(model, "metadata")
                and "eda_markdown_reports" in model.metadata
                and model.metadata["eda_markdown_reports"]
            ):
                for dataset_name, report_markdown in model.metadata["eda_markdown_reports"].items():
                    info = tarfile.TarInfo(f"metadata/eda_report_{dataset_name}.md")
                    content = report_markdown.encode("utf-8")
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        if Path(path).exists():
            Path(path).unlink()
        raise

    logger.info(f"Model saved to {path}")
    return str(path)


def _load_model_data_from_tar(path: str | Path) -> Dict[str, Any]:
    """
    Core implementation of loading model data from a tar archive.

    Args:
        path: Path to the model archive

    Returns:
        Dictionary with model data to reconstruct a Model instance
    """
    if not Path(path).exists():
        raise ValueError(f"Model not found: {path}")

    try:
        with tarfile.open(path, "r:gz") as tar:
            # Extract metadata
            intent = tar.extractfile("metadata/intent.txt").read().decode("utf-8")
            state = ModelState(tar.extractfile("metadata/state.txt").read().decode("utf-8"))
            metrics_data = _load_yaml_or_json_from_tar(tar, "metadata/metrics.yaml", "metadata/metrics.json")
            metadata = _load_yaml_or_json_from_tar(tar, "metadata/metadata.yaml", "metadata/metadata.json")
            identifier = tar.extractfile("metadata/identifier.txt").read().decode("utf-8")

            # Extract schema information
            input_schema_dict = _load_yaml_or_json_from_tar(
                tar, "schemas/input_schema.yaml", "schemas/input_schema.json"
            )
            output_schema_dict = _load_yaml_or_json_from_tar(
                tar, "schemas/output_schema.yaml", "schemas/output_schema.json"
            )

            # Process schemas into Pydantic models
            input_schema = _process_schema_dict(input_schema_dict)
            output_schema = _process_schema_dict(output_schema_dict)

            # Extract code if available
            trainer_source = None
            if "code/trainer.py" in [m.name for m in tar.getmembers()]:
                trainer_source = tar.extractfile("code/trainer.py").read().decode("utf-8")

            predictor_source = None
            if "code/predictor.py" in [m.name for m in tar.getmembers()]:
                predictor_source = tar.extractfile("code/predictor.py").read().decode("utf-8")
                # FIXME: this is a hack required to ensure backwards compatibility with old models
                predictor_source = predictor_source.replace("plexe.internal.models.interfaces", "plexe.core.interfaces")

            feature_transformer_source = None
            if "code/feature_transformer.py" in [m.name for m in tar.getmembers()]:
                feature_transformer_source = tar.extractfile("code/feature_transformer.py").read().decode("utf-8")

            dataset_splitter_source = None
            if "code/dataset_splitter.py" in [m.name for m in tar.getmembers()]:
                dataset_splitter_source = tar.extractfile("code/dataset_splitter.py").read().decode("utf-8")

            testing_source = None
            if "code/testing.py" in [m.name for m in tar.getmembers()]:
                testing_source = tar.extractfile("code/testing.py").read().decode("utf-8")

            evaluation_report = None
            try:
                evaluation_report = _load_yaml_or_json_from_tar(
                    tar, "metadata/evaluation_report.yaml", "metadata/evaluation_report.json"
                )
            except FileNotFoundError:
                pass

            # Load EDA markdown reports if available
            eda_markdown_reports = {}
            for member in tar.getmembers():
                if member.name.startswith("metadata/eda_report_") and member.name.endswith(".md"):
                    dataset_name = member.name.replace("metadata/eda_report_", "").replace(".md", "")
                    report_content = tar.extractfile(member).read().decode("utf-8")
                    eda_markdown_reports[dataset_name] = report_content

            # Collect artifact data
            artifact_data = []
            for member in tar.getmembers():
                if member.name.startswith("artifacts/") and not member.isdir():
                    file_data = tar.extractfile(member)
                    if file_data:
                        artifact_data.append({"name": Path(member.name).name, "data": file_data.read()})

            # Prepare result dictionary with both raw schema dicts and processed schemas
            model_data = {
                "intent": intent,
                "state": state,
                "metadata": metadata,
                "identifier": identifier,
                "trainer_source": trainer_source,
                "predictor_source": predictor_source,
                "feature_transformer_source": feature_transformer_source,
                "dataset_splitter_source": dataset_splitter_source,
                "testing_source": testing_source,
                "evaluation_report": evaluation_report,
                "artifact_data": artifact_data,
                "input_schema_dict": input_schema_dict,
                "output_schema_dict": output_schema_dict,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "metrics_data": metrics_data,
            }

            # Add EDA reports if found
            if eda_markdown_reports:
                model_data["eda_markdown_reports"] = eda_markdown_reports

            logger.debug(f"Model data successfully loaded from {path}")
            return model_data

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def _save_checkpoint_to_tar(model: Any, iteration: int, path: Optional[str | Path] = None) -> str:
    """
    Core implementation of saving a checkpoint to a tar archive.

    Args:
        model: The model to checkpoint (any object with required attributes)
        iteration: Current iteration number
        path: Optional custom path

    Returns:
        Path where the checkpoint was saved
    """
    # Generate default path if not provided
    if path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path(config.file_storage.cache_dir) / config.file_storage.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{model.identifier}_{timestamp}.checkpoint.tar.gz"

    # Ensure .tar.gz extension
    if not str(path).endswith(".tar.gz"):
        raise ValueError("Path must end with .tar.gz")

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(path, "w:gz") as tar:
            # Add checkpoint marker
            info = tarfile.TarInfo("checkpoint.marker")
            content = b"1"
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

            # Add current iteration
            info = tarfile.TarInfo("metadata/iteration.txt")
            content = str(iteration).encode("utf-8")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))

            # Add model metadata
            metadata = {
                "intent": model.intent,
                "state": model.state.value,
                "metadata": model.metadata,
                "identifier": model.identifier,
            }

            # Save each metadata item separately
            for key, value in metadata.items():
                if key in ["metadata"]:
                    info = tarfile.TarInfo(f"metadata/{key}.yaml")
                    safe_value = _convert_to_native_types(value)
                    content = yaml.safe_dump(safe_value, default_flow_style=False).encode("utf-8")
                else:
                    info = tarfile.TarInfo(f"metadata/{key}.txt")
                    content = str(value).encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save schemas
            for name, schema in [("input_schema", model.input_schema), ("output_schema", model.output_schema)]:
                schema_dict = {name: field.annotation.__name__ for name, field in schema.model_fields.items()}
                info = tarfile.TarInfo(f"schemas/{name}.yaml")
                content = yaml.safe_dump(schema_dict, default_flow_style=False).encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save previously tried solutions from the ObjectRegistry if available
            solutions = []
            if hasattr(model, "object_registry") and hasattr(model.object_registry, "get_all_solutions"):
                solutions = model.object_registry.get_all_solutions()

            if solutions:
                solutions_data = json.dumps(solutions, default=str).encode("utf-8")
                info = tarfile.TarInfo("solutions/solutions.json")
                info.size = len(solutions_data)
                tar.addfile(info, io.BytesIO(solutions_data))

    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        if Path(path).exists():
            Path(path).unlink()
        raise

    logger.info(f"Checkpoint saved to {path}")
    return str(path)


def _load_checkpoint_data_from_tar(path: str | Path) -> Dict[str, Any]:
    """
    Core implementation of loading checkpoint data from a tar archive.

    Args:
        path: Path to the checkpoint archive

    Returns:
        Dictionary with checkpoint data
    """
    if not Path(path).exists():
        raise ValueError(f"Checkpoint not found: {path}")

    try:
        with tarfile.open(path, "r:gz") as tar:
            # Verify this is a checkpoint archive
            if "checkpoint.marker" not in [m.name for m in tar.getmembers()]:
                raise ValueError(f"Archive at {path} is not a valid checkpoint")

            # Extract metadata
            intent = tar.extractfile("metadata/intent.txt").read().decode("utf-8")
            state = ModelState(tar.extractfile("metadata/state.txt").read().decode("utf-8"))
            metadata = _load_yaml_or_json_from_tar(tar, "metadata/metadata.yaml", "metadata/metadata.json")
            identifier = tar.extractfile("metadata/identifier.txt").read().decode("utf-8")
            iteration = int(tar.extractfile("metadata/iteration.txt").read().decode("utf-8"))

            # Extract schema information
            input_schema_dict = _load_yaml_or_json_from_tar(
                tar, "schemas/input_schema.yaml", "schemas/input_schema.json"
            )
            output_schema_dict = _load_yaml_or_json_from_tar(
                tar, "schemas/output_schema.yaml", "schemas/output_schema.json"
            )

            # Process schemas into Pydantic models
            input_schema = _process_schema_dict(input_schema_dict)
            output_schema = _process_schema_dict(output_schema_dict)

            # Extract previous solutions if available
            solutions = []
            if "solutions/solutions.json" in [m.name for m in tar.getmembers()]:
                solutions_json = tar.extractfile("solutions/solutions.json").read().decode("utf-8")
                solutions = json.loads(solutions_json)

            # Prepare result dictionary with both raw schema dicts and processed schemas
            checkpoint_data = {
                "intent": intent,
                "identifier": identifier,
                "input_schema_dict": input_schema_dict,
                "output_schema_dict": output_schema_dict,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "state": state,
                "metadata": metadata,
                "iteration": iteration,
                "solutions": solutions,
            }

            logger.debug(f"Checkpoint successfully loaded from {path}")
            return checkpoint_data

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise


def _list_checkpoint_files(model_id: Optional[str] = None) -> List[str]:
    """
    Core implementation of listing checkpoint files.

    Args:
        model_id: Optional model identifier to filter checkpoints

    Returns:
        List of checkpoint paths
    """
    checkpoint_dir = Path(config.file_storage.cache_dir) / config.file_storage.checkpoint_dir
    if not checkpoint_dir.exists():
        return []

    checkpoints = list(checkpoint_dir.glob("*.checkpoint.tar.gz"))

    if model_id:
        checkpoints = [cp for cp in checkpoints if model_id in cp.stem]

    return [str(cp) for cp in checkpoints]


def _delete_checkpoint_file(path: str | Path) -> bool:
    """
    Core implementation of deleting a checkpoint file.

    Args:
        path: Path to the checkpoint to delete

    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        Path(path).unlink(missing_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error deleting checkpoint: {e}")
        return False


def _clear_checkpoint_files(model_id: Optional[str] = None, older_than_days: Optional[int] = None) -> int:
    """
    Core implementation of clearing checkpoints based on criteria.

    Args:
        model_id: Optional model identifier to filter checkpoints
        older_than_days: Optional age in days to filter checkpoints

    Returns:
        Number of checkpoints deleted
    """
    checkpoints = _list_checkpoint_files(model_id)
    deleted_count = 0

    # Apply age filter if provided
    if older_than_days is not None:
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=older_than_days)
        checkpoints = [cp for cp in checkpoints if Path(cp).stat().st_mtime < cutoff_time.timestamp()]

    # Delete matching checkpoints
    for cp in checkpoints:
        if _delete_checkpoint_file(cp):
            deleted_count += 1

    return deleted_count


def _process_schema_dict(schema_dict: Dict[str, str]) -> Type:
    """
    Process a schema dictionary to create a Pydantic model.

    Args:
        schema_dict: Dictionary mapping field names to type names

    Returns:
        A Pydantic model class
    """

    def type_from_name(type_name: str) -> type:
        # Map string type names to actual Python types
        type_map = {"str": str, "int": int, "float": float, "bool": bool}
        return type_map[type_name]

    # Create a Pydantic model from the schema dictionary
    schema_name = "Schema"  # This will be overridden by map_to_basemodel
    return map_to_basemodel(schema_name, {name: type_from_name(type_name) for name, type_name in schema_dict.items()})
