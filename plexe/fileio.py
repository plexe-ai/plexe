"""
This module provides file I/O utilities for saving and loading models to and from archive files.
"""

import io
import json
import logging
import pickle
import tarfile
import types
from pathlib import Path

from plexe.models import Model, ModelState
from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.common.utils.pydantic_utils import map_to_basemodel
from plexe.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod

logger = logging.getLogger(__name__)


def save_model(model: Model, path: str | Path) -> str:
    """
    Save a model to a tar archive.

    :param model: The model to save
    :param path: Optional custom path. If not provided, saves to smolcache/models/
    :return: Path where the model was saved
    """
    #     Archive structure:
    #     - metadata/
    #         - intent.txt
    #         - state.txt
    #         - metrics.json
    #         - metadata.json
    #     - schemas/
    #         - input_schema.json
    #         - output_schema.json
    #     - code/
    #         - trainer.py
    #         - predictor.py
    #     - artifacts/
    #         - [model files]

    # Ensure .tar.gz extension
    if not str(path).endswith(".tar.gz"):
        raise ValueError("Path must end with .tar.gz")

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(path, "w:gz") as tar:
            metrics_data = {}
            if model.metric:
                metrics_data = {
                    "name": model.metric.name,
                    "value": model.metric.value,
                    "comparison_method": model.metric.comparator.comparison_method.value,
                    "target": model.metric.comparator.target,
                }

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
                    info = tarfile.TarInfo(f"metadata/{key}.json")
                    content = json.dumps(value, indent=2, default=str).encode("utf-8")
                else:
                    info = tarfile.TarInfo(f"metadata/{key}.txt")
                    content = str(value).encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            for name, schema in [("input_schema", model.input_schema), ("output_schema", model.output_schema)]:
                schema_dict = {name: field.annotation.__name__ for name, field in schema.model_fields.items()}
                info = tarfile.TarInfo(f"schemas/{name}.json")
                content = json.dumps(schema_dict, default=str).encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            if model.trainer_source:
                info = tarfile.TarInfo("code/trainer.py")
                content = model.trainer_source.encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            if model.predictor_source:
                info = tarfile.TarInfo("code/predictor.py")
                content = model.predictor_source.encode("utf-8")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

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

            if model.constraints:
                info = tarfile.TarInfo("metadata/constraints.pkl")
                content = pickle.dumps(model.constraints)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))

            # Save EDA markdown reports if available
            if "eda_markdown_reports" in model.metadata and model.metadata["eda_markdown_reports"]:
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


def load_model(path: str | Path) -> Model:
    """
    Instantiate a model from a tar archive.

    :param path: path to tar archive
    :return: the loaded model
    :raises ValueError: If model is not found
    :raises Exception: If there are errors during loading
    """
    if not Path(path).exists():
        raise ValueError(f"Model not found: {path}")

    # TODO: the mapping between model internals and the archive structure should be defined once somewhere
    # for example, in this module but outside the load/save functions
    try:
        with tarfile.open(path, "r:gz") as tar:
            # Extract metadata
            intent = tar.extractfile("metadata/intent.txt").read().decode("utf-8")
            state = ModelState(tar.extractfile("metadata/state.txt").read().decode("utf-8"))
            metrics_data = json.loads(tar.extractfile("metadata/metrics.json").read().decode("utf-8"))
            metadata = json.loads(tar.extractfile("metadata/metadata.json").read().decode("utf-8"))
            identifier = tar.extractfile("metadata/identifier.txt").read().decode("utf-8")

            # Extract schema information
            input_schema_dict = json.loads(tar.extractfile("schemas/input_schema.json").read().decode("utf-8"))
            output_schema_dict = json.loads(tar.extractfile("schemas/output_schema.json").read().decode("utf-8"))

            # Extract code if available
            trainer_source = None
            if "code/trainer.py" in [m.name for m in tar.getmembers()]:
                trainer_source = tar.extractfile("code/trainer.py").read().decode("utf-8")

            predictor_source = None
            if "code/predictor.py" in [m.name for m in tar.getmembers()]:
                predictor_source = tar.extractfile("code/predictor.py").read().decode("utf-8")

            # Extract constraints if available
            constraints = []
            if "metadata/constraints.pkl" in [m.name for m in tar.getmembers()]:
                constraints = pickle.loads(tar.extractfile("metadata/constraints.pkl").read())

            # Load EDA markdown reports if available
            eda_markdown_reports = {}
            for member in tar.getmembers():
                if member.name.startswith("metadata/eda_report_") and member.name.endswith(".md"):
                    dataset_name = member.name.replace("metadata/eda_report_", "").replace(".md", "")
                    report_content = tar.extractfile(member).read().decode("utf-8")
                    eda_markdown_reports[dataset_name] = report_content

            # Get handles for all model artifacts
            artifact_handles = []
            for member in tar.getmembers():
                if member.name.startswith("artifacts/") and not member.isdir():
                    file_data = tar.extractfile(member)
                    if file_data:
                        artifact_handles.append(Artifact.from_data(Path(member.name).name, file_data.read()))

            # Reconstruct Metric object if metrics data exists
            metrics = None
            if metrics_data:
                comparator = MetricComparator(
                    comparison_method=ComparisonMethod(metrics_data["comparison_method"]), target=metrics_data["target"]
                )
                metrics = Metric(name=metrics_data["name"], value=metrics_data["value"], comparator=comparator)

            def type_from_name(type_name: str) -> type:
                type_map = {"str": str, "int": int, "float": float, "bool": bool}
                return type_map[type_name]

            # Create schemas from the schema dictionaries
            input_schema = map_to_basemodel(
                "InputSchema", {name: type_from_name(type_name) for name, type_name in input_schema_dict.items()}
            )
            output_schema = map_to_basemodel(
                "OutputSchema", {name: type_from_name(type_name) for name, type_name in output_schema_dict.items()}
            )

            # Create the model instance
            model = Model(
                intent=intent, input_schema=input_schema, output_schema=output_schema, constraints=constraints
            )
            model.state = state
            model.metric = metrics
            model.metadata = metadata
            model.identifier = identifier
            model.trainer_source = trainer_source

            # Add to the metadata if reports were found
            if eda_markdown_reports:
                model.metadata["eda_markdown_reports"] = eda_markdown_reports
            model.predictor_source = predictor_source

            if predictor_source:
                predictor_module = types.ModuleType("predictor")
                exec(predictor_source, predictor_module.__dict__)
                model.predictor = predictor_module.PredictorImplementation(artifact_handles)

            model.artifacts = artifact_handles

            logger.debug(f"Model successfully loaded from {path}")
            return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
