"""
Tools for providing context to agents for code generation tasks.
"""

import json
import logging
from typing import Dict, Any, List, Callable

from pydantic import BaseModel
from smolagents import tool

from plexe.config import code_templates
from plexe.internal.common.provider import Provider
from plexe.core.object_registry import ObjectRegistry
from plexe.internal.models.entities.code import Code

logger = logging.getLogger(__name__)


def get_inference_context_tool(llm_to_use: str) -> Callable:
    """Returns a tool function to get inference context with the model ID pre-filled."""

    @tool
    def get_inference_context() -> Dict[str, Any]:
        """
        Provides comprehensive context needed for generating inference code. Use this tool to retrieve
        a summary of the training code, schemas, expected inputs for the purpose of planning the inference
        code.

        Returns:
            A dictionary containing all context needed for inference code generation
        """
        object_registry = ObjectRegistry()

        # Retrieve training code
        try:
            training_code = object_registry.get(Code, "best_performing_training_code").code
        except Exception as e:
            raise ValueError(f"Training code with ID 'best_performing_training_code' not found: {str(e)}")

        # Retrieve schemas
        try:
            input_schema = object_registry.get(dict, "input_schema")
            output_schema = object_registry.get(dict, "output_schema")
        except Exception as e:
            raise ValueError(f"Failed to retrieve schemas from registry: {str(e)}")

        # Retrieve input sample
        try:
            input_sample = object_registry.get(list, "predictor_input_sample")
        except Exception as e:
            raise ValueError(f"Failed to retrieve input sample: {str(e)}")

        # Extract artifacts
        try:
            artifact_names = _extract_artifacts(llm_to_use, training_code)
            object_registry.register(list, "model_artifact_names", artifact_names, overwrite=True, immutable=True)
        except Exception as e:
            raise ValueError(f"Failed to extract artifacts from training code: {str(e)}")

        return {
            "training_code": training_code,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "predictor_interface": code_templates.predictor_interface,
            "predictor_template": code_templates.predictor_template,
            "input_sample": input_sample,
            "artifact_names": artifact_names,
        }

    return get_inference_context


def _extract_artifacts(llm_to_use: str, code: str) -> List[str]:
    """Extract model artifact names from training code using LLM"""

    class ArtifactResponse(BaseModel):
        artifact_names: List[str]

    try:
        provider = Provider(llm_to_use)

        names = json.loads(
            provider.query(
                "You are a code analysis assistant.",
                (
                    "Extract the names of all saved ML model artifacts from the following code. "
                    "The artifacts are usually saved using a function like torch.save, joblib.dump, or pickle.dump. "
                    "Any model artifact that is saved by the training script must be included in the output so that "
                    "it can be used in the inference code. DO NOT modify the artifact names from the script.\n\n"
                    "Here is the training code:\n"
                    f"```python\n{code}\n```"
                ),
                ArtifactResponse,
            )
        )["artifact_names"]
        return names
    except Exception as e:
        raise RuntimeError(f"Artifact extraction failed: {e}") from e
