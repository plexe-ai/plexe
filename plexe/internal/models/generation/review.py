"""
This module provides functionality for reviewing and analyzing generated models.

It examines the solution plan, training code, and inference code to extract
metadata about the model, such as the framework used, model type, and provides
explanations about how the model works and why it's appropriate for the task.
"""

import json
import logging
from datetime import datetime
from typing import Dict

from pydantic import BaseModel

from plexe.config import prompt_templates
from plexe.internal.common.provider import Provider

logger = logging.getLogger(__name__)


class ModelReviewResponse(BaseModel):
    """
    Response model for the model review operation.
    """

    framework: str  # e.g. PyTorch, TensorFlow, Scikit-Learn
    model_type: str  # e.g. CNN, Transformer, RandomForest

    task_type: str  # e.g. classification, regression, generation
    domain: str  # e.g. NLP, computer vision, tabular, multimodal

    behavior: str  # what the model 'does', what relationship is it really learning

    preprocessing_summary: str  # high-level view of input processing pipeline
    architecture_summary: str  # concise summary of model architecture
    training_procedure: str  # brief description of optimizer, loss, epochs, etc.
    evaluation_metric: str  # list of metrics used for evaluation (e.g., accuracy, F1)

    inference_behavior: str  # how inference is performed, assumptions, outputs
    strengths: str  # where the model is expected to perform well
    limitations: str  # known weaknesses, assumptions, or risks

    selection_rationale: str  # why this model was selected for the given intent


class ModelReviewer:
    """
    A class for analyzing and reviewing generated models.
    """

    def __init__(self, provider: Provider):
        """
        Initialize the model reviewer with a provider.

        :param provider: The provider to use for generating model reviews
        """
        self.provider = provider

    def review_model(
        self,
        intent: str,
        input_schema: Dict[str, str],
        output_schema: Dict[str, str],
        solution_plan: str,
        training_code: str,
        inference_code: str,
    ) -> Dict[str, str]:
        """
        Review a generated model to extract metadata, explanations and insights about the trained model.

        :param intent: The original model intent
        :param input_schema: The input schema for the model, for example {"feat_1": "int", "feat_2": "str"}
        :param output_schema: The output schema for the model, for example {"output": "float"}
        :param solution_plan: The solution plan used to generate the model
        :param training_code: The generated training code
        :param inference_code: The generated inference code
        :return: A dictionary containing framework, model_type, creation_date and various explanations
        """
        try:
            response = self.provider.query(
                system_message=prompt_templates.review_system(),
                user_message=prompt_templates.review_model(
                    intent=intent,
                    input_schema=json.dumps(input_schema, indent=2),
                    output_schema=json.dumps(output_schema, indent=2),
                    solution_plan=solution_plan,
                    training_code=training_code,
                    inference_code=inference_code,
                ),
                response_format=ModelReviewResponse,
            )

            # Parse the response and create metadata dictionary
            review_data = json.loads(response)

            # Create metadata dictionary with review results and creation date
            metadata = {
                "framework": review_data["framework"],
                "model_type": review_data["model_type"],
                "task_type": review_data["task_type"],
                "domain": review_data["domain"],
                "behavior": review_data["behavior"],
                "preprocessing_summary": review_data["preprocessing_summary"],
                "architecture_summary": review_data["architecture_summary"],
                "training_procedure": review_data["training_procedure"],
                "evaluation_metric": review_data["evaluation_metric"],
                "inference_behavior": review_data["inference_behavior"],
                "strengths": review_data["strengths"],
                "limitations": review_data["limitations"],
                "selection_rationale": review_data["selection_rationale"],
                "creation_date": datetime.now().isoformat(),
            }
            return metadata

        except Exception as e:
            logger.warning(f"Error during model review: {str(e)}")
            # Return default values if there was an error
            return {
                "framework": "Unknown",
                "model_type": "Unknown",
                "task_type": "Unknown",
                "domain": "Unknown",
                "behavior": "Unknown",
                "preprocessing_summary": "Unknown",
                "architecture_summary": "Unknown",
                "training_procedure": "Unknown",
                "evaluation_metric": "Unknown",
                "inference_behavior": "Unknown",
                "strengths": "Unknown",
                "limitations": "Unknown",
                "selection_rationale": "Could not determine model details due to an error.",
                "creation_date": datetime.now().isoformat(),
            }
