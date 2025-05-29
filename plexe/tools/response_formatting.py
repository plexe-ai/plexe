"""
This module provides tools for forcing an agent to return its response in a specific format.
"""

from typing import Dict, List, Optional

from smolagents import tool


@tool
def format_final_orchestrator_agent_response(
    solution_plan: str,
    performance_metric_name: str,
    performance_metric_value: float,
    performance_metric_comparison_method: str,
    model_review_output: Dict[str, str],
) -> dict:
    """
    Returns a dictionary containing the exact fields that the agent must return in its final response. The purpose
    of this tool is to 'package' the final deliverables of the ML engineering task. The 'solution_plan' has to be
    the plan from which the best model was built.

    Args:
        solution_plan: The solution plan explanation for the selected ML model
        performance_metric_name: The name of the performance metric to optimise that was used in this task
        performance_metric_value: The value of the performance attained by the selected ML model
        performance_metric_comparison_method: The comparison method used to evaluate the performance metric
        model_review_output: The output of the 'review_model' tool which contains a review of the selected ML model

    Returns:
        Dictionary containing the fields that must be returned by the agent in its final response
    """

    return {
        "solution_plan": solution_plan,
        "performance": {
            "name": performance_metric_name,
            "value": performance_metric_value,
            "comparison_method": performance_metric_comparison_method,
        },
        "metadata": model_review_output,
    }


@tool
def format_final_mle_agent_response(
    training_code_id: str,
    execution_success: bool,
    performance_value: Optional[float] = None,
    exception: Optional[str] = None,
    model_artifact_names: Optional[List[str]] = None,
) -> dict:
    """
    Returns a dictionary containing the exact fields that the agent must return in its final response. The fields
    'performance_value', 'exception', and 'model_artifact_names' are optional. They MUST be included if they are
    available, but can be omitted if they are not available.

    Args:
        training_code_id: The training code id returned by the code execution tool after executing the training code
        execution_success: Boolean indicating if the training code executed successfully
        performance_value: The value of the performance attained by the selected ML model, if any
        exception: Exception message if the code execution failed, if any
        model_artifact_names: A list with the names of all the model artifacts created by the model training script

    Returns:
        Dictionary containing the fields that must be returned by the agent in its final response
    """

    return {
        "training_code_id": training_code_id,
        "execution_success": execution_success,
        "performance_value": performance_value,
        "exception": exception,
        "model_artifact_names": model_artifact_names,
    }


@tool
def format_final_mlops_agent_response(inference_code_id: str) -> dict:
    """
    Returns a dictionary containing the exact fields that the agent must return in its final response.

    Args:
        inference_code_id: The inference code id returned by the code validation tool after validating the inference code

    Returns:
        Dictionary containing the fields that must be returned by the agent in its final response
    """
    return {"python_inference_code": inference_code_id}
