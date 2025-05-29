"""
Tools for model testing and evaluation.

These tools help with model evaluation operations within the model generation pipeline,
including registering testing code and evaluation reports.
"""

import logging
from typing import Dict, List

from smolagents import tool

from plexe.core.object_registry import ObjectRegistry

logger = logging.getLogger(__name__)


@tool
def register_testing_code(solution_id: str, testing_code: str) -> str:
    """
    Register the testing/evaluation code in the object registry and update the Solution object. The testing code
    must first have been executed successfully before registration.

    Args:
        solution_id: ID of the Solution object to update
        testing_code: Python code used for model testing and evaluation

    Returns:
        Success message confirming registration
    """
    object_registry = ObjectRegistry()

    try:
        # Update the Solution object with testing code
        from plexe.core.entities.solution import Solution

        solution = object_registry.get(Solution, solution_id)

        solution.testing_code = testing_code

        object_registry.register(Solution, solution_id, solution, overwrite=True)

        logger.debug(f"✅ Registered model testing code for solution '{solution_id}'")
        return f"Successfully registered model testing code for solution '{solution_id}'"

    except Exception as e:
        logger.warning(f"⚠️ Error registering testing code: {str(e)}")
        raise RuntimeError(f"Failed to register testing code: {str(e)}")


@tool
def register_evaluation_report(
    solution_id: str,
    model_performance_summary: Dict,
    detailed_metrics: Dict,
    quality_analysis: Dict,
    recommendations: List[str],
    testing_insights: List[str],
) -> str:
    """
    Register comprehensive evaluation report in the object registry and link to Solution.

    This tool creates a structured report with model evaluation results and registers
    it in the Object Registry for use by other agents or final model output.

    Args:
        solution_id: ID of the Solution object to link the evaluation report to
        model_performance_summary: Overall performance metrics and scores
        detailed_metrics: Comprehensive metrics breakdown by class/category
        quality_analysis: Error patterns, robustness, interpretability insights
        recommendations: Specific recommendations for deployment/improvement
        testing_insights: Key insights from testing that impact model usage

    Returns:
        Success message confirming registration
    """
    from plexe.core.entities.solution import Solution

    object_registry = ObjectRegistry()

    try:
        # Create structured evaluation report
        evaluation_report = {
            "solution_id": solution_id,
            "model_performance_summary": model_performance_summary,
            "detailed_metrics": detailed_metrics,
            "quality_analysis": quality_analysis,
            "recommendations": recommendations,
            "testing_insights": testing_insights,
        }

        # Update Solution object with summary analysis
        solution = object_registry.get(Solution, solution_id)
        solution.model_evaluation_report = evaluation_report
        object_registry.register(Solution, solution_id, solution, overwrite=True)

        logger.debug(f"✅ Registered model evaluation report for solution '{solution_id}'")
        return f"Successfully registered model evaluation report for solution '{solution_id}'"

    except Exception as e:
        logger.warning(f"⚠️ Error registering evaluation report: {str(e)}")
        raise RuntimeError(f"Failed to register evaluation report: {str(e)}")
