"""
Tools for model testing and evaluation.

These tools help with model evaluation operations within the model generation pipeline,
including registering testing code and evaluation reports.
"""

import logging
from typing import Dict, List

from smolagents import tool

from plexe.core.object_registry import ObjectRegistry
from plexe.internal.models.entities.code import Code

logger = logging.getLogger(__name__)


@tool
def register_testing_code(testing_code: str) -> str:
    """
    Register the testing/evaluation code in the object registry.

    Args:
        testing_code: Python code used for model testing and evaluation

    Returns:
        Success message confirming registration
    """
    object_registry = ObjectRegistry()

    try:
        # Register the testing code with a fixed ID
        code_id = "model_testing_code"
        object_registry.register(Code, code_id, Code(testing_code), overwrite=True)

        logger.debug(f"✅ Registered model testing code with ID '{code_id}'")
        return f"Successfully registered model testing code with ID '{code_id}'"

    except Exception as e:
        logger.warning(f"⚠️ Error registering testing code: {str(e)}")
        raise RuntimeError(f"Failed to register testing code: {str(e)}")


@tool
def register_evaluation_report(
    model_performance_summary: Dict,
    detailed_metrics: Dict,
    quality_analysis: Dict,
    recommendations: List[str],
    testing_insights: List[str],
) -> str:
    """
    Register comprehensive evaluation report in the object registry.

    This tool creates a structured report with model evaluation results and registers
    it in the Object Registry for use by other agents or final model output.

    Args:
        model_performance_summary: Overall performance metrics and scores
        detailed_metrics: Comprehensive metrics breakdown by class/category
        quality_analysis: Error patterns, robustness, interpretability insights
        recommendations: Specific recommendations for deployment/improvement
        testing_insights: Key insights from testing that impact model usage

    Returns:
        Success message confirming registration
    """
    object_registry = ObjectRegistry()

    try:
        # Create structured evaluation report
        evaluation_report = {
            "model_performance_summary": model_performance_summary,
            "detailed_metrics": detailed_metrics,
            "quality_analysis": quality_analysis,
            "recommendations": recommendations,
            "testing_insights": testing_insights,
        }

        # Register in registry
        report_key = "model_evaluation_report"
        object_registry.register(dict, report_key, evaluation_report, overwrite=True)

        logger.debug(f"✅ Registered model evaluation report with key '{report_key}'")
        return f"Successfully registered model evaluation report with key '{report_key}'"

    except Exception as e:
        logger.warning(f"⚠️ Error registering evaluation report: {str(e)}")
        raise RuntimeError(f"Failed to register evaluation report: {str(e)}")
