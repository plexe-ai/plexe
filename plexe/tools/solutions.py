"""
Tools for creating and managing Solution objects in the ML workflow.

These tools handle the creation, registration, and management of Solution objects
that represent complete ML approaches from planning through execution.
"""

import logging
from typing import Dict

from smolagents import tool

from plexe.core.object_registry import ObjectRegistry
from plexe.core.entities.solution import Solution

logger = logging.getLogger(__name__)


@tool
def create_solution(plan: str) -> Dict[str, str]:
    """
    Creates a new Solution object with the given plan and registers it in the object registry so
    that other agents in the team can access it.

    This tool should be used by the ML Research Scientist agent when developing new solution
    approaches. Each solution represents a distinct ML strategy that will be implemented
    and evaluated.

    Args:
        plan: The detailed solution plan and strategy description for this ML approach

    Returns:
        Dictionary containing the solution ID and success confirmation:
        {
            "solution_id": "unique_solution_identifier",
            "message": "Success message"
        }
    """
    object_registry = ObjectRegistry()

    try:
        # Create a new Solution object with the provided plan
        solution = Solution(plan=plan)

        # Register the solution in the object registry
        object_registry.register(Solution, solution.id, solution, overwrite=False)

        logger.debug(f"✅ Created and registered solution with ID '{solution.id}'")

        return {
            "solution_id": solution.id,
            "message": f"Successfully created and registered solution with ID '{solution.id}'",
        }

    except Exception as e:
        logger.warning(f"⚠️ Error creating solution: {str(e)}")
        raise RuntimeError(f"Failed to create solution: {str(e)}")
