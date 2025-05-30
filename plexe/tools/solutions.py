"""
Tools for creating and managing Solution objects in the ML workflow.

These tools handle the creation, registration, and management of Solution objects
that represent complete ML approaches from planning through execution.
"""

import logging
from typing import Dict, List

from smolagents import tool

from plexe.core.object_registry import ObjectRegistry
from plexe.core.entities.solution import Solution

logger = logging.getLogger(__name__)


def get_solution_creation_tool(max_solutions: int = 1):
    """
    Returns a tool function to create a new Solution object with a plan.
    This tool is used by the ML Research Scientist agent to develop new solution approaches.

    Args:
        max_solutions: Maximum number of solutions that can be created at once

    Returns:
        A tool function that creates a Solution object
    """
    if max_solutions <= 0:
        raise ValueError("max_solutions must be greater than 0")

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

        # Check if the maximum number of solutions has been reached
        if len(object_registry.get_all(Solution)) >= max_solutions:
            raise RuntimeError(f"Maximum number of solutions ({max_solutions}) reached. Cannot create more solutions.")

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

    return create_solution


@tool
def get_solution_plan_by_id(solution_id: str) -> str:
    """
    Retrieves a model solution plan by its ID.

    Args:
        solution_id: ID of the Solution

    Returns:
        The plan string of the Solution
    """
    object_registry = ObjectRegistry()

    try:
        solution = object_registry.get(Solution, solution_id)
        return solution.plan if solution.plan else "No plan available for this solution"

    except Exception as e:
        logger.warning(f"⚠️ Error retrieving solution plan: {str(e)}")
        raise RuntimeError(f"Failed to retrieve solution plan: {str(e)}")


@tool
def list_solutions() -> List[str]:
    """
    Lists all Solution IDs currently available. Use this tool to see all available solutions if you run into
    issues with retrieving a specific solution.

    Returns:
        List of solution IDs currently available
    """
    object_registry = ObjectRegistry()

    try:
        solution_ids = object_registry.list_by_type(Solution)

        logger.debug(f"✅ Available solutions: {solution_ids}")
        return solution_ids

    except Exception as e:
        logger.warning(f"⚠️ Error listing solutions: {str(e)}")
        raise RuntimeError(f"Failed to list solutions: {str(e)}")
