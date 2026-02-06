"""
Search policy abstract base class.

Plugin interface for different search strategies.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from plexe.models import BuildContext, Solution
    from plexe.search.journal import SearchJournal


class SearchPolicy(ABC):
    """
    Search strategy for selecting which solution node to expand next.

    Implementations decide:
    - Which solution to expand (tree-search strategy)
    - When to stop searching (early termination criteria)
    """

    @abstractmethod
    def decide_next_solution(
        self, journal: "SearchJournal", context: "BuildContext", iteration: int, max_iterations: int
    ) -> Optional["Solution"]:
        """
        Select which solution node to expand in the next iteration.

        Args:
            journal: Search journal with tree state and history
            context: Build context with task info and baselines
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            Solution node to expand, or None to start from scratch
        """
        pass

    @abstractmethod
    def should_stop(self, journal: "SearchJournal", iteration: int, max_iterations: int) -> bool:
        """
        Decide if search should terminate early.

        Args:
            journal: Search journal with current state
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations

        Returns:
            True to stop search, False to continue
        """
        pass
