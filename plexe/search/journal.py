"""
Search journal for tracking model search tree.

Maintains tree structure of solutions with draft/debug/improve relationships.
"""

import logging

from plexe.models import Baseline, Solution

logger = logging.getLogger(__name__)


def _compute_stage(node: Solution) -> str:
    """
    Compute display stage for a solution node.

    Returns:
        "root" for bootstrap nodes (no parent)
        "child" for all other nodes
    """
    return "root" if node.parent is None else "child"


class SearchJournal:
    """
    Tracks solution search tree.

    Maintains tree structure of Solutions with:
    - Draft nodes (initial diverse solutions)
    - Debug nodes (bug fixes)
    - Improve nodes (enhancements)
    """

    def __init__(self, baseline: Baseline | None = None):
        """
        Initialize journal.

        Args:
            baseline: Baseline model for comparison
        """
        self.baseline = baseline
        self.baseline_performance = baseline.performance if baseline else 0.0

        self.nodes: list[Solution] = []
        self.successful_attempts = 0
        self.failed_attempts = 0

    # ============================================
    # Adding Nodes
    # ============================================

    def add_node(self, node: Solution) -> None:
        """
        Add a solution to the journal.

        Args:
            node: Solution to add
        """
        self.nodes.append(node)

        if node.is_buggy:
            self.failed_attempts += 1
            logger.warning(f"Solution {node.solution_id} failed: {node.error}")
        else:
            self.successful_attempts += 1
            if node.performance is not None:
                logger.info(f"Solution {node.solution_id} succeeded: {node.performance:.4f}")

    # ============================================
    # Tree Queries
    # ============================================

    @property
    def draft_nodes(self) -> list[Solution]:
        """Get all root nodes (bootstrap solutions without parents)."""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Solution]:
        """Get all buggy nodes that could be debugged."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Solution]:
        """Get all non-buggy nodes with valid performance."""
        return [n for n in self.nodes if not n.is_buggy and n.performance is not None]

    @property
    def best_node(self) -> Solution | None:
        """
        Get best performing solution.

        Returns:
            Solution with highest performance, or None if no successful solutions
        """
        good = self.good_nodes
        if not good:
            return None
        return max(good, key=lambda n: n.performance)

    @property
    def best_performance(self) -> float:
        """Get best performance achieved so far."""
        best = self.best_node
        if best:
            return best.performance
        return self.baseline_performance

    # ============================================
    # History Methods
    # ============================================

    def get_history(self, limit: int = 10) -> list[dict]:
        """
        Get recent search history for agent consumption.

        Args:
            limit: Number of recent solutions to include

        Returns:
            List of solution summaries
        """
        recent = self.nodes[-limit:] if len(self.nodes) > limit else self.nodes

        history = []
        for node in recent:
            entry = {
                "solution_id": node.solution_id,
                "stage": _compute_stage(node),
                "success": not node.is_buggy,
                "performance": node.performance,
                "error": node.error,
            }

            # Include structured plan if available
            if node.plan:
                entry["plan"] = {
                    "variant_id": node.plan.variant_id,
                    "features": {
                        "strategy": node.plan.features.strategy,
                        "rationale": node.plan.features.rationale[:80],
                    },
                    "model": {
                        "directive": node.plan.model.directive[:100],
                        "change_summary": node.plan.model.change_summary,
                        "rationale": node.plan.model.rationale[:80],
                    },
                    "hypothesis_rationale": node.plan.hypothesis_rationale[:100],
                }
            else:
                # Legacy nodes without structured plans
                entry["plan"] = None

            if node.parent:
                entry["parent_solution_id"] = node.parent.solution_id

            history.append(entry)

        return history

    def summarize(self) -> str:
        """
        Generate text summary of search progress.

        Returns:
            Human-readable summary
        """
        total = len(self.nodes)

        summary = "Search Progress:\n"
        summary += f"  Total attempts: {total}\n"
        summary += f"  Successful: {self.successful_attempts}\n"
        summary += f"  Failed: {self.failed_attempts}\n"

        # Stage breakdown
        root_count = len([n for n in self.nodes if n.parent is None])
        child_count = len([n for n in self.nodes if n.parent is not None])
        buggy_count = len([n for n in self.nodes if n.is_buggy])

        summary += f"  Nodes: {root_count} roots, {child_count} children ({buggy_count} buggy)\n"

        if self.baseline:
            summary += f"  Baseline: {self.baseline_performance:.4f}\n"

        best = self.best_node
        if best:
            improvement = (
                (best.performance - self.baseline_performance) / self.baseline_performance * 100
                if self.baseline_performance > 0
                else 0
            )
            summary += (
                f"  Best: {best.performance:.4f} ({improvement:+.1f}% vs baseline) [solution {best.solution_id}]\n"
            )
        else:
            summary += "  Best: None\n"

        return summary

    def get_improvement_trend(self, window: int = 5) -> float:
        """
        Calculate improvement trend over recent successful iterations.

        Args:
            window: Number of recent solutions to analyze

        Returns:
            Average performance improvement per iteration
        """
        successful = self.good_nodes

        if len(successful) < 2:
            return 0.0

        # Look at last N successful attempts
        recent = successful[-window:] if len(successful) > window else successful
        performances = [n.performance for n in recent]

        # Calculate average delta
        deltas = [performances[i + 1] - performances[i] for i in range(len(performances) - 1)]

        return sum(deltas) / len(deltas) if deltas else 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate overall failure rate."""
        total = len(self.nodes)
        return self.failed_attempts / total if total > 0 else 0.0

    # ============================================
    # Analysis Methods
    # ============================================

    def get_tree_depth(self) -> int:
        """Get maximum depth of the search tree."""
        if not self.nodes:
            return 0

        def node_depth(node: Solution) -> int:
            if node.parent is None:
                return 0
            return node_depth(node.parent) + 1

        return max(node_depth(n) for n in self.nodes)

    def get_successful_improvements(self, limit: int = 5) -> list[Solution]:
        """
        Get recent successful child nodes for learning.

        Args:
            limit: Number of improvements to return

        Returns:
            List of successful child solutions (non-root) that performed well
        """
        child_nodes = [n for n in self.nodes if n.parent is not None and not n.is_buggy and n.performance is not None]

        # Sort by performance
        child_nodes.sort(key=lambda n: n.performance, reverse=True)

        return child_nodes[:limit]

    # ============================================
    # Serialization
    # ============================================

    def to_dict(self) -> dict:
        """Serialize SearchJournal to dict for checkpointing."""
        return {
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "baseline_performance": self.baseline_performance,
            "nodes": [node.to_dict() for node in self.nodes],
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
        }

    @staticmethod
    def from_dict(d: dict) -> "SearchJournal":
        """
        Deserialize SearchJournal from checkpoint dict.

        Reconstructs Solution tree with parent/child links.
        """
        from plexe.models import Baseline, Solution

        # Recreate baseline
        baseline = Baseline.from_dict(d["baseline"]) if d.get("baseline") else None

        # Create journal
        journal = SearchJournal(baseline=baseline)
        journal.baseline_performance = d.get("baseline_performance", 0.0)
        journal.successful_attempts = d.get("successful_attempts", 0)
        journal.failed_attempts = d.get("failed_attempts", 0)

        # Two-pass reconstruction of Solution tree
        # Pass 1: Create all Solution nodes without parent/child links
        all_solutions: dict[int, Solution] = {}
        for node_dict in d.get("nodes", []):
            solution = Solution.from_dict(node_dict, all_solutions)
            all_solutions[solution.solution_id] = solution

        # Pass 2: Link parent/child relationships
        for node_dict in d.get("nodes", []):
            solution_id = node_dict["solution_id"]
            solution = all_solutions[solution_id]

            # Link parent
            parent_solution_id = node_dict.get("parent_solution_id")
            if parent_solution_id is not None and parent_solution_id in all_solutions:
                solution.parent = all_solutions[parent_solution_id]

            # Link children
            child_solution_ids = node_dict.get("child_solution_ids", [])
            solution.children = [
                all_solutions[child_id] for child_id in child_solution_ids if child_id in all_solutions
            ]

            # Add to journal
            journal.nodes.append(solution)

        return journal
