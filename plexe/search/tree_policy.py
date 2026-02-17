"""
Tree-search policy inspired by AIDE.

Simple three-stage search:
1. Draft: Create diverse initial solutions
2. Debug: Fix buggy solutions
3. Improve: Enhance best solutions (with exploration→exploitation annealing)
"""

import logging
import random

import numpy as np

from plexe.constants import SearchDefaults
from plexe.models import BuildContext, Solution
from plexe.search.journal import SearchJournal
from plexe.search.policy import SearchPolicy

logger = logging.getLogger(__name__)


class TreeSearchPolicy(SearchPolicy):
    """AIDE-inspired tree-search with draft/debug/improve stages."""

    def __init__(
        self,
        num_drafts: int = SearchDefaults.NUM_DRAFTS,
        debug_prob: float = SearchDefaults.DEBUG_PROB,
        max_debug_depth: int = SearchDefaults.MAX_DEBUG_DEPTH,
    ):
        self.num_drafts = num_drafts
        self.debug_prob = debug_prob
        self.max_debug_depth = max_debug_depth

    def decide_next_solution(
        self, journal: SearchJournal, context: BuildContext, iteration: int, max_iterations: int
    ) -> Solution | None:
        """Decide which solution node to expand next."""

        # Stage 1: Bootstrap - create initial root nodes
        root_nodes = [n for n in journal.nodes if n.parent is None]
        if len(root_nodes) < self.num_drafts:
            logger.info(f"Bootstrap {len(root_nodes) + 1}/{self.num_drafts} - starting from scratch")
            return None

        # Stage 2: Debugging (Probabilistic)
        if random.random() < self.debug_prob:
            debuggable = [n for n in journal.buggy_nodes if n.is_leaf and n.debug_depth < self.max_debug_depth]
            if debuggable:
                buggy = random.choice(debuggable)
                logger.info(f"Debug solution {buggy.solution_id}")
                return buggy

        # Stage 3: Improvement with exploration→exploitation annealing
        if not journal.good_nodes:
            logger.info("No good solutions - continuing from scratch")
            return None

        progress = iteration / max_iterations
        k = max(1, round(self.num_drafts * (1 - progress)))
        temp = max(0.3, (1 - progress) ** 2)

        sorted_nodes = sorted(journal.good_nodes, key=lambda n: n.performance, reverse=True)
        top_k = sorted_nodes[:k]

        # Greedy if k=1 or low temperature
        if len(top_k) == 1 or temp < 0.35:
            logger.info(f"Greedy: solution {top_k[0].solution_id} (perf={top_k[0].performance:.4f})")
            return top_k[0]

        # Softmax sampling
        perfs = np.array([n.performance for n in top_k])
        probs = np.exp((perfs / temp) - np.max(perfs / temp))
        probs /= probs.sum()
        selected = np.random.choice(top_k, p=probs)

        logger.info(
            f"Softmax: solution {selected.solution_id} (perf={selected.performance:.4f}, k={k}, temp={temp:.2f})"
        )
        return selected

    def should_stop(self, journal: SearchJournal, iteration: int, max_iterations: int) -> bool:
        """Decide if search should terminate early."""

        # Max iterations
        if iteration >= max_iterations - 1:
            logger.info(f"Reached max iterations ({max_iterations})")
            return True

        # Stagnation check
        # if iteration >= SearchDefaults.STAGNATION_WINDOW:
        #     trend = journal.get_improvement_trend(window=SearchDefaults.STAGNATION_WINDOW)
        #     if abs(trend) < SearchDefaults.STAGNATION_THRESHOLD:
        #         logger.info(f"Search stagnating (trend={trend:.6f})")

        return False
