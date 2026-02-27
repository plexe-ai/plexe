"""
PiEvolve-inspired evolutionary search policy with adaptive state analysis.

Implements probabilistic action selection based on search state metrics:
- Performance variance analysis
- Recent progress trend detection
- Stagnation scoring
- Dynamic action probability adjustment
"""

import logging
import random

import numpy as np

from plexe.constants import SearchDefaults
from plexe.models import BuildContext, Solution
from plexe.search.journal import SearchJournal
from plexe.search.policy import SearchPolicy

logger = logging.getLogger(__name__)


class EvolutionarySearchPolicy(SearchPolicy):
    """PiEvolve-inspired probabilistic action selection with adaptive search state analysis."""

    def __init__(
        self,
        num_drafts: int = SearchDefaults.NUM_DRAFTS,
        debug_prob: float = SearchDefaults.DEBUG_PROB,
        max_debug_depth: int = SearchDefaults.MAX_DEBUG_DEPTH,
        seed: int | None = None,
    ):
        self.num_drafts = num_drafts
        self.debug_prob = debug_prob  # Used as baseline, but overridden by state analysis
        self.max_debug_depth = max_debug_depth
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def decide_next_solution(
        self, journal: SearchJournal, context: BuildContext, iteration: int, max_iterations: int
    ) -> Solution | None:
        """PiEvolve-style probabilistic action selection based on search state."""

        # Stage 1: Bootstrap - same as original but with logging
        root_nodes = [n for n in journal.nodes if n.parent is None]
        if len(root_nodes) < self.num_drafts:
            logger.info(f"Bootstrap {len(root_nodes) + 1}/{self.num_drafts} - starting from scratch")
            return None

        # Analyze current search state
        performance_variance = self._calculate_performance_variance(journal)
        recent_progress = self._calculate_recent_progress(journal, window=5)
        stagnation_score = self._calculate_stagnation(journal, window=3)

        logger.info(
            f"Search State Analysis [Iter {iteration}/{max_iterations}]: "
            f"variance={performance_variance:.3f}, progress={recent_progress:.3f}, "
            f"stagnation={stagnation_score:.3f}, good_solutions={len(journal.good_nodes)}, "
            f"buggy_solutions={len(journal.buggy_nodes)}"
        )

        # Probabilistic action selection based on search state
        action = self._select_action(journal, performance_variance, recent_progress, stagnation_score)
        logger.info(f"Selected action: {action}")

        # Execute selected action
        if action == "explore":
            return self._explore_action(journal)
        elif action == "exploit":
            return self._exploit_action(journal, iteration, max_iterations)
        elif action == "debug":
            return self._debug_action(journal)
        else:  # "mutate"
            return self._mutate_action(journal)

    def _calculate_performance_variance(self, journal: SearchJournal) -> float:
        """Calculate performance variance across good solutions."""
        good_nodes = journal.good_nodes
        if len(good_nodes) < 2:
            return 1.0  # High variance when few solutions (encourages exploration)

        performances = [n.performance for n in good_nodes]
        return float(np.var(performances))

    def _calculate_recent_progress(self, journal: SearchJournal, window: int = 5) -> float:
        """Calculate recent performance improvement trend."""
        if len(journal.nodes) < window:
            return 0.5  # Neutral when insufficient data

        recent_nodes = journal.nodes[-window:]
        good_recent = [n for n in recent_nodes if not n.is_buggy and n.performance is not None]

        # If insufficient good recent data, distinguish between early stages vs buggy period
        if len(good_recent) < 2:
            # Check if we have any good solutions in the entire journal
            if len(journal.good_nodes) < 2:
                return 0.5  # Neutral - early exploration phase
            else:
                return -0.3  # Slight negative - recent period mostly buggy but we had good solutions before

        # Calculate trend slope using linear regression
        performances = [n.performance for n in good_recent]
        x = np.arange(len(performances))
        slope = np.polyfit(x, performances, 1)[0] if len(performances) > 1 else 0.0

        # Normalize slope to [-1, 1] range
        return float(np.clip(slope * 10, -1.0, 1.0))  # Scale for typical performance ranges

    def _calculate_stagnation(self, journal: SearchJournal, window: int = 3) -> float:
        """Calculate stagnation score based on recent improvement patterns."""
        if not journal.good_nodes:
            return 0.0  # No stagnation if no good solutions yet

        best_performance = journal.best_performance
        recent_nodes = journal.nodes[-window:] if len(journal.nodes) >= window else journal.nodes

        # Safety check: ensure we have nodes to analyze
        if not recent_nodes:
            return 0.0

        # Count recent solutions that are close to best performance
        improvements = 0
        threshold = 0.01  # 1% improvement threshold

        for node in reversed(recent_nodes):
            if not node.is_buggy and node.performance is not None:
                if node.performance >= best_performance - threshold:
                    improvements += 1

        # High stagnation = few improvements in recent window
        # Division is safe since recent_nodes is guaranteed non-empty
        stagnation = 1.0 - (improvements / len(recent_nodes))
        return min(1.0, stagnation)

    def _select_action(self, journal: SearchJournal, variance: float, progress: float, stagnation: float) -> str:
        """Select action based on search state metrics using PiEvolve-style logic."""

        # Count debugging candidates
        debuggable_count = len([n for n in journal.buggy_nodes if n.is_leaf and n.debug_depth < self.max_debug_depth])

        # State-based action probability calculation
        if stagnation > 0.7 or (variance < 0.1 and progress < 0.05):
            # HIGH STAGNATION: Need diversity and exploration
            probs = {"explore": 0.5, "mutate": 0.3, "debug": 0.15, "exploit": 0.05}
            logger.info("State: HIGH STAGNATION - favoring exploration")

        elif progress > 0.1 and variance > 0.2:
            # GOOD PROGRESS WITH DIVERSITY: Exploit promising areas
            probs = {"exploit": 0.5, "explore": 0.2, "debug": 0.2, "mutate": 0.1}
            logger.info("State: GOOD PROGRESS - favoring exploitation")

        elif debuggable_count > 2:
            # MANY BUGS: Focus on debugging
            probs = {"debug": 0.4, "exploit": 0.3, "explore": 0.2, "mutate": 0.1}
            logger.info(f"State: MANY BUGS ({debuggable_count}) - favoring debugging")

        elif len(journal.good_nodes) < 2:
            # FEW GOOD SOLUTIONS: Need more exploration
            probs = {"explore": 0.6, "mutate": 0.2, "debug": 0.1, "exploit": 0.1}
            logger.info("State: FEW GOOD SOLUTIONS - favoring exploration")

        else:
            # BALANCED STATE: Mixed strategy
            probs = {"exploit": 0.35, "explore": 0.25, "debug": 0.25, "mutate": 0.15}
            logger.info("State: BALANCED - mixed strategy")

        # Adjust probabilities if certain actions aren't viable
        if debuggable_count == 0:
            # Redistribute debug probability to other actions
            debug_prob = probs.pop("debug", 0)
            remaining_prob = sum(probs.values())
            for action in probs:
                probs[action] += debug_prob * (probs[action] / remaining_prob)

        if len(journal.good_nodes) == 0:
            # Can't exploit or mutate without good solutions
            exploit_prob = probs.pop("exploit", 0)
            mutate_prob = probs.pop("mutate", 0)
            probs["explore"] = probs.get("explore", 0) + exploit_prob + mutate_prob

        # Sample action with calculated probabilities
        actions, weights = list(probs.keys()), list(probs.values())
        return str(self._np_rng.choice(actions, p=weights))

    def _explore_action(self, journal: SearchJournal) -> Solution | None:
        """Generate diverse new solution from scratch."""
        logger.info("Action: EXPLORE - creating new solution for diversity")
        return None  # None signals to start from scratch

    def _exploit_action(self, journal: SearchJournal, iteration: int, max_iterations: int) -> Solution | None:
        """Focus on improving best performing solutions with intelligent selection."""
        good_nodes = journal.good_nodes
        if not good_nodes:
            logger.info("Action: EXPLOIT - no good solutions, falling back to explore")
            return None

        # Use progressive focusing: early iterations explore top-k, later focus on best
        progress = iteration / max_iterations
        k = max(1, min(len(good_nodes), max(1, len(good_nodes) // 2) - int(progress * len(good_nodes) // 3)))

        # Temperature annealing for softmax selection
        temp = max(0.2, (1 - progress) ** 1.5)

        # Focus on top-k performers
        sorted_nodes = sorted(good_nodes, key=lambda n: n.performance, reverse=True)
        top_k = sorted_nodes[:k]

        if len(top_k) == 1 or temp < 0.25:
            selected = top_k[0]
            logger.info(f"Action: EXPLOIT (greedy) - solution {selected.solution_id} (perf={selected.performance:.4f})")
        else:
            # Softmax selection among top-k
            perfs = np.array([n.performance for n in top_k])
            # Numerical stability: subtract max before exp
            exp_probs = np.exp((perfs / temp) - np.max(perfs / temp))
            probs = exp_probs / np.sum(exp_probs)
            selected = self._np_rng.choice(top_k, p=probs)
            logger.info(
                f"Action: EXPLOIT (softmax) - solution {selected.solution_id} "
                f"(perf={selected.performance:.4f}, k={k}, temp={temp:.2f})"
            )

        return selected

    def _debug_action(self, journal: SearchJournal) -> Solution | None:
        """Systematically debug failed solutions, prioritizing recent failures."""
        debuggable = [n for n in journal.buggy_nodes if n.is_leaf and n.debug_depth < self.max_debug_depth]

        if not debuggable:
            logger.info("Action: DEBUG - no debuggable solutions, falling back to explore")
            return self._explore_action(journal)

        # Prioritize recent failures (more likely to be fixable)
        # Recent failures are at the end of the list
        selected = debuggable[-1] if debuggable else None

        if selected:
            logger.info(
                f"Action: DEBUG - fixing recent failure solution {selected.solution_id} "
                f"(debug_depth={selected.debug_depth})"
            )
        else:
            logger.info("Action: DEBUG - no suitable debug candidates")

        return selected

    def _mutate_action(self, journal: SearchJournal) -> Solution | None:
        """Apply small parameter variations to existing good solutions."""
        good_nodes = journal.good_nodes
        if not good_nodes:
            logger.info("Action: MUTATE - no good solutions, falling back to explore")
            return self._explore_action(journal)

        # Prefer solutions with medium performance for mutation (not best, not worst)
        sorted_nodes = sorted(good_nodes, key=lambda n: n.performance, reverse=True)
        mid_range_start = max(0, len(sorted_nodes) // 4)
        mid_range_end = min(len(sorted_nodes), 3 * len(sorted_nodes) // 4)
        mid_range = sorted_nodes[mid_range_start:mid_range_end] if mid_range_end > mid_range_start else sorted_nodes

        selected = self._rng.choice(mid_range) if mid_range else self._rng.choice(good_nodes)
        logger.info(f"Action: MUTATE - varying solution {selected.solution_id} (perf={selected.performance:.4f})")
        return selected

    def should_stop(self, journal: SearchJournal, iteration: int, max_iterations: int) -> bool:
        """Enhanced stopping criteria with intelligent early stopping."""

        # Always stop at max iterations
        if iteration >= max_iterations - 1:
            logger.info(f"Stopping: Reached max iterations ({max_iterations})")
            return True

        # Early stopping logic (only after halfway point to allow for exploration)
        if iteration > max_iterations * 0.4:
            stagnation = self._calculate_stagnation(journal, window=5)

            # Stop if highly stagnant AND we have a good solution (>10% improvement over baseline)
            if (
                stagnation > 0.8
                and journal.best_performance > journal.baseline_performance * 1.1
                and len(journal.good_nodes) >= 2
            ):

                logger.info(
                    f"Early stopping: High stagnation ({stagnation:.3f}) with good performance "
                    f"({journal.best_performance:.4f} vs baseline {journal.baseline_performance:.4f})"
                )
                return True

            # Stop if we have exceptional performance (>50% improvement) and some stagnation
            if stagnation > 0.6 and journal.best_performance > journal.baseline_performance * 1.5:

                logger.info(
                    f"Early stopping: Exceptional performance ({journal.best_performance:.4f}) "
                    f"with moderate stagnation ({stagnation:.3f})"
                )
                return True

        return False
