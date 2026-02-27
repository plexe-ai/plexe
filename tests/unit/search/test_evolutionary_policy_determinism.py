"""Determinism tests for EvolutionarySearchPolicy local RNG behavior."""

from pathlib import Path
from unittest.mock import MagicMock
import random

import numpy as np

from plexe.models import BuildContext, Solution
from plexe.search.evolutionary_search_policy import EvolutionarySearchPolicy
from plexe.search.journal import SearchJournal


def _make_context(work_dir: Path) -> BuildContext:
    return BuildContext(
        user_id="user",
        experiment_id="exp",
        dataset_uri="dataset",
        work_dir=work_dir,
        intent="intent",
    )


def _make_solution(solution_id: int, performance: float | None, is_buggy: bool = False) -> Solution:
    return Solution(
        solution_id=solution_id,
        feature_pipeline=MagicMock(),
        model=MagicMock(),
        model_type="xgboost",
        performance=performance,
        is_buggy=is_buggy,
    )


def test_evolutionary_policy_determinism(monkeypatch, tmp_path):
    journal = SearchJournal()
    for idx, perf in enumerate([0.52, 0.64, 0.71, 0.79], start=1):
        journal.add_node(_make_solution(idx, performance=perf))

    journal.add_node(_make_solution(99, performance=None, is_buggy=True))

    context = _make_context(tmp_path)

    def _fail(*args, **kwargs):
        raise AssertionError("Global RNG should not be used in EvolutionarySearchPolicy")

    monkeypatch.setattr(random, "choice", _fail)
    monkeypatch.setattr(np.random, "choice", _fail)

    policy_a = EvolutionarySearchPolicy(num_drafts=2, seed=456)
    policy_b = EvolutionarySearchPolicy(num_drafts=2, seed=456)

    selected_a = policy_a.decide_next_solution(journal, context, iteration=2, max_iterations=10)
    selected_b = policy_b.decide_next_solution(journal, context, iteration=2, max_iterations=10)

    assert (selected_a is None) == (selected_b is None)
    if selected_a is not None:
        assert selected_a.solution_id == selected_b.solution_id
