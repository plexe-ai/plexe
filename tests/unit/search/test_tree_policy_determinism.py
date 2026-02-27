"""Determinism tests for TreeSearchPolicy local RNG behavior."""

from pathlib import Path
from unittest.mock import MagicMock
import random

import numpy as np

from plexe.models import BuildContext, Solution
from plexe.search.journal import SearchJournal
from plexe.search.tree_policy import TreeSearchPolicy


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


def test_tree_policy_determinism(monkeypatch, tmp_path):
    journal = SearchJournal()
    for idx, perf in enumerate([0.61, 0.72, 0.83], start=1):
        journal.add_node(_make_solution(idx, performance=perf))

    context = _make_context(tmp_path)

    def _fail(*args, **kwargs):
        raise AssertionError("Global RNG should not be used in TreeSearchPolicy")

    monkeypatch.setattr(random, "random", _fail)
    monkeypatch.setattr(random, "choice", _fail)
    monkeypatch.setattr(np.random, "choice", _fail)

    policy_a = TreeSearchPolicy(num_drafts=2, debug_prob=0.0, seed=123)
    policy_b = TreeSearchPolicy(num_drafts=2, debug_prob=0.0, seed=123)

    selected_a = policy_a.decide_next_solution(journal, context, iteration=1, max_iterations=10)
    selected_b = policy_b.decide_next_solution(journal, context, iteration=1, max_iterations=10)

    assert selected_a is not None
    assert selected_b is not None
    assert selected_a.solution_id == selected_b.solution_id
