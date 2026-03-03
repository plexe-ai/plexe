"""
Unit tests for SearchJournal.
"""

from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plexe.models import Baseline, Solution
from plexe.search.journal import SearchJournal


# ============================================
# Helpers
# ============================================


def _make_solution(solution_id: int, performance: float | None = None, is_buggy: bool = False, parent=None, error=None):
    """Create a Solution with minimal required fields for testing."""
    return Solution(
        solution_id=solution_id,
        feature_pipeline=Pipeline([("scaler", StandardScaler())]),
        model=None,
        model_type="xgboost",
        model_artifacts_path=Path(f"/tmp/model_{solution_id}"),
        performance=performance,
        is_buggy=is_buggy,
        parent=parent,
        error=error,
    )


# ============================================
# Initialization Tests
# ============================================


def test_journal_initialization():
    """Test journal initializes correctly."""
    baseline = Baseline(name="test_baseline", model_type="heuristic", performance=0.75)

    journal = SearchJournal(baseline=baseline)

    assert journal.baseline == baseline
    assert journal.baseline_performance == 0.75
    assert journal.successful_attempts == 0
    assert journal.failed_attempts == 0
    assert journal.best_node is None


def test_journal_initialization_no_baseline():
    """Test journal initializes without baseline."""
    journal = SearchJournal()

    assert journal.baseline is None
    assert journal.baseline_performance == 0.0


# ============================================
# Adding Nodes Tests
# ============================================


def test_journal_add_successful_node():
    """Test recording a successful solution."""
    journal = SearchJournal()

    solution = _make_solution(0, performance=0.85)
    journal.add_node(solution)

    assert journal.successful_attempts == 1
    assert journal.failed_attempts == 0
    assert len(journal.nodes) == 1


def test_journal_add_buggy_node():
    """Test recording a failed attempt."""
    journal = SearchJournal()

    solution = _make_solution(0, is_buggy=True, error="Training failed")
    journal.add_node(solution)

    assert journal.successful_attempts == 0
    assert journal.failed_attempts == 1


def test_journal_best_node_tracks_best():
    """Test best_node returns the highest performing solution."""
    journal = SearchJournal()

    sol1 = _make_solution(0, performance=0.80)
    sol2 = _make_solution(1, performance=0.90)
    sol3 = _make_solution(2, performance=0.85)

    journal.add_node(sol1)
    journal.add_node(sol2)
    journal.add_node(sol3)

    assert journal.best_node == sol2
    assert journal.best_performance == 0.90


def test_journal_best_node_respects_lower_direction():
    """best_node should select smallest metric when optimization is lower."""
    journal = SearchJournal(optimization_direction="lower")

    sol1 = _make_solution(0, performance=0.40)
    sol2 = _make_solution(1, performance=0.25)
    sol3 = _make_solution(2, performance=0.31)

    journal.add_node(sol1)
    journal.add_node(sol2)
    journal.add_node(sol3)

    assert journal.best_node == sol2
    assert journal.best_performance == 0.25


# ============================================
# Failure Rate Tests
# ============================================


def test_journal_failure_rate():
    """Test failure rate computation."""
    journal = SearchJournal()

    journal.add_node(_make_solution(0, performance=0.80))
    journal.add_node(_make_solution(1, is_buggy=True, error="Failed"))
    journal.add_node(_make_solution(2, performance=0.85))
    journal.add_node(_make_solution(3, is_buggy=True, error="Failed"))

    assert journal.failure_rate == pytest.approx(0.5)


def test_journal_failure_rate_empty():
    """Test failure rate on empty journal."""
    journal = SearchJournal()
    assert journal.failure_rate == 0.0


# ============================================
# History Tests
# ============================================


def test_journal_get_history():
    """Test history returns recent entries."""
    journal = SearchJournal()

    for i in range(5):
        journal.add_node(_make_solution(i, performance=0.70 + i * 0.05))

    history = journal.get_history(limit=3)

    assert len(history) == 3
    assert history[0]["solution_id"] == 2  # most recent 3: ids 2, 3, 4
    assert all(entry["success"] for entry in history)


# ============================================
# Improvement Trend Tests
# ============================================


def test_journal_improvement_trend_improving():
    """Test improvement trend with steadily improving solutions."""
    journal = SearchJournal()

    journal.add_node(_make_solution(0, performance=0.70))
    journal.add_node(_make_solution(1, performance=0.75))
    journal.add_node(_make_solution(2, performance=0.80))

    trend = journal.get_improvement_trend()

    assert trend > 0  # Positive trend


def test_journal_improvement_trend_insufficient_data():
    """Test improvement trend with fewer than 2 successful solutions."""
    journal = SearchJournal()
    journal.add_node(_make_solution(0, performance=0.80))

    trend = journal.get_improvement_trend()

    assert trend == 0.0


# ============================================
# Train Performance in History Tests
# ============================================


def test_journal_get_history_includes_train_performance():
    """get_history should include train_performance when set on a solution."""
    journal = SearchJournal()
    sol = _make_solution(0, performance=0.85)
    sol.train_performance = 0.92
    journal.add_node(sol)

    history = journal.get_history()
    assert history[0]["train_performance"] == 0.92


def test_journal_get_history_train_performance_none():
    """get_history should include train_performance=None when not set."""
    journal = SearchJournal()
    journal.add_node(_make_solution(0, performance=0.85))

    history = journal.get_history()
    assert history[0]["train_performance"] is None


def test_journal_serialization_preserves_optimization_direction():
    """to_dict/from_dict should preserve optimization_direction."""
    journal = SearchJournal(optimization_direction="lower")
    journal.add_node(_make_solution(0, performance=0.3))

    restored = SearchJournal.from_dict(journal.to_dict())
    assert restored.optimization_direction == "lower"
    assert restored.best_performance == pytest.approx(0.3)


def test_journal_from_dict_defaults_optimization_direction_to_higher():
    """Older checkpoints without optimization_direction should default to higher."""
    journal = SearchJournal()
    journal.add_node(_make_solution(0, performance=0.3))
    payload = journal.to_dict()
    payload.pop("optimization_direction")

    restored = SearchJournal.from_dict(payload)
    assert restored.optimization_direction == "higher"


def test_journal_optimization_direction_setter_validates_values():
    journal = SearchJournal()

    with pytest.raises(ValueError, match="optimization_direction must be 'higher' or 'lower'"):
        journal.optimization_direction = "maximize"
