"""Unit tests for core model dataclasses."""

from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline

from plexe.models import BuildContext, Solution


def test_build_context_update_and_unknown_key():
    """Update should set known fields and reject unknown keys."""
    context = BuildContext(
        user_id="u1",
        experiment_id="e1",
        dataset_uri="/tmp/data.parquet",
        work_dir=Path("/tmp/work"),
        intent="predict",
    )

    context.update(intent="classify")
    assert context.intent == "classify"

    with pytest.raises(ValueError, match="no attribute"):
        context.update(not_a_field=123)


# ============================================
# Solution train_performance tests
# ============================================


def _make_solution(**kwargs) -> Solution:
    defaults = dict(
        solution_id=1,
        feature_pipeline=Pipeline([]),
        model=None,
        model_type="xgboost",
    )
    defaults.update(kwargs)
    return Solution(**defaults)


def test_solution_train_performance_defaults_to_none():
    """New field should default to None for backward compatibility."""
    sol = _make_solution()
    assert sol.train_performance is None


def test_solution_to_dict_includes_train_performance():
    """to_dict should serialize train_performance."""
    sol = _make_solution(performance=0.85, train_performance=0.95)
    d = sol.to_dict()
    assert d["train_performance"] == 0.95
    assert d["performance"] == 0.85


def test_solution_from_dict_backward_compatible():
    """Old checkpoints missing train_performance should deserialize cleanly."""
    d = {
        "solution_id": 1,
        "model_type": "xgboost",
        "performance": 0.85,
        # no "train_performance" key
    }
    sol = Solution.from_dict(d, {})
    assert sol.performance == 0.85
    assert sol.train_performance is None


def test_solution_from_dict_with_train_performance():
    """Checkpoints with train_performance should round-trip correctly."""
    sol = _make_solution(performance=0.85, train_performance=0.93)
    d = sol.to_dict()
    restored = Solution.from_dict(d, {})
    assert restored.train_performance == 0.93
    assert restored.performance == 0.85
