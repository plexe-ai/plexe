"""Unit tests for core model dataclasses."""

from pathlib import Path
import pytest

from plexe.models import BuildContext


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
