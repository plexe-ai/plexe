"""Unit tests for dashboard experiment discovery."""

import json
from pathlib import Path

from plexe.utils.dashboard.discovery import discover_experiments


def _write_checkpoint(checkpoints_dir: Path, filename: str, phase: str, intent: str = "test intent") -> None:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "status": "completed",
        "context": {
            "intent": intent,
            "experiment_id": "local",
            "metric": {"name": "roc_auc"},
        },
    }
    (checkpoints_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_discover_flat_workdir_layout(tmp_path: Path) -> None:
    """Standalone runs write checkpoints/ directly under --work-dir."""
    _write_checkpoint(tmp_path / "checkpoints", "06_package_final_model.json", "06_package_final_model")

    experiments = discover_experiments(tmp_path)

    assert len(experiments) == 1
    assert experiments[0].path == tmp_path
    assert experiments[0].phase_number == 6
    assert experiments[0].status == "completed"
    assert experiments[0].intent == "test intent"


def test_discover_nested_one_level_layout(tmp_path: Path) -> None:
    """Nested dataset folders remain discoverable."""
    exp_dir = tmp_path / "weatherAUS"
    _write_checkpoint(exp_dir / "checkpoints", "03_build_baselines.json", "03_build_baselines")

    experiments = discover_experiments(tmp_path)

    assert len(experiments) == 1
    assert experiments[0].path == exp_dir
    assert experiments[0].phase_number == 3
