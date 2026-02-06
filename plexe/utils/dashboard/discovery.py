"""
Experiment discovery and metadata extraction for dashboard.

Scans workdir at correct depth (dataset_name/timestamp/) and loads checkpoint metadata.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for a discovered experiment."""

    # Identity
    dataset_name: str  # First level folder name
    timestamp: str  # Second level folder name
    experiment_id: str  # From checkpoint
    path: Path  # Full path to experiment directory

    # Status
    status: str  # "completed", "running", "failed", "unknown"
    current_phase: str | None  # e.g., "search_models"
    phase_number: int  # 0-6 (0=not started, 6=completed)

    # Performance
    best_performance: float | None
    metric_name: str | None

    # Timing
    last_modified: datetime
    intent: str | None  # User's task description


def discover_experiments(workdir: Path) -> list[ExperimentMetadata]:
    """
    Discover all experiments in workdir.

    Scans at 1 or 2-level depth:
    - workdir/{dataset_name}/checkpoints/ (1-level)
    - workdir/{dataset_name}/{timestamp}/checkpoints/ (2-level)

    Args:
        workdir: Root working directory (e.g., ./workdir)

    Returns:
        List of ExperimentMetadata, sorted by last_modified (newest first)
    """
    experiments = []

    if not workdir.exists():
        logger.warning(f"Workdir does not exist: {workdir}")
        return experiments

    # Scan first level (dataset names)
    for dataset_dir in workdir.iterdir():
        if not dataset_dir.is_dir():
            continue

        # Check if this is a 1-level structure (direct checkpoints)
        checkpoints_dir = dataset_dir / "checkpoints"
        if checkpoints_dir.exists():
            # This is a 1-level structure
            try:
                metadata = _extract_metadata(
                    dataset_name=dataset_dir.name,
                    timestamp=dataset_dir.name,  # Use dataset name as timestamp for 1-level
                    experiment_path=dataset_dir,
                )
                experiments.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to extract metadata from {dataset_dir}: {e}")

        # Also check for 2-level structure (subdirectories with timestamps)
        for timestamp_dir in dataset_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            # Skip special directories like 'checkpoints' to avoid conflicts
            if timestamp_dir.name == "checkpoints":
                continue

            # Check if this looks like an experiment (has checkpoints/)
            checkpoints_dir = timestamp_dir / "checkpoints"
            if not checkpoints_dir.exists():
                continue

            # Extract metadata
            try:
                metadata = _extract_metadata(
                    dataset_name=dataset_dir.name,
                    timestamp=timestamp_dir.name,
                    experiment_path=timestamp_dir,
                )
                experiments.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to extract metadata from {timestamp_dir}: {e}")
                continue

    # Sort by last modified (newest first)
    experiments.sort(key=lambda x: x.last_modified, reverse=True)

    return experiments


def _extract_metadata(dataset_name: str, timestamp: str, experiment_path: Path) -> ExperimentMetadata:
    """Extract metadata from experiment directory."""

    checkpoints_dir = experiment_path / "checkpoints"

    # Find all checkpoint files
    checkpoint_files = list(checkpoints_dir.glob("*.json"))
    if not checkpoint_files:
        raise ValueError("No checkpoint files found")

    # Load the most recent checkpoint to get current state
    latest_checkpoint_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    with open(latest_checkpoint_file) as f:
        latest_checkpoint = json.load(f)

    # Determine phase progression
    phase_map = {
        "analyze_data": 1,
        "prepare_data": 2,
        "build_baselines": 3,
        "search_models": 4,
        "evaluate_final": 5,
        "package_final_model": 6,
    }

    current_phase = latest_checkpoint.get("phase")
    phase_number = phase_map.get(current_phase, 0)

    # Determine status
    checkpoint_status = latest_checkpoint.get("status", "completed")
    if checkpoint_status == "in_progress":
        status = "running"
    elif current_phase == "package_final_model" and checkpoint_status == "completed":
        status = "completed"
    else:
        # Check if there's an error or if it looks abandoned
        if "error" in latest_checkpoint:
            status = "failed"
        else:
            # If last modified > 1 hour ago and not completed, probably failed/abandoned
            last_modified = datetime.fromtimestamp(latest_checkpoint_file.stat().st_mtime)
            age_hours = (datetime.now() - last_modified).total_seconds() / 3600
            if age_hours > 1 and phase_number < 6:
                status = "paused"
            else:
                status = "running" if phase_number < 6 else "completed"

    # Extract performance info
    best_performance = None
    metric_name = None

    context = latest_checkpoint.get("context", {})
    if context:
        # Try to get metric name
        metric_data = context.get("metric", {})
        if isinstance(metric_data, dict):
            metric_name = metric_data.get("name")

        # Try to get best performance from baseline or search journal
        baseline = context.get("heuristic_baseline", {})
        if isinstance(baseline, dict):
            best_performance = baseline.get("performance")

    # Check search journal for better performance
    search_journal = latest_checkpoint.get("search_journal", {})
    if isinstance(search_journal, dict):
        journal_best = search_journal.get("best_performance")
        if journal_best is not None:
            best_performance = journal_best

    # Extract intent
    intent = context.get("intent") if context else None

    # Get last modified time
    last_modified = datetime.fromtimestamp(latest_checkpoint_file.stat().st_mtime)

    # Extract experiment_id
    experiment_id = context.get("experiment_id") if context else timestamp

    return ExperimentMetadata(
        dataset_name=dataset_name,
        timestamp=timestamp,
        experiment_id=experiment_id or timestamp,
        path=experiment_path,
        status=status,
        current_phase=current_phase,
        phase_number=phase_number,
        best_performance=best_performance,
        metric_name=metric_name,
        last_modified=last_modified,
        intent=intent,
    )


def load_experiment_checkpoints(experiment_path: Path) -> dict[str, dict]:
    """
    Load all checkpoints for an experiment.

    Args:
        experiment_path: Path to experiment directory

    Returns:
        Dict mapping phase_name -> checkpoint_data
    """
    checkpoints_dir = experiment_path / "checkpoints"
    if not checkpoints_dir.exists():
        return {}

    checkpoints = {}
    for checkpoint_file in checkpoints_dir.glob("*.json"):
        # Strip number prefix from filename (e.g., "02_prepare_data" -> "prepare_data")
        stem = checkpoint_file.stem
        phase_name = stem.split("_", 1)[1] if "_" in stem else stem
        try:
            with open(checkpoint_file) as f:
                checkpoints[phase_name] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
            continue

    return checkpoints
