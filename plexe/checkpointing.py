"""
Checkpointing functionality for plexe.

Provides serialization/deserialization of workflow state to enable:
1. Fault tolerance - resume from last completed phase on failure
2. Long-running builds - pause and resume across sessions

Core workflow saves checkpoints to LOCAL disk only. External persistence must be handled elsewhere.

User feedback is persisted in checkpoint JSON to support offline feedback workflows
(pause → user edits checkpoint → resume). Agents can access feedback via context.scratch["_user_feedback"].
"""

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cloudpickle

from plexe.models import BuildContext
from plexe.search.journal import SearchJournal
from plexe.search.insight_store import InsightStore

logger = logging.getLogger(__name__)


# ============================================
# Serialization Helpers
# ============================================


def pickle_to_base64(obj) -> str:
    """
    Serialize object to base64-encoded pickle string.

    Used for objects that don't have native JSON serialization
    (e.g., Keras optimizers, loss functions, sklearn Pipelines).

    Args:
        obj: Object to serialize

    Returns:
        Base64-encoded pickle string
    """
    pickled = cloudpickle.dumps(obj)
    return base64.b64encode(pickled).decode("utf-8")


def base64_to_pickle(b64_string: str):
    """
    Deserialize base64-encoded pickle string to object.

    Args:
        b64_string: Base64-encoded pickle string

    Returns:
        Deserialized object
    """
    pickled = base64.b64decode(b64_string.encode("utf-8"))
    return cloudpickle.loads(pickled)


# ============================================
# Public API
# ============================================


def save_checkpoint(
    experiment_id: str,
    phase_name: str,
    context: BuildContext,
    work_dir: Path,
    search_journal: SearchJournal | None = None,
    insight_store: InsightStore | None = None,
) -> Path | None:
    """
    Save checkpoint to local disk only.

    External persistence (S3, GCS, etc.) must be handled elsewhere.

    Args:
        experiment_id: Experiment identifier
        phase_name: Phase name (e.g., "analyze_data", "prepare_data", "search_models")
        context: BuildContext with workflow state
        work_dir: Working directory for checkpoint storage
        search_journal: SearchJournal (only populated after Phase 4)
        insight_store: InsightStore (only populated after Phase 4)

    Returns:
        Path to saved checkpoint file, or None if failed
    """
    logger.info(f"Saving checkpoint for phase: {phase_name}")

    try:
        # Extract user feedback from context scratch space (if provided)
        user_feedback = context.scratch.get("_user_feedback") if hasattr(context, "scratch") else None

        checkpoint_data = {
            "experiment_id": experiment_id,
            "phase": phase_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "plexe-v1",
            "context": context.to_dict() if hasattr(context, "to_dict") else {},
            "user_feedback": user_feedback,  # Persist feedback for offline editing and audit trails
            "search_journal": (
                search_journal.to_dict() if search_journal and hasattr(search_journal, "to_dict") else None
            ),
            "insight_store": insight_store.to_dict() if insight_store and hasattr(insight_store, "to_dict") else None,
        }

        # Save to local disk
        checkpoint_dir = work_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{phase_name}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        logger.info(f"✓ Checkpoint saved locally: {checkpoint_path}")
        if user_feedback:
            logger.info("✓ User feedback persisted in checkpoint for future agent use")
        return checkpoint_path

    except Exception as e:
        logger.error(f"Failed to save checkpoint locally: {e}", exc_info=True)
        return None


def load_checkpoint(
    phase_name: str,
    work_dir: Path,
) -> dict | None:
    """
    Load checkpoint from local disk.

    External download (from S3, etc.) must be handled elsewhere before calling this.

    Args:
        phase_name: Phase name to load
        work_dir: Working directory containing checkpoints

    Returns:
        Checkpoint data dict, or None if not found
    """
    try:
        checkpoint_path = work_dir / "checkpoints" / f"{phase_name}.json"
        if not checkpoint_path.exists():
            logger.debug(f"Checkpoint not found locally: {checkpoint_path}")
            return None

        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        logger.info(f"✓ Checkpoint loaded from local disk: {checkpoint_path}")
        return checkpoint_data

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None
