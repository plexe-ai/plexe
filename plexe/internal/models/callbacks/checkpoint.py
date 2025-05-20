"""
Checkpoint callback for model building process in Plexe.

This module provides a checkpoint callback implementation that saves model state
at regular intervals during the build process.
"""

import logging
from typing import Optional

from plexe.callbacks import Callback, BuildStateInfo
import plexe.fileio as fileio  # Import the module, not individual functions

logger = logging.getLogger(__name__)


class ModelCheckpointCallback(Callback):
    """
    Callback that saves model state checkpoints during the build process.

    This callback periodically saves the model state after each iteration to allow
    resuming a build from a checkpoint if the process is interrupted.
    """

    def __init__(
        self,
        keep_n_latest: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        delete_on_success: Optional[bool] = None,
    ):
        """
        Initialize the model checkpoint callback.

        Args:
            keep_n_latest: Number of most recent checkpoints to keep for this model
            checkpoint_dir: Optional custom directory for checkpoints
            delete_on_success: Whether to delete checkpoints when build completes successfully
        """
        from plexe.config import config

        # Use provided values or defaults from config
        self.keep_n_latest = keep_n_latest if keep_n_latest is not None else config.file_storage.keep_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.delete_on_success = (
            delete_on_success if delete_on_success is not None else config.file_storage.delete_checkpoints_on_success
        )
        self.model = None
        self.checkpoints = []

    def on_build_start(self, info: BuildStateInfo) -> None:
        """
        Store reference to the model on build start.

        Args:
            info: Information about the model building process start
        """
        # We need access to the model object to create checkpoints
        # This is not directly accessible via BuildStateInfo, so we'll
        # need to get a reference to it when the callback is registered with the model
        pass

    def on_iteration_end(self, info: BuildStateInfo) -> None:
        """
        Create a checkpoint after each iteration.

        Args:
            info: Information about the iteration end
        """
        if not hasattr(info, "model"):
            logger.warning("Cannot create checkpoint: no model reference available")
            return

        model = info.model
        try:
            # Save checkpoint with current iteration number
            checkpoint_path = fileio.save_checkpoint(model, info.iteration, self.checkpoint_dir)
            self.checkpoints.append(checkpoint_path)
            logger.info(f"Created checkpoint at {checkpoint_path}")

            # Manage checkpoint retention
            self._manage_checkpoints(model.identifier)
        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")

    def on_build_end(self, info: BuildStateInfo) -> None:
        """
        Optionally clean up checkpoints when build completes successfully.

        Args:
            info: Information about the model building process end
        """
        if not self.delete_on_success:
            return

        if not hasattr(info, "model"):
            return

        model = info.model

        if model.state.name == "READY":
            # Build completed successfully, clean up checkpoints if configured
            for checkpoint in self.checkpoints:
                try:
                    fileio.delete_checkpoint(checkpoint)
                    logger.info(f"Deleted checkpoint {checkpoint} after successful build")
                except Exception as e:
                    logger.error(f"Error deleting checkpoint {checkpoint}: {e}")

    def _manage_checkpoints(self, model_id: str) -> None:
        """
        Manage checkpoint retention policy.

        Args:
            model_id: Model identifier to filter checkpoints
        """
        # Get all checkpoints for this model
        all_checkpoints = fileio.list_checkpoints(model_id)

        # Sort by modification time (newest first)
        all_checkpoints.sort(key=lambda p: str(p), reverse=True)

        # Delete older checkpoints beyond our retention limit
        if len(all_checkpoints) > self.keep_n_latest:
            for checkpoint in all_checkpoints[self.keep_n_latest :]:
                try:
                    fileio.delete_checkpoint(checkpoint)
                    logger.info(f"Deleted old checkpoint {checkpoint} (retention policy)")
                except Exception as e:
                    logger.error(f"Error deleting old checkpoint {checkpoint}: {e}")
