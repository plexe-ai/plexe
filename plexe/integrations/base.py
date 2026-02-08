"""
Base integration interface for connecting plexe to external infrastructure.

Defines the contract between the model-building workflow and the environment
it runs in (storage, tracking, model registries, etc.). Implement this interface
to integrate plexe with your platform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class WorkflowIntegration(ABC):
    """
    Integration interface for environment-specific infrastructure.

    Implement this to connect plexe to your storage, tracking, and
    model registry systems. See ``StandaloneIntegration`` for a reference
    implementation.

    Methods fall into three categories:

    **Workspace & data preparation** (called before/during workflow):
    - ``prepare_workspace`` — restore state from durable storage if resuming
    - ``get_artifact_location`` — resolve output paths for intermediate artifacts
    - ``ensure_local`` — download remote URIs to local filesystem
    - ``prepare_original_model`` — fetch a previously trained model for retraining

    **Artifact persistence** (called after workflow phases):
    - ``on_checkpoint`` — persist checkpoint and work directory
    - ``on_completion`` — persist final model and update tracking

    **Lifecycle notifications** (optional, have default no-op implementations):
    - ``on_failure`` — handle workflow failure
    - ``on_pause`` — handle workflow pause for user feedback
    """

    @abstractmethod
    def prepare_workspace(
        self,
        experiment_id: str,
        work_dir: Path,
    ) -> None:
        """
        Prepare workspace for a model-building run.

        Restore previous work directory from durable storage if resuming,
        and create any necessary directory structure. Dataset normalization
        is handled separately by the workflow.

        Args:
            experiment_id: Experiment identifier
            work_dir: Local working directory
        """

    @abstractmethod
    def get_artifact_location(
        self,
        artifact_type: str,
        dataset_uri: str,
        experiment_id: str,
        work_dir: Path,
    ) -> str:
        """
        Determine where an intermediate artifact should be written.

        Args:
            artifact_type: One of ``"splits"``, ``"samples"``, ``"transformed"``, ``"normalized"``
            dataset_uri: Input dataset URI (may influence storage location)
            experiment_id: Experiment identifier
            work_dir: Local working directory

        Returns:
            Output location (local path or remote URI)
        """

    @abstractmethod
    def ensure_local(self, uris: list[str], work_dir: Path) -> list[str]:
        """
        Ensure remote URIs are available on the local filesystem.

        For local paths, this is a no-op. For remote URIs (S3, etc.),
        downloads the data to a local directory under ``work_dir``.

        Args:
            uris: List of URIs (local or remote)
            work_dir: Local working directory for downloads

        Returns:
            List of local filesystem paths
        """

    @abstractmethod
    def prepare_original_model(self, model_reference: str, work_dir: Path) -> str:
        """
        Locate and download an existing model for retraining.

        How ``model_reference`` is interpreted depends on the implementation:
        a local path, an S3 URI, an experiment ID, a model registry key, etc.

        Args:
            model_reference: Model reference (implementation-specific)
            work_dir: Working directory for staging

        Returns:
            Local filesystem path to model.tar.gz
        """

    @abstractmethod
    def on_checkpoint(
        self,
        experiment_id: str,
        phase_name: str,
        checkpoint_path: Path,
        work_dir: Path,
    ) -> None:
        """
        Persist checkpoint and work directory after a phase completes.

        Called after each workflow phase. Implementations typically upload
        the checkpoint JSON and work directory to durable storage, and
        update any experiment tracking systems.

        Args:
            experiment_id: Experiment identifier
            phase_name: Phase name (e.g., ``"01_analyze_data"``)
            checkpoint_path: Path to checkpoint JSON file on local disk
            work_dir: Working directory containing all artifacts
        """

    @abstractmethod
    def on_completion(
        self,
        experiment_id: str,
        work_dir: Path,
        final_metrics: dict,
        evaluation_report: Any,
    ) -> None:
        """
        Persist final model and update tracking on successful completion.

        Args:
            experiment_id: Experiment identifier
            work_dir: Working directory containing final model package
            final_metrics: Final evaluation metrics
            evaluation_report: Full evaluation report (None if evaluation not run)
        """

    def on_failure(self, experiment_id: str, error: Exception) -> None:
        """
        Handle workflow failure.

        Default: no-op (error is already logged by the workflow).
        Override to update experiment tracking, send alerts, etc.

        Args:
            experiment_id: Experiment identifier
            error: Exception that caused failure
        """

    def on_pause(self, phase_name: str) -> None:
        """
        Handle workflow pause for user feedback.

        Default: no-op. Override to update experiment status in tracking systems.

        Args:
            phase_name: Name of the phase where workflow paused
        """
