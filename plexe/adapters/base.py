"""
Base adapter interface for environment-specific infrastructure.

Defines the contract for integrating model-builder-v2 with external systems.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class WorkflowAdapter(ABC):
    """
    Adapter interface for environment-specific infrastructure.

    Separates core model-building logic from external environment concerns
    like storage, metadata tracking, and dataset access.
    """

    @abstractmethod
    def setup_environment(self):
        """
        Setup environment-specific configuration.

        Examples:
        - Load API keys from secrets manager
        - Configure cloud SDK credentials
        - Set environment variables
        - Populate OTEL tracing credentials in self.config

        Called once before workflow starts.
        """
        pass

    @abstractmethod
    def prepare_workspace(
        self,
        experiment_id: str,
        data_refs: list[str],
        work_dir: Path,
    ) -> tuple[str, str]:
        """
        Setup workspace and prepare datasets before workflow starts.

        Handles:
        1. Restore previous workdir from storage if exists (for resume)
        2. Download/prepare datasets
        3. Detect and normalize dataset format to Parquet
        4. Return dataset URI ready for workflow

        Args:
            experiment_id: Experiment identifier
            data_refs: Dataset references (S3 paths, dataset IDs, local paths, etc.)
            work_dir: Local working directory

        Returns:
            (dataset_uri, detected_format): Parquet URI ready for workflow + original format string
        """
        pass

    @abstractmethod
    def on_checkpoint(
        self,
        phase_name: str,
        checkpoint_path: Path,
        work_dir: Path,
    ):
        """
        Persist checkpoint and workdir to external storage.

        Called after each workflow phase completes successfully.

        Args:
            phase_name: Phase name (e.g., "analyze_data", "search_models")
            checkpoint_path: Path to checkpoint JSON file on local disk
            work_dir: Working directory containing all artifacts
        """
        pass

    @abstractmethod
    def on_completion(
        self,
        experiment_id: str,
        work_dir: Path,
        final_metrics: dict,
        evaluation_report,  # EvaluationReport | None
    ):
        """
        Upload final model and update metadata when workflow completes.

        Args:
            experiment_id: Experiment identifier
            work_dir: Working directory containing final model package
            final_metrics: Final evaluation metrics
            evaluation_report: Full evaluation report (None if evaluation not run)
        """
        pass

    @abstractmethod
    def on_failure(
        self,
        experiment_id: str,
        error: Exception,
    ):
        """
        Handle workflow failure.

        Update status, cleanup resources, etc.

        Args:
            experiment_id: Experiment identifier
            error: Exception that caused failure
        """
        pass

    @abstractmethod
    def prepare_original_model(
        self,
        model_reference: str,
        work_dir: Path,
    ) -> str:
        """
        Ensure original model is available locally for retraining.

        Each adapter interprets model_reference according to its capabilities:
        - Local adapter: expects a local file path
        - AWS adapter: expects a platform experiment ID

        The adapter is responsible for locating the model (by whatever means necessary)
        and ensuring it's available as a local file.

        Args:
            model_reference: Model reference string (adapter-specific interpretation)
            work_dir: Working directory for downloads/staging

        Returns:
            Local filesystem path to model.tar.gz (never remote URIs)

        Raises:
            ValueError: If model_reference format is invalid for this adapter
            FileNotFoundError: If referenced model doesn't exist
            RuntimeError: If download/preparation fails
        """
        pass

    @abstractmethod
    def on_pause(
        self,
        phase_name: str,
    ):
        """
        Handle workflow pause for user feedback.

        Called when workflow pauses at a checkpoint to await user review and feedback.
        Adapter should update external state tracking (e.g., DynamoDB status) if applicable.

        Args:
            phase_name: Name of the phase where workflow paused
        """
        pass

    @abstractmethod
    def get_splits_output_location(
        self,
        dataset_uri: str,
        experiment_id: str,
        work_dir: Path,
    ) -> str:
        """
        Determine where dataset splits should be written.

        Decision based on input dataset location (not Spark mode):
        - Local dataset → local splits
        - S3 dataset → S3 splits

        Args:
            dataset_uri: Input dataset URI (determines storage backend)
            experiment_id: Experiment identifier
            work_dir: Local working directory

        Returns:
            Output location for splits (local path or S3 URI)
        """
        pass

    @abstractmethod
    def get_samples_output_location(
        self,
        dataset_uri: str,
        experiment_id: str,
        work_dir: Path,
    ) -> str:
        """
        Determine where dataset samples should be written (follows same logic as splits).

        Args:
            dataset_uri: Input dataset URI
            experiment_id: Experiment identifier
            work_dir: Local working directory

        Returns:
            Output location for samples
        """
        pass

    @abstractmethod
    def get_transformed_output_location(
        self,
        dataset_uri: str,
        experiment_id: str,
        work_dir: Path,
    ) -> str:
        """
        Determine where transformed datasets should be written (follows same logic as splits).

        Args:
            dataset_uri: Input dataset URI
            experiment_id: Experiment identifier
            work_dir: Local working directory

        Returns:
            Output location for transformed data
        """
        pass

    @abstractmethod
    def ensure_samples_local(
        self,
        sample_uris: list[str],
        work_dir: Path,
    ) -> list[str]:
        """
        Ensure samples are available locally (download from S3 if needed).

        Samples must be local for training subprocess (loads into pandas).

        Args:
            sample_uris: List of sample URIs (may be local or S3)
            work_dir: Local working directory for downloads

        Returns:
            List of local file paths
        """
        pass
