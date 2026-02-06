"""
Standalone adapter for development and testing.

Provides minimal infrastructure - no Plexe platform integration.
Supports optional external S3 storage for persistence.
"""

import json
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from plexe.adapters.base import WorkflowAdapter
from plexe.constants import DirNames
from plexe.execution.dataproc.dataset_io import DatasetNormalizer
from plexe.execution.dataproc.session import get_or_create_spark_session

logger = logging.getLogger(__name__)


class StandaloneAdapter(WorkflowAdapter):
    """
    Standalone adapter for local development and testing.

    Provides minimal infrastructure with no Plexe platform integration:
    - Optional S3 storage for checkpoints, workdir, and final model (when external_storage_uri provided)
    - No DynamoDB experiment tracking
    - No Secrets Manager integration
    - Works with both local files and S3 datasets (if AWS credentials available)

    Suitable for:
    - Local development and debugging
    - Testing without platform dependencies
    - Standalone deployments outside Plexe platform (e.g., Flyte workflows)
    - Cloud storage persistence without full platform integration

    Databricks Support:
    - Databricks mode IS supported when using S3 datasets with --external-storage-uri
    """

    def __init__(self, config, external_storage_uri: str | None = None, user_id: str | None = None):
        """
        Initialize standalone adapter.

        Args:
            config: Config object with workflow settings
            external_storage_uri: Optional S3 URI for external storage
                                 (e.g., s3://bucket/prefix)
                                 Used for:
                                 - Intermediate datasets (splits, transformed, normalized)
                                 - Workdir backups (for fault tolerance)
            user_id: User identifier for S3 path scoping (users/{user_id}/experiments/...)
        """
        self.config = config
        self.external_storage_uri = external_storage_uri
        self.user_id = user_id

        # Initialize S3 client if using S3 storage
        # TODO(storage-abstraction): Replace with pluggable storage backend (S3Storage, AzureBlobStorage, etc.)
        if external_storage_uri and external_storage_uri.startswith("s3://"):
            self.s3_client = boto3.client("s3")
            self.bucket, self.prefix = self._parse_s3_uri(external_storage_uri)
        else:
            self.s3_client = None
            self.bucket = None
            self.prefix = None

    def setup_environment(self):
        """No-op - standalone mode uses environment variables as-is."""
        logger.debug("Standalone mode: no environment setup needed")

    def prepare_workspace(
        self,
        experiment_id: str,
        data_refs: list[str],
        work_dir: Path,
    ) -> tuple[str, str]:
        """
        Prepare workspace and normalize dataset format.

        Accepts both local files and S3 URIs. Detects format and converts to Parquet if needed.
        If external_storage_uri is configured, attempts to restore previous workdir from S3.

        Args:
            experiment_id: Experiment identifier
            data_refs: List of dataset URIs (local paths or s3:// URIs)
            work_dir: Local working directory

        Returns:
            (dataset_uri, detected_format): Parquet URI for workflow + original format string
        """
        # Try to restore workdir from S3 if available (enables resume)
        if self.s3_client:
            workdir_key = self._s3_key(experiment_id, "workdir", "workdir.tar.gz")
            if self._s3_object_exists(workdir_key):
                logger.info("Found existing workdir in S3, restoring...")
                build_dir = work_dir / DirNames.BUILD_DIR
                temp_extract = work_dir / ".build_restore_tmp"

                try:
                    # Download and extract to temporary location first (atomic operation)
                    if temp_extract.exists():
                        shutil.rmtree(temp_extract)
                    temp_extract.mkdir(parents=True, exist_ok=True)

                    self._download_and_extract_tarball(workdir_key, temp_extract)

                    # Only after successful extraction, atomically swap directories
                    restored_build = temp_extract / DirNames.BUILD_DIR
                    if not restored_build.exists():
                        # Log extracted contents for debugging
                        extracted = [str(p.relative_to(temp_extract)) for p in temp_extract.rglob("*")]
                        logger.error(
                            f"Extracted tarball missing {DirNames.BUILD_DIR} directory. "
                            f"Contents: {extracted[:10] or '[empty]'}"
                        )
                        raise ValueError(f"Extracted tarball missing {DirNames.BUILD_DIR} directory")

                    if build_dir.exists():
                        logger.info("Removing existing .build directory")
                        shutil.rmtree(build_dir)

                    shutil.move(str(restored_build), str(build_dir))
                    shutil.rmtree(temp_extract)

                    logger.info("✓ Restored workdir from S3")
                except Exception as e:
                    logger.warning(f"Failed to restore workdir from S3: {e}")
                    # Cleanup temp directory if it exists
                    if temp_extract.exists():
                        shutil.rmtree(temp_extract)
            else:
                logger.debug("No previous workdir found in S3 - starting fresh")

        if not data_refs:
            raise ValueError("No dataset references provided")

        input_dataset_uri = data_refs[0]

        # Determine output location for normalized dataset
        if input_dataset_uri.startswith("s3://"):
            if not self.external_storage_uri:
                raise ValueError(
                    f"S3 dataset requires --external-storage-uri for normalization.\n"
                    f"Dataset: {input_dataset_uri}\n"
                    f"Provide: --external-storage-uri s3://your-bucket/prefix/"
                )
            s3_key = self._s3_key(experiment_id, "datasets", "normalized", "dataset.parquet")
            output_uri = f"s3://{self.bucket}/{s3_key}"
        else:
            output_uri = str(work_dir / DirNames.BUILD_DIR / "normalized" / "dataset.parquet")

        # Normalize dataset format
        spark = get_or_create_spark_session(self.config)
        normalizer = DatasetNormalizer(spark)

        # Build CSV options from config
        csv_options = {
            "sep": self.config.csv_delimiter,
            "header": self.config.csv_header,
        }

        dataset_uri, detected_format = normalizer.normalize(
            input_uri=input_dataset_uri,
            output_uri=output_uri,
            read_options=csv_options,
        )

        logger.info(f"Dataset ready: {dataset_uri} (original format: {detected_format})")
        return dataset_uri, detected_format.value

    def on_checkpoint(
        self,
        phase_name: str,
        checkpoint_path: Path,
        work_dir: Path,
    ):
        """
        Upload checkpoint and workdir to S3 if external storage configured.

        Args:
            phase_name: Phase name
            checkpoint_path: Local checkpoint file
            work_dir: Working directory
        """
        if not self.s3_client:
            logger.debug(f"Local mode: checkpoint saved at {checkpoint_path}")
            return

        # Extract experiment_id from checkpoint JSON
        try:
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)
                experiment_id = checkpoint_data.get("experiment_id", "unknown")
        except Exception as e:
            logger.warning(f"Failed to read experiment_id from checkpoint: {e}")
            experiment_id = "unknown"

        try:
            # Upload checkpoint JSON (separate from workdir for fast resume checks)
            checkpoint_key = self._s3_key(experiment_id, "checkpoints", f"{phase_name}.json")
            self._upload_file_to_s3(checkpoint_path, checkpoint_key)
            logger.info(f"✓ Uploaded checkpoint to S3: s3://{self.bucket}/{checkpoint_key}")

            # Tar and upload .build/ directory
            build_dir = work_dir / DirNames.BUILD_DIR
            if build_dir.exists():
                workdir_key = self._s3_key(experiment_id, "workdir", "workdir.tar.gz")
                self._tar_and_upload_directory(build_dir, workdir_key)
                logger.info(f"✓ Uploaded workdir to S3: s3://{self.bucket}/{workdir_key}")
            else:
                logger.warning(".build/ directory not found, skipping workdir upload")

        except (ClientError, OSError) as e:
            logger.warning(f"S3 upload failed (non-critical): {e}")

    def on_completion(
        self,
        experiment_id: str,
        work_dir: Path,
        final_metrics: dict,
        evaluation_report,  # EvaluationReport | None
    ):
        """
        Upload final model to S3 if external storage configured.

        Args:
            experiment_id: Experiment identifier
            work_dir: Working directory with final model
            final_metrics: Final metrics
            evaluation_report: Full evaluation report (None if evaluation not run)
        """
        if not self.s3_client:
            logger.info(f"Local mode: model available at {work_dir / 'model'}")
            return

        try:
            # Upload final model package
            model_tarball = work_dir / "model.tar.gz"
            if not model_tarball.exists():
                logger.warning(f"Model tarball not found: {model_tarball}")
                return

            logger.info(f"Uploading model package ({model_tarball.stat().st_size / 1024 / 1024:.1f} MB)...")
            model_key = self._s3_key(experiment_id, "model", "model.tar.gz")
            uri = self._upload_file_to_s3(model_tarball, model_key)
            logger.info(f"✓ Final model uploaded: {uri}")

        except Exception as e:
            logger.error(f"Failed to upload model to S3: {e}")
            raise

    def on_failure(
        self,
        experiment_id: str,
        error: Exception,
    ):
        """
        No-op - error already logged.

        Args:
            experiment_id: Experiment identifier
            error: Exception that caused failure
        """
        logger.debug(f"Local mode: experiment {experiment_id} failed: {error}")

    def prepare_original_model(
        self,
        model_reference: str,
        work_dir: Path,
    ) -> str:
        """
        Prepare original model for retraining.

        Model reference can be:
        - Local file path: /path/to/model.tar.gz
        - S3 URI: s3://bucket/path/to/model.tar.gz (when external_storage_uri configured)
        - Experiment ID: experiment_123 (will look in {external_storage_uri}/models/{id}/model.tar.gz)

        Args:
            model_reference: Local path, S3 URI, or experiment ID
            work_dir: Working directory for downloads

        Returns:
            Local filesystem path to model.tar.gz

        Raises:
            FileNotFoundError: If model doesn't exist
            ValueError: If reference is invalid
        """
        # Case 1: S3 URI
        if model_reference.startswith("s3://"):
            if not self.s3_client:
                raise ValueError(
                    f"S3 model reference requires --external-storage-uri.\n"
                    f"Model: {model_reference}\n"
                    f"Provide: --external-storage-uri s3://your-bucket/prefix/"
                )

            logger.info(f"Downloading original model from S3: {model_reference}")
            try:
                # Parse S3 URI using helper method
                bucket, key = self._parse_s3_uri(model_reference)
                if not key:
                    raise ValueError(
                        f"Invalid S3 model reference, missing object key: {model_reference}\n"
                        f"Expected format: s3://bucket/path/to/model.tar.gz"
                    )

                # Download to local path
                original_model_dir = work_dir / "original_model"
                original_model_dir.mkdir(parents=True, exist_ok=True)
                local_model_path = original_model_dir / Path(key).name

                # Reuse existing S3 client (can access any bucket)
                self.s3_client.download_file(bucket, key, str(local_model_path))

                file_size_mb = local_model_path.stat().st_size / (1024**2)
                logger.info(f"✓ Downloaded original model ({file_size_mb:.1f} MB): {local_model_path}")
                return str(local_model_path)

            except ClientError as e:
                self._handle_s3_download_error(e, model_reference)
            except ValueError:
                # Re-raise validation errors (e.g., invalid S3 URI format)
                raise
            except Exception as e:
                logger.error(f"Unexpected error downloading model from S3: {e}")
                raise

        # Case 2: Experiment ID (if external_storage_uri configured)
        if self.s3_client and not Path(model_reference).exists():
            logger.warning(
                f"Local model file not found at '{model_reference}'. "
                f"Attempting to download as experiment ID from external storage."
            )
            model_key = self._s3_key(model_reference, "model", "model.tar.gz")

            if self._s3_object_exists(model_key):
                logger.info(f"Downloading model from s3://{self.bucket}/{model_key}...")
                try:
                    original_model_dir = work_dir / "original_model"
                    original_model_dir.mkdir(parents=True, exist_ok=True)
                    local_model_path = original_model_dir / "model.tar.gz"

                    self._download_file_from_s3(model_key, local_model_path)

                    file_size_mb = local_model_path.stat().st_size / (1024**2)
                    logger.info(f"✓ Downloaded original model ({file_size_mb:.1f} MB): {local_model_path}")
                    return str(local_model_path)

                except ClientError as e:
                    self._handle_s3_download_error(e, model_reference, " for experiment")
                except Exception as e:
                    logger.error(f"Unexpected error downloading model for experiment: {e}")
                    raise

        # Case 3: Local file path
        model_path = Path(model_reference)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_reference}\n"
                f"Provide either:\n"
                f"  - Local file path: /path/to/model.tar.gz\n"
                f"  - S3 URI: s3://bucket/path/model.tar.gz\n"
                f"  - Experiment ID: exp_123 (requires --external-storage-uri)"
            )

        if not model_path.is_file():
            raise ValueError(f"Model path is not a file: {model_reference}")

        logger.info(f"Using local model for retraining: {model_path}")
        return str(model_path)

    def on_pause(
        self,
        phase_name: str,
    ):
        """
        No-op - local mode has no external state tracking.

        Args:
            phase_name: Name of the phase where workflow paused
        """
        logger.debug(f"Local mode: workflow paused at {phase_name}")

    def get_splits_output_location(self, dataset_uri: str, experiment_id: str, work_dir: Path) -> str:
        return self._get_storage_location(dataset_uri, experiment_id, work_dir, "splits")

    def get_samples_output_location(self, dataset_uri: str, experiment_id: str, work_dir: Path) -> str:
        return self._get_storage_location(dataset_uri, experiment_id, work_dir, "samples")

    def get_transformed_output_location(self, dataset_uri: str, experiment_id: str, work_dir: Path) -> str:
        return self._get_storage_location(dataset_uri, experiment_id, work_dir, "transformed")

    def ensure_samples_local(self, sample_uris: list[str], work_dir: Path) -> list[str]:
        """Download S3 samples to local if needed (handles Spark parquet directories)."""
        # Reuse existing S3 client or create one if not configured
        s3_client = self.s3_client if self.s3_client else boto3.client("s3")
        local_uris = []

        for uri in sample_uris:
            if uri.startswith("s3://"):
                parts = uri[5:].split("/", 1)
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(f"Invalid S3 sample URI (expected s3://bucket/key): {uri}")
                bucket, prefix = parts[0], parts[1]

                # Spark writes parquet as directory - list all objects under prefix
                logger.info(f"Downloading sample directory from S3: {uri}")
                response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

                if "Contents" not in response:
                    raise ValueError(f"No objects found at {uri}")

                # Create local directory
                local_dir = work_dir / DirNames.BUILD_DIR / "data" / "samples_local" / Path(prefix).name
                local_dir.mkdir(parents=True, exist_ok=True)

                # Download all parquet parts
                for obj in response["Contents"]:
                    obj_key = obj["Key"]
                    relative_path = obj_key[len(prefix) :].lstrip("/")
                    local_file = local_dir / relative_path

                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    s3_client.download_file(bucket, obj_key, str(local_file))

                logger.info(f"Downloaded {len(response['Contents'])} files to {local_dir}")
                local_uris.append(str(local_dir))
            else:
                local_uris.append(uri)
        return local_uris

    def _get_storage_location(self, dataset_uri: str, experiment_id: str, work_dir: Path, artifact_type: str) -> str:
        """Determine storage location for intermediate datasets."""
        if dataset_uri.startswith("s3://"):
            if not self.external_storage_uri:
                raise ValueError(
                    f"S3 dataset requires --external-storage-uri for {artifact_type} storage.\n"
                    f"Dataset: {dataset_uri}\n"
                    f"Provide: --external-storage-uri s3://your-bucket/prefix/"
                )
            # Samples stay local (fast iteration), others go to S3 datasets/
            if artifact_type == "samples":
                return str(work_dir / DirNames.BUILD_DIR / "data" / "samples")
            else:
                # splits, transformed, normalized go to S3 datasets/
                s3_key = self._s3_key(experiment_id, "datasets", artifact_type)
                return f"s3://{self.bucket}/{s3_key}"
        else:
            return str(work_dir / DirNames.BUILD_DIR / "data" / artifact_type)

    # ============================================
    # S3 Helper Methods (TODO: Extract to storage backend abstraction)
    # ============================================

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """
        Parse s3://bucket/prefix into (bucket, prefix).

        Args:
            uri: S3 URI (e.g., s3://my-bucket/model-builder)

        Returns:
            (bucket, prefix) tuple
        """
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
        return bucket, prefix

    def _s3_key(self, experiment_id: str, *parts: str) -> str:
        """
        Build S3 key with users/{user_id}/experiments/{experiment_id}/ prefix.

        Args:
            experiment_id: Experiment identifier
            *parts: Additional path components

        Returns:
            Full S3 key with user/experiment scoping
        """
        if self.user_id:
            base_parts = [self.prefix, "users", self.user_id, "experiments", experiment_id]
        else:
            # Fallback when user_id not provided
            base_parts = [self.prefix, experiment_id]

        key_parts = [p for p in base_parts + list(parts) if p]
        return "/".join(key_parts)

    def _upload_file_to_s3(self, local_path: Path, s3_key: str) -> str:
        """
        Upload file to S3.

        Args:
            local_path: Local file path
            s3_key: S3 key (without bucket)

        Returns:
            Full S3 URI (s3://bucket/key)
        """
        self.s3_client.upload_file(str(local_path), self.bucket, s3_key)
        return f"s3://{self.bucket}/{s3_key}"

    def _download_file_from_s3(self, s3_key: str, local_path: Path) -> None:
        """
        Download file from S3.

        Args:
            s3_key: S3 key (without bucket)
            local_path: Local destination path
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3_client.download_file(self.bucket, s3_key, str(local_path))

    def _s3_object_exists(self, s3_key: str) -> bool:
        """
        Check if S3 object exists.

        Args:
            s3_key: S3 key (without bucket)

        Returns:
            True if object exists

        Raises:
            ClientError: If S3 access fails for reasons other than "not found"
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            # Re-raise for access denied, invalid credentials, etc.
            logger.error(f"S3 access error for s3://{self.bucket}/{s3_key}: {e}")
            raise

    def _handle_s3_download_error(self, error: ClientError, model_reference: str, context: str = "") -> None:
        """Handle S3 download ClientError and raise appropriate exception."""
        error_code = error.response["Error"]["Code"]
        if error_code in ("404", "NoSuchKey"):
            raise FileNotFoundError(f"Model not found{context}: {model_reference}") from error
        elif error_code in ("403", "AccessDenied"):
            logger.error(f"Access denied to model{context}: {model_reference}")
            raise PermissionError(f"Access denied to model{context}: {model_reference}") from error
        else:
            logger.error(f"S3 error downloading model: {error}")
            raise

    def _tar_and_upload_directory(self, local_dir: Path, s3_key: str) -> str:
        """
        Tar directory and upload to S3.

        Args:
            local_dir: Local directory to tar
            s3_key: S3 key for tarball (without bucket)

        Returns:
            Full S3 URI (s3://bucket/key)
        """
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tarball_path = tmp.name

        try:
            logger.info(f"Creating tarball from {local_dir}...")
            with tarfile.open(tarball_path, "w:gz") as tar:
                # Use explicit arcname for consistency with restoration logic
                tar.add(local_dir, arcname=DirNames.BUILD_DIR)

            tarball_size_mb = Path(tarball_path).stat().st_size / (1024**2)
            logger.info(f"Tarball size: {tarball_size_mb:.1f} MB")

            uri = self._upload_file_to_s3(Path(tarball_path), s3_key)
            return uri
        finally:
            # Cleanup temp file even on failure
            if Path(tarball_path).exists():
                os.remove(tarball_path)

    def _download_and_extract_tarball(self, s3_key: str, extract_to: Path) -> None:
        """
        Download tarball from S3 and extract.

        Args:
            s3_key: S3 key for tarball (without bucket)
            extract_to: Local directory to extract to

        Raises:
            ValueError: If tarball contains paths outside extract_to (path traversal attack)
        """
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tarball_path = tmp.name

        try:
            self._download_file_from_s3(s3_key, Path(tarball_path))

            tarball_size_mb = Path(tarball_path).stat().st_size / (1024**2)
            logger.info(f"Downloaded tarball: {tarball_size_mb:.1f} MB")

            logger.info(f"Extracting tarball to {extract_to}...")
            with tarfile.open(tarball_path, "r:gz") as tar:
                # Security: Validate all paths stay within extract_to (CVE-2007-4559)
                extract_to_resolved = extract_to.resolve()
                for member in tar.getmembers():
                    member_path = (extract_to / member.name).resolve()
                    if not member_path.is_relative_to(extract_to_resolved):
                        raise ValueError(
                            f"Tarball contains unsafe path traversal: {member.name}\n"
                            f"Attempted to extract outside {extract_to}"
                        )
                tar.extractall(path=extract_to)
        finally:
            # Cleanup temp file even on failure
            if Path(tarball_path).exists():
                os.remove(tarball_path)
