"""
Standalone integration for local development and S3-backed deployments.

Serves as both the default integration and a reference implementation
for custom ``WorkflowIntegration`` subclasses.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from plexe.integrations.base import WorkflowIntegration
from plexe.constants import DirNames

logger = logging.getLogger(__name__)


class StandaloneIntegration(WorkflowIntegration):
    """
    Standalone integration for local development and testing.

    Supports two modes:
    - **Local only** (default): all artifacts on local filesystem, no cloud dependencies
    - **S3-backed**: optional S3 persistence for checkpoints, workdir, and models
      (when ``external_storage_uri`` is provided)

    Suitable for:
    - Local development and debugging
    - Standalone deployments (e.g., Flyte, Airflow)
    - Cloud storage persistence without full platform integration

    .. note:: Future direction

        Currently, S3 is the only supported remote storage backend. In future
        revisions, this class should detect the URI scheme (``s3://``, ``az://``,
        ``gs://``) and dispatch to the appropriate storage helper from
        ``plexe.integrations.storage``, rather than containing S3-specific logic directly.
    """

    def __init__(self, external_storage_uri: str | None = None, user_id: str | None = None):
        self.s3 = None
        if external_storage_uri:
            if not external_storage_uri.startswith("s3://"):
                raise ValueError(
                    f"Unsupported storage URI scheme: {external_storage_uri}\n"
                    f"StandaloneIntegration currently supports only s3:// URIs.\n"
                    f"For Azure (az://) or GCS (gs://), implement a custom WorkflowIntegration."
                )
            from plexe.integrations.storage.s3 import S3Helper

            bucket, prefix = S3Helper.parse_uri(external_storage_uri)
            self.s3 = S3Helper(bucket, prefix, user_id)
        self.external_storage_uri = external_storage_uri
        self.user_id = user_id

    def prepare_workspace(self, experiment_id: str, work_dir: Path) -> None:
        """Restore workspace from S3 if a previous run exists."""
        self._try_restore_workdir(experiment_id, work_dir)

    def get_artifact_location(
        self,
        artifact_type: str,
        dataset_uri: str,
        experiment_id: str,
        work_dir: Path,
    ) -> str:
        """Determine storage location based on dataset location."""
        if dataset_uri.startswith("s3://"):
            if not self.external_storage_uri:
                raise ValueError(
                    f"S3 dataset requires --external-storage-uri for {artifact_type} storage.\n"
                    f"Dataset: {dataset_uri}\n"
                    f"Provide: --external-storage-uri s3://your-bucket/prefix/"
                )
            if artifact_type == "samples":
                return str(work_dir / DirNames.BUILD_DIR / "data" / "samples")
            s3_key = self.s3.build_key(experiment_id, "datasets", artifact_type)
            return f"s3://{self.s3.bucket}/{s3_key}"
        return str(work_dir / DirNames.BUILD_DIR / "data" / artifact_type)

    def ensure_local(self, uris: list[str], work_dir: Path) -> list[str]:
        """Download S3 URIs to local if needed (handles Spark parquet directories)."""
        local_uris = []
        for uri in uris:
            if uri.startswith("s3://"):
                if self.s3:
                    helper = self.s3
                else:
                    from plexe.integrations.storage.s3 import S3Helper

                    helper = S3Helper(*S3Helper.parse_uri(uri))
                local_dir = work_dir / DirNames.BUILD_DIR / "data" / "samples_local"
                local_uris.append(helper.download_directory(uri, local_dir))
            else:
                local_uris.append(uri)
        return local_uris

    def prepare_original_model(self, model_reference: str, work_dir: Path) -> str:
        """
        Prepare original model for retraining.

        Accepts local paths, S3 URIs, or experiment IDs (when S3 configured).
        """
        if model_reference.startswith("s3://"):
            return self._download_model_from_s3_uri(model_reference, work_dir)

        if self.s3 and not Path(model_reference).exists():
            return self._download_model_by_experiment_id(model_reference, work_dir)

        return self._resolve_local_model(model_reference)

    def on_checkpoint(
        self,
        experiment_id: str,
        phase_name: str,
        checkpoint_path: Path,
        work_dir: Path,
    ) -> None:
        """Upload checkpoint and workdir to S3 if configured."""
        if not self.s3:
            logger.debug(f"Local mode: checkpoint saved at {checkpoint_path}")
            return

        try:
            checkpoint_key = self.s3.build_key(experiment_id, "checkpoints", f"{phase_name}.json")
            self.s3.upload_file(checkpoint_path, checkpoint_key)
            logger.info(f"Uploaded checkpoint to s3://{self.s3.bucket}/{checkpoint_key}")

            build_dir = work_dir / DirNames.BUILD_DIR
            if build_dir.exists():
                workdir_key = self.s3.build_key(experiment_id, "workdir", "workdir.tar.gz")
                self.s3.tar_and_upload(build_dir, workdir_key)
                logger.info(f"Uploaded workdir to s3://{self.s3.bucket}/{workdir_key}")
        except Exception as e:
            logger.warning(f"S3 upload failed (non-critical): {e}")

    def on_completion(
        self,
        experiment_id: str,
        work_dir: Path,
        final_metrics: dict,
        evaluation_report: Any,
    ) -> None:
        """Upload final model to S3 if configured."""
        if not self.s3:
            logger.info(f"Local mode: model available at {work_dir / 'model'}")
            return

        try:
            model_tarball = work_dir / "model.tar.gz"
            if not model_tarball.exists():
                logger.warning(f"Model tarball not found: {model_tarball}")
                return

            logger.info(f"Uploading model package ({model_tarball.stat().st_size / 1024 / 1024:.1f} MB)...")
            model_key = self.s3.build_key(experiment_id, "model", "model.tar.gz")
            uri = self.s3.upload_file(model_tarball, model_key)
            logger.info(f"Final model uploaded: {uri}")
        except Exception as e:
            logger.error(f"Failed to upload model to S3: {e}")
            raise

    # ============================================
    # Private helpers
    # ============================================

    def _try_restore_workdir(self, experiment_id: str, work_dir: Path) -> None:
        """Attempt to restore work directory from S3."""
        if not self.s3:
            return

        workdir_key = self.s3.build_key(experiment_id, "workdir", "workdir.tar.gz")
        if not self.s3.object_exists(workdir_key):
            logger.debug("No previous workdir found in S3 — starting fresh")
            return

        logger.info("Found existing workdir in S3, restoring...")
        build_dir = work_dir / DirNames.BUILD_DIR
        temp_extract = work_dir / ".build_restore_tmp"

        try:
            if temp_extract.exists():
                shutil.rmtree(temp_extract)
            temp_extract.mkdir(parents=True, exist_ok=True)

            self.s3.download_and_extract_tar(workdir_key, temp_extract)

            restored_build = temp_extract / DirNames.BUILD_DIR
            if not restored_build.exists():
                extracted = [str(p.relative_to(temp_extract)) for p in temp_extract.rglob("*")]
                logger.error(f"Extracted tarball missing {DirNames.BUILD_DIR}. Contents: {extracted[:10] or '[empty]'}")
                raise ValueError(f"Extracted tarball missing {DirNames.BUILD_DIR} directory")

            if build_dir.exists():
                shutil.rmtree(build_dir)

            shutil.move(str(restored_build), str(build_dir))
            shutil.rmtree(temp_extract)
            logger.info("Restored workdir from S3")
        except Exception as e:
            logger.warning(f"Failed to restore workdir from S3: {e}")
            if temp_extract.exists():
                shutil.rmtree(temp_extract)

    def _download_model_from_s3_uri(self, s3_uri: str, work_dir: Path) -> str:
        """Download model from explicit S3 URI."""
        if not self.s3:
            raise ValueError(f"S3 model reference requires --external-storage-uri.\nModel: {s3_uri}")

        from botocore.exceptions import ClientError
        from plexe.integrations.storage.s3 import S3Helper

        logger.info(f"Downloading original model from S3: {s3_uri}")
        bucket, key = S3Helper.parse_uri(s3_uri)
        if not key:
            raise ValueError(f"Invalid S3 model reference, missing object key: {s3_uri}")

        original_model_dir = work_dir / "original_model"
        original_model_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = original_model_dir / Path(key).name

        # URI may point to a different bucket than self.s3 is configured for
        target_helper = S3Helper(bucket)
        try:
            target_helper.download_file(key, local_model_path)
        except ClientError as e:
            self.s3.handle_download_error(e, s3_uri)

        file_size_mb = local_model_path.stat().st_size / (1024**2)
        logger.info(f"Downloaded original model ({file_size_mb:.1f} MB): {local_model_path}")
        return str(local_model_path)

    def _download_model_by_experiment_id(self, experiment_id: str, work_dir: Path) -> str:
        """Download model by experiment ID from external storage."""
        from botocore.exceptions import ClientError

        logger.warning(f"Local model not found at '{experiment_id}'. Trying as experiment ID from external storage.")
        model_key = self.s3.build_key(experiment_id, "model", "model.tar.gz")

        if not self.s3.object_exists(model_key):
            raise FileNotFoundError(f"Model not found locally or in S3: {experiment_id}")

        original_model_dir = work_dir / "original_model"
        original_model_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = original_model_dir / "model.tar.gz"

        try:
            self.s3.download_file(model_key, local_model_path)
        except ClientError as e:
            self.s3.handle_download_error(e, experiment_id, " for experiment")

        file_size_mb = local_model_path.stat().st_size / (1024**2)
        logger.info(f"Downloaded original model ({file_size_mb:.1f} MB): {local_model_path}")
        return str(local_model_path)

    @staticmethod
    def _resolve_local_model(model_reference: str) -> str:
        """Resolve a local model path."""
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
