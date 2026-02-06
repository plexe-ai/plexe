"""
Plexe Platform adapter for model-builder-v2.

Handles full Plexe platform integration: S3 storage, DynamoDB tracking,
Secrets Manager, and optimized workflows for both PySpark and Databricks modes.
"""

import json
import logging
import math
import os
import tarfile
import tempfile
from decimal import Decimal
from pathlib import Path

import boto3

from plexe.adapters.base import WorkflowAdapter
from plexe.constants import DirNames
from plexe.execution.dataproc.dataset_io import DatasetNormalizer
from plexe.execution.dataproc.session import get_or_create_spark_session
from plexe_commons.repositories.experiment_store import ExperimentStore

logger = logging.getLogger(__name__)


def convert_floats_to_decimal(obj):
    """
    Recursively convert floats to Decimals for DynamoDB compatibility.

    Handles nested structures (dicts, lists, tuples) and special float values
    (NaN, Infinity) which are not supported by DynamoDB.

    Also converts non-string dictionary keys to strings, as DynamoDB requires
    all map keys to be strings (e.g., boolean keys True/False become "True"/"False").

    Args:
        obj: Any Python object (dict, list, tuple, float, etc.)

    Returns:
        Object with all floats converted to Decimal, special values converted to None,
        and all dict keys converted to strings

    Raises:
        ValueError: If a NaN or Infinity value is encountered (can be changed to return None)
    """
    if isinstance(obj, float):
        # Handle special float values that DynamoDB doesn't support
        if math.isnan(obj) or math.isinf(obj):
            # Convert to None rather than raising error to allow data to be stored
            # If these values appear, it indicates a potential issue in metric calculation
            logger.warning(f"Special float value encountered (NaN/Infinity), converting to None: {obj}")
            return None
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        # Convert all keys to strings (DynamoDB requires string keys)
        return {str(k): convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(v) for v in obj]
    elif isinstance(obj, tuple):
        # Convert tuples (used for confidence intervals in evaluation reports)
        return tuple(convert_floats_to_decimal(v) for v in obj)
    return obj


# TODO: why does this class not use DatasetRepository for dataset download like model-builder does?
#  In the case where we do download the dataset (local PySpark), shouldn't we use DatasetRepository?


class PlexePlatformAdapter(WorkflowAdapter):
    """
    Plexe Platform adapter for full cloud integration.

    Provides complete Plexe platform integration:
    - S3: Dataset access, checkpoint/workdir sync, model storage
    - DynamoDB: Experiment tracking via ExperimentStore
    - Secrets Manager: API key loading
    - Optimized for both PySpark and Databricks modes:
      * PySpark: Downloads S3 datasets to local disk for processing
      * Databricks: Uses S3 URIs directly for zero-copy data access

    Suitable for:
    - Production deployments on Plexe platform (ECS)
    - Local testing with full platform integration
    - Any scenario requiring experiment tracking and persistence
    """

    def __init__(self, experiment_id: str, experiment_store: ExperimentStore, config):
        """
        Initialize Plexe Platform adapter.

        Args:
            experiment_id: Experiment identifier
            experiment_store: ExperimentStore instance for DynamoDB
            config: Config object with workflow settings
        """
        self.experiment_id = experiment_id
        self.experiment_store = experiment_store
        self.config = config
        self.s3_client = boto3.client("s3")
        self.bucket = os.environ["MODELS_BUCKET_NAME"]
        self.intermediate_bucket = os.environ.get("INTERMEDIATE_BUCKET_NAME")  # Optional, validated when needed

    def setup_environment(self):
        """Load API keys from AWS Secrets Manager."""
        logger.info("Loading API keys from AWS Secrets Manager...")

        secrets_client = boto3.client("secretsmanager")

        if "OPENAI_API_KEY_SECRET_ARN" in os.environ:
            secret = secrets_client.get_secret_value(SecretId=os.environ["OPENAI_API_KEY_SECRET_ARN"])
            os.environ["OPENAI_API_KEY"] = secret["SecretString"]
            logger.info("✓ Loaded OPENAI_API_KEY from Secrets Manager")

        if "ANTHROPIC_API_KEY_SECRET_ARN" in os.environ:
            secret = secrets_client.get_secret_value(SecretId=os.environ["ANTHROPIC_API_KEY_SECRET_ARN"])
            os.environ["ANTHROPIC_API_KEY"] = secret["SecretString"]
            logger.info("✓ Loaded ANTHROPIC_API_KEY from Secrets Manager")

        # Load OTEL API key for Phoenix tracing
        if "OTEL_API_KEY_SECRET_ARN" in os.environ:
            try:
                secret = secrets_client.get_secret_value(SecretId=os.environ["OTEL_API_KEY_SECRET_ARN"])
                secret_data = json.loads(secret["SecretString"])
                api_key = secret_data["key"]
                self.config.otel_headers["authorization"] = f"Bearer {api_key}"
                logger.info("✓ Loaded OTEL API key from Secrets Manager")
            except Exception as e:
                logger.warning(f"Failed to load OTEL API key from Secrets Manager: {e}")

    def prepare_workspace(
        self,
        experiment_id: str,
        data_refs: list[str],
        work_dir: Path,
    ) -> tuple[str, str]:
        """
        Prepare workspace and normalize dataset format.

        Steps:
        1. Restore previous workdir from S3 if exists (for resume)
        2. Detect input dataset format (CSV, ORC, Avro, Parquet)
        3. Convert to Parquet if needed (writes to intermediate bucket)
        4. Return Parquet URI for workflow

        Note: Only the first dataset ref is used. Multi-dataset joining will be handled
        by a dedicated "dataset joiner agent" in the future.

        Args:
            experiment_id: Experiment identifier
            data_refs: List of S3 paths (only first one used)
            work_dir: Local working directory

        Returns:
            (dataset_uri, detected_format): Parquet URI for workflow + original format string
        """
        # Step 1: Try to restore previous workdir from S3 (for resume)
        workdir_s3_key = f"workdirs/{experiment_id}/workdir.tar.gz"

        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=workdir_s3_key)

            # Workdir exists - download and extract
            logger.info("Found existing workdir in S3, downloading for resume...")

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tarball_path = tmp.name

            self.s3_client.download_file(
                Bucket=self.bucket,
                Key=workdir_s3_key,
                Filename=tarball_path,
            )

            tarball_size_mb = Path(tarball_path).stat().st_size / (1024**2)
            logger.info(f"Downloaded workdir: {tarball_size_mb:.1f} MB")

            # Extract
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(path=work_dir)

            os.remove(tarball_path)
            logger.info("✓ Restored workdir from S3")

        except Exception:
            logger.debug("No previous workdir found in S3 - starting fresh")

        # Step 2: Get input dataset and validate it's in S3
        input_dataset_uri = data_refs[0]

        if not input_dataset_uri.startswith("s3://"):
            raise ValueError(
                f"PlexePlatformAdapter requires S3 datasets.\n"
                f"Got: {input_dataset_uri}\n"
                f"Upload your dataset to S3 first."
            )

        if len(data_refs) > 1:
            logger.warning(
                f"Multiple dataset refs provided, using first: {input_dataset_uri}. "
                "Multi-dataset joining will be supported in future."
            )

        # Step 3: Normalize dataset format
        if not self.intermediate_bucket:
            raise ValueError("INTERMEDIATE_BUCKET_NAME environment variable required")

        spark = get_or_create_spark_session(self.config)
        normalizer = DatasetNormalizer(spark)

        output_uri = f"s3://{self.intermediate_bucket}/normalized/{experiment_id}/dataset.parquet"

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
        Upload checkpoint and workdir to S3, update DynamoDB.

        Args:
            phase_name: Phase name
            checkpoint_path: Local checkpoint JSON file
            work_dir: Working directory
        """
        try:
            # Step 1: Upload checkpoint JSON to S3
            checkpoint_s3_key = f"checkpoints/{self.experiment_id}/{phase_name}.json"
            self.s3_client.upload_file(
                Filename=str(checkpoint_path),
                Bucket=self.bucket,
                Key=checkpoint_s3_key,
            )
            logger.info(f"✓ Uploaded checkpoint to S3: s3://{self.bucket}/{checkpoint_s3_key}")

            # Step 2: Tar + Upload Workdir to S3
            build_dir = work_dir / DirNames.BUILD_DIR
            if build_dir.exists():
                # Create tarball in temp location
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                    tarball_path = tmp.name

                logger.info(f"Creating workdir tarball from {build_dir}...")
                with tarfile.open(tarball_path, "w:gz") as tar:
                    tar.add(build_dir, arcname=DirNames.BUILD_DIR)

                tarball_size_mb = Path(tarball_path).stat().st_size / (1024**2)
                logger.info(f"Tarball size: {tarball_size_mb:.1f} MB")

                # Upload to S3
                workdir_s3_key = f"workdirs/{self.experiment_id}/workdir.tar.gz"
                self.s3_client.upload_file(
                    Filename=tarball_path,
                    Bucket=self.bucket,
                    Key=workdir_s3_key,
                )
                logger.info(f"✓ Uploaded workdir to S3: s3://{self.bucket}/{workdir_s3_key}")

                # Cleanup temp file
                os.remove(tarball_path)
            else:
                logger.warning(f"{DirNames.BUILD_DIR}/ directory not found, skipping workdir upload")

            # Step 3: Update DynamoDB
            checkpoints_dir = work_dir / "checkpoints"
            checkpoint_files = (
                sorted([f.stem for f in checkpoints_dir.glob("*.json")]) if checkpoints_dir.exists() else []
            )

            self.experiment_store.update_experiment_checkpoint(
                experiment_id=self.experiment_id,
                phase=phase_name,
                checkpoints_available=checkpoint_files,
            )
            logger.info("✓ Updated DynamoDB with checkpoint metadata")

        except Exception as e:
            logger.warning(f"Checkpoint sync to AWS failed (non-critical): {e}")

    def on_completion(
        self,
        experiment_id: str,
        work_dir: Path,
        final_metrics: dict,
        evaluation_report,  # EvaluationReport | None
    ):
        """
        Upload final model to S3 and mark experiment as completed in DynamoDB.

        Args:
            experiment_id: Experiment identifier
            work_dir: Working directory
            final_metrics: Final evaluation metrics
            evaluation_report: Full evaluation report (None if evaluation not run)
        """
        try:
            # Upload model package to S3
            model_package = work_dir / "model.tar.gz"
            if not model_package.exists():
                raise FileNotFoundError(f"Model package not found at {model_package}")

            logger.info(f"Uploading model package ({model_package.stat().st_size / 1024 / 1024:.1f} MB)...")
            s3_key = self.experiment_store.upload_model_to_s3(experiment_id, model_package)
            if not s3_key:
                raise RuntimeError("Failed to upload model to S3")

            model_path = f"s3://{self.bucket}/{s3_key}"
            logger.info(f"✓ Model uploaded: {model_path}")

            # Build result dict for DynamoDB
            result = {
                "model_tar_path": model_path,
                "model_type": "model-builder-v2",
                "deployment_method": "sagemaker-v2",
                "valid_metric": Decimal(str(final_metrics.get("performance", 0.0))),
                "metric_name": final_metrics.get("metric", "validation_performance"),
            }

            # Add evaluation if available
            if evaluation_report is not None:
                result["evaluation"] = evaluation_report.to_dict()
                logger.info("✓ Including evaluation report in DynamoDB update")

            # Read and include input/output schemas
            input_schema_path = work_dir / "model" / "schemas" / "input.json"
            output_schema_path = work_dir / "model" / "schemas" / "output.json"

            if input_schema_path.exists() and output_schema_path.exists():
                with open(input_schema_path) as f:
                    result["input_schema"] = json.load(f)

                with open(output_schema_path) as f:
                    result["output_schema"] = json.load(f)

                logger.info("✓ Including input/output schemas in DynamoDB update")
            else:
                logger.warning("Schema files not found - skipping input_schema and output_schema in DynamoDB update")

            # Extract and include code files from model package
            code_dict = {}
            model_dir = work_dir / "model"

            if model_dir.exists():
                logger.info("Extracting code files from model package...")

                # Read predictor.py at root (prefix with "root_" to avoid namespace collision)
                predictor_path = model_dir / "predictor.py"
                if predictor_path.exists():
                    try:
                        code_dict["root_predictor"] = predictor_path.read_text(encoding="utf-8")
                        logger.debug("✓ Extracted predictor.py")
                    except Exception as e:
                        logger.warning(f"Failed to read predictor.py: {e}")

                # Read all .py files from src/ directory (prefix with "src_" for clarity)
                src_dir = model_dir / "src"
                if src_dir.exists():
                    for py_file in src_dir.glob("*.py"):
                        if py_file.name != "__init__.py":  # Skip __init__.py
                            key = f"src_{py_file.stem}"  # e.g., "src_trainer", "src_pipeline"
                            try:
                                code_dict[key] = py_file.read_text(encoding="utf-8")
                                logger.debug(f"✓ Extracted src/{py_file.name}")
                            except Exception as e:
                                logger.warning(f"Failed to read src/{py_file.name}: {e}")

                if code_dict:
                    result["code"] = code_dict
                    logger.info(
                        f"✓ Including {len(code_dict)} code file(s) in DynamoDB update: {list(code_dict.keys())}"
                    )
                else:
                    logger.warning("No code files found in model package")
            else:
                logger.warning("Model directory not found - skipping code extraction")

            # Convert all floats to Decimals for DynamoDB compatibility
            result = convert_floats_to_decimal(result)

            # Update experiment status in DynamoDB
            self.experiment_store.update_experiment_status(
                experiment_id=experiment_id,
                status="completed",
                workspace_dir=work_dir,
                result=result,
            )
            logger.info(f"✓ Experiment {experiment_id} marked as completed in DynamoDB")

        except Exception as e:
            logger.error(f"Failed to complete experiment finalization: {e}")
            raise

    def on_failure(
        self,
        experiment_id: str,
        error: Exception,
    ):
        """
        Mark experiment as failed in DynamoDB.

        Args:
            experiment_id: Experiment identifier
            error: Exception that caused failure
        """
        try:
            self.experiment_store.update_experiment_status(
                experiment_id=experiment_id,
                status="failed",
                error=str(error),
            )
            logger.info(f"✓ Experiment {experiment_id} marked as failed in DynamoDB")

        except Exception as e:
            logger.error(f"Failed to update experiment failure status: {e}")

    def prepare_original_model(
        self,
        model_reference: str,
        work_dir: Path,
    ) -> str:
        """
        Download original model from S3 for retraining.

        In AWS mode, model_reference is interpreted as a platform experiment ID.
        The model is located in S3 at: s3://{bucket}/models/{experiment_id}/model.tar.gz

        Args:
            model_reference: Platform experiment ID (e.g., "exp_abc123")
            work_dir: Working directory for download

        Returns:
            Local filesystem path to downloaded model.tar.gz

        Raises:
            FileNotFoundError: If model for experiment_id doesn't exist in S3
            RuntimeError: If download fails
        """
        experiment_id = model_reference
        logger.info(f"Preparing original model for retraining from experiment: {experiment_id}")

        try:
            # Construct S3 prefix for experiment's models
            s3_prefix = f"models/{experiment_id}/"

            logger.info(f"Searching for model at s3://{self.bucket}/{s3_prefix}")

            # List all files in the models directory for this experiment
            response = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=s3_prefix)

            if "Contents" not in response:
                raise FileNotFoundError(
                    f"No model files found for experiment {experiment_id} in S3. "
                    f"Searched: s3://{self.bucket}/{s3_prefix}"
                )

            # Find the first .tar.gz file
            tar_gz_key = None
            for obj in response["Contents"]:
                if obj["Key"].endswith(".tar.gz"):
                    tar_gz_key = obj["Key"]
                    break

            if not tar_gz_key:
                available_files = [obj["Key"] for obj in response["Contents"]]
                raise FileNotFoundError(
                    f"No .tar.gz model file found for experiment {experiment_id}. " f"Files found: {available_files}"
                )

            # Create directory for original model
            original_model_dir = work_dir / "original_model"
            original_model_dir.mkdir(parents=True, exist_ok=True)

            # Download to local path (preserve original filename)
            filename = Path(tar_gz_key).name
            local_model_path = original_model_dir / filename

            logger.info(f"Downloading model from s3://{self.bucket}/{tar_gz_key}...")
            self.s3_client.download_file(
                Bucket=self.bucket,
                Key=tar_gz_key,
                Filename=str(local_model_path),
            )

            file_size_mb = local_model_path.stat().st_size / (1024**2)
            logger.info(f"✓ Downloaded original model ({file_size_mb:.1f} MB): {local_model_path}")

            return str(local_model_path)

        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise

        except Exception as e:
            logger.error(f"Failed to download original model for {experiment_id}: {e}")
            raise RuntimeError(f"Model download failed: {e}") from e

    def on_pause(
        self,
        phase_name: str,
    ):
        """
        Update experiment status to paused in DynamoDB.

        Args:
            phase_name: Name of the phase where workflow paused
        """
        try:
            self.experiment_store.update_experiment_status(
                experiment_id=self.experiment_id,
                status="paused_awaiting_feedback",
            )
            logger.info(f"✓ Experiment {self.experiment_id} marked as paused at {phase_name} in DynamoDB")

        except Exception as e:
            logger.warning(f"Failed to update pause status in DynamoDB: {e}")

    def get_splits_output_location(self, dataset_uri: str, experiment_id: str, work_dir: Path) -> str:
        return self._get_storage_location(dataset_uri, experiment_id, work_dir, "splits")

    def get_samples_output_location(self, dataset_uri: str, experiment_id: str, work_dir: Path) -> str:
        return self._get_storage_location(dataset_uri, experiment_id, work_dir, "samples")

    def get_transformed_output_location(self, dataset_uri: str, experiment_id: str, work_dir: Path) -> str:
        return self._get_storage_location(dataset_uri, experiment_id, work_dir, "transformed")

    def ensure_samples_local(self, sample_uris: list[str], work_dir: Path) -> list[str]:
        """Download S3 samples to local if needed (handles Spark parquet directories)."""
        local_uris = []

        for uri in sample_uris:
            if uri.startswith("s3://"):
                parts = uri[5:].split("/", 1)
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(f"Invalid S3 sample URI (expected s3://bucket/key): {uri}")
                bucket, prefix = parts[0], parts[1]

                # Spark writes parquet as directory - list all objects under prefix
                logger.info(f"Downloading sample directory from S3: {uri}")
                response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

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
                    self.s3_client.download_file(bucket, obj_key, str(local_file))

                logger.info(f"Downloaded {len(response['Contents'])} files to {local_dir}")
                local_uris.append(str(local_dir))
            else:
                local_uris.append(uri)
        return local_uris

    # ============================================
    # Helper Methods
    # ============================================

    def _get_storage_location(self, dataset_uri: str, experiment_id: str, work_dir: Path, artifact_type: str) -> str:
        """Determine storage location based on dataset location."""
        if dataset_uri.startswith("s3://"):
            if not self.intermediate_bucket:
                raise ValueError(f"INTERMEDIATE_BUCKET_NAME required for S3 {artifact_type} storage")
            return f"s3://{self.intermediate_bucket}/{artifact_type}/{experiment_id}"
        return str(work_dir / DirNames.BUILD_DIR / "data" / artifact_type)

    @staticmethod
    def _parse_s3_dataset_path(s3_path: str) -> tuple[str, str, str] | None:
        """
        Parse S3 dataset path.

        Format: s3://bucket/users/{user_id}/datasets/{dataset_id}/versions/{version}/

        Args:
            s3_path: S3 URI

        Returns:
            (user_id, dataset_id, version) or None if invalid format
        """
        if not s3_path.startswith("s3://"):
            return None

        parts = s3_path[5:].rstrip("/").split("/")
        if len(parts) >= 6 and parts[0] and parts[1] == "users" and parts[3] == "datasets" and parts[5] == "versions":
            return parts[2], parts[4], parts[6]  # user_id, dataset_id, version

        return None
