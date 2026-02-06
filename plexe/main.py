"""
Universal entry point for model-builder-v2.

Can be called:
- As a Python function: from plexe.main import main; main(...)
- Via CLI: python -m source.main --train-dataset-uri data.parquet --intent "..."
- From ECS: start.sh reads EXPERIMENT_DATA and calls with CLI args
"""

import os


# CRITICAL: Set Keras backend BEFORE any imports (retrain.py imports keras at module level)
os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
import json
import logging
import sys
from pathlib import Path

from plexe.config import setup_logging, setup_litellm, get_config
from plexe.constants import DirNames, PhaseNames
from plexe.execution.dataproc.session import get_or_create_spark_session, stop_spark_session
from plexe.execution.training.local_runner import LocalProcessRunner
from plexe.search.tree_policy import TreeSearchPolicy
from plexe.utils.tracing import setup_opentelemetry
from plexe.workflow import build_model
from plexe.retrain import retrain_model

logger = logging.getLogger(__name__)


def main(
    intent: str,
    data_refs: list[str],  # TODO: Support multiple datasets + join_strategy when multi-dataset joining is implemented
    adapter_type: str = "standalone",
    spark_mode: str = "local",
    user_id: str = "default_user",
    experiment_id: str = "local",
    max_iterations: int = 10,
    work_dir: Path = Path("/tmp/model_builder_v2"),
    test_dataset_uri: str | None = None,
    enable_final_evaluation: bool = False,
    max_epochs: int | None = None,
    allowed_model_types: list[str] | None = None,
    is_retrain: bool = False,
    original_model_uri: str | None = None,
    original_experiment_id: str | None = None,
    auto_mode: bool = True,
    user_feedback: dict | None = None,
    enable_otel: bool = False,
    otel_endpoint: str | None = None,
    otel_headers: dict[str, str] | None = None,
    external_storage_uri: str | None = None,
    csv_delimiter: str = ",",
    csv_header: bool = True,
):
    """
    Main model building function.

    Args:
        intent: ML task description
        data_refs: Dataset references for adapter.prepare_workspace()
        adapter_type: Adapter type ("standalone")
        spark_mode: Spark backend ("local" or "databricks")
        user_id: User identifier
        experiment_id: Experiment identifier
        max_iterations: Maximum search iterations
        work_dir: Working directory for artifacts
        test_dataset_uri: Optional test dataset URI
        enable_final_evaluation: Whether to run test evaluation
        max_epochs: Cap Keras epochs (for testing)
        allowed_model_types: Restrict model types
        is_retrain: Whether this is a retraining job
        original_model_uri: URI to original model.tar.gz (for retraining)
        original_experiment_id: Experiment ID of original model (for retraining)
        auto_mode: If True, runs to completion without pausing for user feedback
        user_feedback: User feedback dict when resuming from pause (optional)
        enable_otel: Enable OpenTelemetry tracing
        otel_endpoint: OTLP endpoint URL (e.g., https://cloud.langfuse.com/api/public/otel)
        otel_headers: Authentication headers for OTLP endpoint
        external_storage_uri: S3 URI for external storage (datasets, checkpoints, workdir, models)
        csv_delimiter: CSV delimiter character (default: comma)
        csv_header: Whether CSV has header row (default: True)

    Returns:
        (best_solution, final_metrics, evaluation_report) tuple

    Raises:
        KeyboardInterrupt: If user interrupts
        Exception: On workflow failure
    """

    # Adapter will be initialized inside try-except to ensure on_failure() can always be called
    adapter = None

    try:

        # Load config from YAML file (if CONFIG_FILE env var set) + apply env var overrides
        config = get_config()

        # Apply CLI argument overrides (highest priority)
        config.max_search_iterations = max_iterations
        config.spark_mode = spark_mode
        if max_epochs:
            config.keras_default_epochs = min(max_epochs, config.keras_default_epochs)
        if allowed_model_types:
            config.allowed_model_types = allowed_model_types

        # Apply CSV format options from CLI args
        config.csv_delimiter = csv_delimiter
        config.csv_header = csv_header

        # Apply OTLP tracing config from CLI args
        config.enable_otel = enable_otel
        if otel_endpoint:
            config.otel_endpoint = otel_endpoint
        if otel_headers:
            config.otel_headers.update(otel_headers)

        # Setup basic logging early (before workspace preparation)
        setup_logging(config)

        # Configure LiteLLM global settings
        setup_litellm(config)

        # Log config source
        config_file = os.getenv("CONFIG_FILE")
        if config_file:
            logger.info(f"Configuration loaded from: {config_file}")
        else:
            logger.info("Configuration: using defaults + environment variables")

        # Create adapter
        if adapter_type == "standalone":
            from plexe.adapters.standalone import StandaloneAdapter

            adapter = StandaloneAdapter(config, external_storage_uri=external_storage_uri, user_id=user_id)
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}. Use 'standalone'.")

        # Setup environment (adapter can populate config.otel_headers with secrets)
        adapter.setup_environment()

        # Setup OTEL tracing (requires secrets from setup_environment)
        setup_opentelemetry(config)

        # Prepare workspace
        work_dir.mkdir(parents=True, exist_ok=True)
        train_dataset_uri, input_format = adapter.prepare_workspace(experiment_id, data_refs, work_dir)

        # Prepare original model if retraining
        if is_retrain:
            # Pick which reference to use (prioritize explicit URI over experiment ID)
            model_reference = original_model_uri or original_experiment_id

            if not model_reference:
                raise ValueError(
                    "Retraining requires either --original-model-uri (local path) or "
                    "--original-experiment-id (platform experiment ID)"
                )

            # Let adapter interpret the string reference and ensure model is local
            original_model_uri = adapter.prepare_original_model(model_reference, work_dir)

        logger.info(f"Experiment: {experiment_id} | User: {user_id}")
        logger.info(f"Adapter: {adapter_type} | Spark: {spark_mode}")
        logger.info(f"LiteLLM routing: {'custom config' if config.routing_config else 'default providers'}")
        logger.info(f"Intent: {intent}")
        logger.info(f"Dataset: {train_dataset_uri} (format: {input_format}) | Max iterations: {max_iterations}")

        runner = LocalProcessRunner(work_dir=str(work_dir / DirNames.BUILD_DIR / "search" / "runs"))

        # Branch: Retraining vs Normal Build
        if is_retrain:
            logger.info("RETRAINING MODE")
            logger.info(f"Original model (local): {original_model_uri}")

            # original_model_uri is now guaranteed to be a local path (prepared by adapter)
            best_solution, final_metrics = retrain_model(
                original_model_uri=original_model_uri,
                train_dataset_uri=train_dataset_uri,
                experiment_id=experiment_id,
                work_dir=work_dir,
                runner=runner,
                config=config,
                on_checkpoint_saved=adapter.on_checkpoint,
            )
            # Retraining doesn't generate evaluation reports
            evaluation_report = None
        else:
            # Normal build workflow
            spark = get_or_create_spark_session(config)
            search_policy = TreeSearchPolicy()
            # search_policy = EvolutionarySearchPolicy()  TODO: enable after testing

            # Convert auto_mode to pause_points
            # If auto_mode=False, pause after Phase 1 for review; if True, no pauses
            pause_points = None if auto_mode else [PhaseNames.ANALYZE_DATA]

            result = build_model(
                spark=spark,
                train_dataset_uri=train_dataset_uri,
                test_dataset_uri=test_dataset_uri,
                user_id=user_id,
                intent=intent,
                experiment_id=experiment_id,
                work_dir=work_dir,
                runner=runner,
                search_policy=search_policy,
                config=config,
                adapter=adapter,
                enable_final_evaluation=enable_final_evaluation,
                on_checkpoint_saved=adapter.on_checkpoint,
                pause_points=pause_points,
                on_pause=adapter.on_pause,
                user_feedback=user_feedback,
            )

            # Handle pause case
            if result is None:
                logger.info("Workflow paused - awaiting user feedback")
                return None, {}, None

            best_solution, final_metrics, evaluation_report = result

        logger.info(f"MODEL COMPLETE | Performance: {best_solution.performance:.4f}")

        # Finalize
        adapter.on_completion(experiment_id, work_dir, final_metrics, evaluation_report)

        return best_solution, final_metrics, evaluation_report

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        if adapter:
            adapter.on_failure(experiment_id, KeyboardInterrupt("User interrupt"))
        raise

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        if adapter:
            adapter.on_failure(experiment_id, e)
        raise

    finally:
        stop_spark_session()


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Model Builder Slim - Automated ML model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-dataset-uri", required=True, help="Path to training dataset (CSV, ORC, Avro, or Parquet)"
    )
    parser.add_argument("--test-dataset-uri", help="Optional: Path to test dataset (CSV, ORC, Avro, or Parquet)")
    parser.add_argument("--user-id", default=os.getenv("USER_ID", "default_user"), help="User identifier")
    parser.add_argument("--intent", required=True, help="ML task description")
    parser.add_argument("--experiment-id", default=os.getenv("EXPERIMENT_ID", "local"), help="Experiment identifier")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max search iterations")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/model_builder_v2"), help="Working directory")
    parser.add_argument("--enable-final-evaluation", action="store_true", help="Enable test set evaluation")
    parser.add_argument("--max-epochs", type=int, help="Cap Keras epochs (testing)")
    parser.add_argument(
        "--allowed-model-types", nargs="+", choices=["xgboost", "catboost", "keras"], help="Restrict models"
    )

    # Retraining support
    parser.add_argument("--is-retrain", action="store_true", help="Enable retraining mode")
    parser.add_argument("--original-model-uri", help="Path to original model.tar.gz (for retraining)")
    parser.add_argument("--original-experiment-id", help="Experiment ID of original model (for retraining)")

    # Workflow control
    parser.add_argument(
        "--auto-mode",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Run to completion without pausing (default: True). Pass 'true' or 'false'",
    )
    parser.add_argument(
        "--user-feedback",
        type=str,
        help="User feedback JSON (for resume_with_feedback)",
    )

    # Infrastructure configuration
    parser.add_argument(
        "--adapter-type",
        choices=["standalone", "plexe", "local", "aws"],  # Include deprecated names for compatibility
        default=os.getenv("ADAPTER_TYPE", "standalone"),
        help="Platform integration: standalone=no cloud integration, plexe=full Plexe platform (S3+DynamoDB). "
        "Deprecated: 'local' (use 'standalone'), 'aws' (use 'plexe')",
    )
    parser.add_argument(
        "--spark-mode",
        choices=["local", "databricks"],
        help="Spark backend (default: auto-selected based on adapter-type)",
    )
    parser.add_argument(
        "--external-storage-uri",
        type=str,
        help="S3 URI for external storage (e.g., s3://bucket/prefix). "
        "Used for intermediate datasets, checkpoints, workdir backup, and final model. "
        "Required for standalone mode with S3 datasets.",
    )

    # OpenTelemetry tracing configuration
    parser.add_argument(
        "--enable-otel",
        action="store_true",
        help="Enable OpenTelemetry tracing (default: disabled)",
    )
    parser.add_argument(
        "--otel-endpoint",
        type=str,
        help="OTLP endpoint URL (e.g., https://cloud.langfuse.com/api/public/otel)",
    )
    parser.add_argument(
        "--otel-header",
        action="append",
        dest="otel_headers",
        metavar="KEY=VALUE",
        help="OTLP auth header (repeatable, e.g., --otel-header 'authorization=Bearer token')",
    )

    # General configuration file
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to YAML configuration file for overriding Config defaults",
    )

    # Dataset format options
    parser.add_argument(
        "--csv-delimiter",
        type=str,
        default=",",
        help="CSV delimiter character (default: comma). Use 'tab' for TSV, '|' for pipe-delimited, etc.",
    )
    parser.add_argument(
        "--csv-header",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether CSV has header row (default: True). Pass 'true' or 'false'",
    )

    args = parser.parse_args()

    # Determine adapter type from CLI or environment
    adapter_type = args.adapter_type

    # Auto-select spark_mode if not specified
    if args.spark_mode:
        spark_mode = args.spark_mode
    else:
        spark_mode = "local"

    # Auto-enable evaluation if test dataset provided
    enable_final_evaluation = args.enable_final_evaluation or (args.test_dataset_uri is not None)

    # Parse user feedback JSON if provided
    user_feedback = None
    if args.user_feedback:
        try:
            user_feedback = json.loads(args.user_feedback)
        except json.JSONDecodeError as e:
            print(f"Invalid user feedback JSON: {e}", file=sys.stderr)
            sys.exit(1)

    # Parse OTLP headers from CLI (list of "key=value" strings)
    otel_headers_dict = {}
    if args.otel_headers:
        for header in args.otel_headers:
            if "=" in header:
                key, value = header.split("=", 1)
                otel_headers_dict[key.strip()] = value.strip()
            else:
                print(f"Invalid otel header format (expected KEY=VALUE): {header}", file=sys.stderr)
                sys.exit(1)

    # Handle special delimiter values
    csv_delimiter = args.csv_delimiter
    if csv_delimiter.lower() == "tab":
        csv_delimiter = "\t"

    # Set CONFIG_FILE environment variable if provided (get_config() will read it)
    if args.config_file:
        os.environ["CONFIG_FILE"] = args.config_file

    # Call main function and handle exit codes at process boundary
    try:
        main(
            intent=args.intent,
            data_refs=[args.train_dataset_uri],
            adapter_type=adapter_type,
            spark_mode=spark_mode,
            user_id=args.user_id,
            experiment_id=args.experiment_id,
            max_iterations=args.max_iterations,
            work_dir=args.work_dir,
            test_dataset_uri=args.test_dataset_uri,
            enable_final_evaluation=enable_final_evaluation,
            max_epochs=args.max_epochs,
            allowed_model_types=args.allowed_model_types,
            is_retrain=args.is_retrain,
            original_model_uri=args.original_model_uri,
            original_experiment_id=args.original_experiment_id,
            auto_mode=args.auto_mode,
            user_feedback=user_feedback,
            enable_otel=args.enable_otel,
            otel_endpoint=args.otel_endpoint,
            otel_headers=otel_headers_dict,
            external_storage_uri=args.external_storage_uri,
            csv_delimiter=csv_delimiter,
            csv_header=args.csv_header,
        )
        sys.exit(0)

    except KeyboardInterrupt:
        sys.exit(130)

    except Exception:
        sys.exit(1)
