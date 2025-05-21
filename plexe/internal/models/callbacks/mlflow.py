"""
MLFlow callback for tracking model building process.

This module provides a callback implementation that logs model building
metrics, parameters, and artifacts to MLFlow.
"""

# Standard library imports at the top level
import os
import re
import tempfile
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Third-party imports
import mlflow
import logging
import warnings

# Plexe imports
from plexe.callbacks import Callback, BuildStateInfo
from plexe.internal.models.entities.metric import Metric

# Setup logger and constants
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Constants
ARTIFACT_DIR = "artifacts"
LOG_DIR = "logs"
EDA_DIR = "eda_reports"


class MLFlowCallback(Callback):
    """
    Callback that logs the model building process to MLFlow with hierarchical run organization.

    Implements nested runs with parent/child relationship:
    - Parent run: Overall model building process, common parameters
    - Child runs: Individual iterations with iteration-specific metrics
    """

    def __init__(self, tracking_uri: str, experiment_name: str, connect_timeout: int = 10):
        """
        Initialize MLFlow callback.

        Args:
            tracking_uri: MLFlow tracking server URI.
            experiment_name: Name for the MLFlow experiment.
            connect_timeout: Timeout in seconds for MLFlow server connection.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.experiment_id = None
        self.connect_timeout = connect_timeout
        self.parent_run_id = None
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Configure MLFlow tracking and clean up any active runs."""
        try:
            # End any active runs from previous sessions
            self._end_active_run()

            # Configure MLFlow environment
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(self.connect_timeout)
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.debug(f"✅ MLFlow configured with tracking URI '{self.tracking_uri}'")

            # Set up MLFlow tracing if available
            try:
                mlflow.smolagents.autolog()
                logger.debug("✅ MLFlow smolagents autolog enabled")
            except ModuleNotFoundError:
                logger.debug("⚠️ MLFlow smolagents autolog not available")

        except Exception as e:
            logger.error(f"❌ Error setting up MLFlow: {e}")
            raise RuntimeError(f"Failed to setup MLFlow: {e}") from e

    def _end_active_run(self) -> None:
        """Safely end any active MLFlow run."""
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception as e:
            logger.warning(f"⚠️ Error ending active MLFlow run: {e}")

    def _get_or_create_experiment(self) -> str:
        """Get or create the MLFlow experiment and return its ID."""
        try:
            # Try to get existing experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                return experiment.experiment_id

            # Create if not exists
            experiment_id = mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(experiment_name=self.experiment_name)
            logger.debug(f"✅ MLFlow experiment created: '{self.experiment_name}' (ID: {experiment_id})")
            print(f"✅ MLFlow: tracking URI '{self.tracking_uri}', experiment '{self.experiment_name}'")
            return experiment_id
        except Exception as e:
            logger.error(f"❌ Error creating MLFlow experiment: {e}")
            raise RuntimeError(f"Failed to create MLFlow experiment: {e}") from e

    def _ensure_parent_run_active(self) -> bool:
        """
        Ensure the parent run is active, activating it if needed.

        Returns:
            True if parent run is active, False otherwise.
        """
        if not self.parent_run_id:
            logger.warning("⚠️ No parent run ID available")
            return False

        try:
            active_run = mlflow.active_run()
            # If already active and it's the parent run, we're good
            if active_run and active_run.info.run_id == self.parent_run_id:
                return True

            # If another run is active, end it first
            if active_run:
                mlflow.end_run()

            # Start the parent run
            mlflow.start_run(run_id=self.parent_run_id)
            return True
        except Exception as e:
            logger.warning(f"⚠️ Could not activate parent run: {e}")
            return False

    def _safe_log_artifact(self, content: str, filename: str, directory: str = None) -> None:
        """
        Safely log an artifact by writing to a temporary file first.

        Args:
            content: Content to write to the file
            filename: Name of the file in MLFlow
            directory: Optional subdirectory for organization
        """
        if not mlflow.active_run():
            logger.warning(f"⚠️ Cannot log artifact '{filename}': No active run")
            return

        # Create unique temp path
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=Path(filename).suffix) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Write content to temp file
            with open(tmp_path, "w") as f:
                f.write(content)

            # Log file to MLFlow in specified directory
            if directory:
                # Create artifact subdirectory structure
                artifact_path = directory
                mlflow.log_artifact(str(tmp_path), artifact_path)
            else:
                mlflow.log_artifact(str(tmp_path))

            logger.debug(f"✅ Logged artifact: {filename}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to log artifact '{filename}': {e}")
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

    def _extract_model_context(self, info: BuildStateInfo) -> Dict[str, Any]:
        """
        Extract useful context about the model for logging.

        Args:
            info: Build state information

        Returns:
            Dictionary of model context information
        """
        context = {"intent": info.intent, "provider": str(info.provider)}

        # Add timing information if available
        if info.timeout:
            context["total_timeout_seconds"] = info.timeout
        if info.run_timeout:
            context["run_timeout_seconds"] = info.run_timeout
        if info.max_iterations:
            context["max_iterations"] = info.max_iterations

        # Add model info if available
        if info.model:
            model = info.model
            if hasattr(model, "identifier"):
                context["model_id"] = model.identifier
            if hasattr(model, "run_id"):
                context["run_id"] = model.run_id
            if hasattr(model, "state"):
                context["model_state"] = str(model.state)

        # Add schema information
        if info.input_schema:
            field_names = list(info.input_schema.model_fields.keys())
            context["input_schema_fields"] = ", ".join(field_names)
            context["input_schema_field_count"] = len(field_names)

        if info.output_schema:
            field_names = list(info.output_schema.model_fields.keys())
            context["output_schema_fields"] = ", ".join(field_names)
            context["output_schema_field_count"] = len(field_names)

        # Add dataset information if available
        if info.datasets:
            dataset_summary = {}
            for name, dataset in info.datasets.items():
                if hasattr(dataset, "shape"):
                    shape = dataset.shape
                    dataset_summary[name] = f"{shape[0]} rows, {shape[1]} columns"
            context["datasets"] = dataset_summary
            context["dataset_count"] = len(info.datasets)

        return context

    def on_build_start(self, info: BuildStateInfo) -> None:
        """
        Start MLFlow parent run and log initial parameters.

        Args:
            info: Information about the model building process start.
        """
        try:
            # Get or create experiment
            self.experiment_id = self._get_or_create_experiment()

            # Extract model information for descriptive run name
            model_id = info.model.identifier if hasattr(info.model, "identifier") else "unknown"
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

            # Start parent run with informative name
            parent_run = mlflow.start_run(
                run_name=f"model-{model_id}-{timestamp}",
                experiment_id=self.experiment_id,
                description=f"Model building for: {info.intent[:100]}...",
            )
            self.parent_run_id = parent_run.info.run_id
            logger.debug(f"✅ Started MLFlow parent run: {self.parent_run_id}")

            # Extract and log common parameters
            model_context = self._extract_model_context(info)

            # Log parameters and tags
            mlflow.log_params(model_context)
            mlflow.set_tags({"provider": str(info.provider), "run_type": "parent", "build_timestamp": timestamp})

            # Log intent as a note for better visibility in UI
            if info.intent:
                self._safe_log_artifact(content=info.intent, filename="intent.txt")

        except Exception as e:
            logger.error(f"❌ Error in on_build_start: {e}")
            # Don't re-raise to allow build process to continue

    def on_iteration_start(self, info: BuildStateInfo) -> None:
        """
        Start a new nested child run for this iteration.

        Args:
            info: Information about the iteration start.
        """
        if not self.parent_run_id:
            logger.warning("⚠️ Cannot start iteration: No parent run exists")
            return

        try:
            # Create meaningful run name including iteration number
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"iteration-{info.iteration}-{timestamp}"

            # Start nested run under the parent
            mlflow.start_run(
                run_name=run_name,
                experiment_id=self.experiment_id,
                nested=True,
                description=f"Iteration {info.iteration} of model training",
            )
            logger.debug(f"✅ Started MLFlow nested run for iteration {info.iteration}")

            # Log iteration-specific parameters
            iteration_params = {"iteration": info.iteration, "timestamp": timestamp}

            mlflow.log_params(iteration_params)
            mlflow.set_tags({"run_type": "iteration", "iteration": str(info.iteration)})

            # Log training datasets only if available
            if info.datasets:
                for name, data in info.datasets.items():
                    try:
                        mlflow.log_input(mlflow.data.from_pandas(data.to_pandas(), name=name), context="training")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not log dataset '{name}': {e}")

        except Exception as e:
            logger.error(f"❌ Error in on_iteration_start: {e}")

    def on_iteration_end(self, info: BuildStateInfo) -> None:
        """
        Log metrics for this iteration and end the child run.

        Args:
            info: Information about the iteration end.
        """
        if not mlflow.active_run():
            logger.warning("⚠️ Cannot end iteration: No active run")
            return

        try:
            # Record validation datasets
            if info.datasets:
                for name, data in info.datasets.items():
                    try:
                        mlflow.log_input(mlflow.data.from_pandas(data.to_pandas(), name=name), context="validation")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not log validation dataset '{name}': {e}")

            # Only process node data if node exists
            if info.node:
                # Log training code if available
                if hasattr(info.node, "training_code") and info.node.training_code:
                    self._safe_log_artifact(
                        content=info.node.training_code, filename="trainer_source.py", directory=f"{ARTIFACT_DIR}/code"
                    )

                # Log performance metrics
                if hasattr(info.node, "performance") and info.node.performance:
                    self._log_metric(info.node.performance)

                # Log execution time
                if hasattr(info.node, "execution_time") and info.node.execution_time:
                    mlflow.log_metric("execution_time", info.node.execution_time)

                # Log exception information
                if hasattr(info.node, "exception_was_raised") and info.node.exception_was_raised:
                    exception_type = "unknown"
                    if hasattr(info.node, "exception") and info.node.exception:
                        exception_type = type(info.node.exception).__name__

                    mlflow.set_tags({"exception_raised": "true", "exception_type": exception_type})

                    # Log exception details if available
                    if hasattr(info.node, "exception") and info.node.exception:
                        self._safe_log_artifact(
                            content=str(info.node.exception),
                            filename=f"exception-iteration-{info.iteration}.txt",
                            directory=f"{LOG_DIR}/exceptions",
                        )

                # Log model artifacts
                if hasattr(info.node, "model_artifacts") and info.node.model_artifacts:
                    for artifact in info.node.model_artifacts:
                        if Path(artifact).exists():
                            try:
                                mlflow.log_artifact(str(artifact), artifact_path=f"{ARTIFACT_DIR}/models")
                            except Exception as e:
                                logger.warning(f"⚠️ Could not log artifact {artifact}: {e}")

            # Determine run status based on iteration outcome
            status = "FINISHED"
            if info.node and hasattr(info.node, "exception_was_raised") and info.node.exception_was_raised:
                status = "FAILED"
            elif info.node and hasattr(info.node, "performance") and info.node.performance is None:
                status = "FAILED"
            elif (
                info.node
                and hasattr(info.node, "performance")
                and hasattr(info.node.performance, "is_worst")
                and info.node.performance.is_worst
            ):
                status = "FAILED"

            # End the child run
            mlflow.end_run(status=status)
            logger.debug(f"✅ Ended MLFlow run for iteration {info.iteration} with status: {status}")

        except Exception as e:
            logger.error(f"❌ Error in on_iteration_end: {e}")
            # Try to end run even if there was an error
            try:
                mlflow.end_run(status="FAILED")
            except Exception:
                pass

    def on_build_end(self, info: BuildStateInfo) -> None:
        """
        Log final model details and end MLFlow parent run.

        Args:
            info: Information about the model building process end.
        """
        try:
            # End any active child run first
            active_run = mlflow.active_run()
            if active_run and active_run.info.run_id != self.parent_run_id:
                mlflow.end_run()

            # Activate parent run for final logging
            if not self._ensure_parent_run_active():
                logger.warning("⚠️ Cannot complete build_end: Parent run unavailable")
                return

            # Log EDA reports if available
            if info.node and hasattr(info.node, "metadata"):
                node_metadata = getattr(info.node, "metadata", {})
                if node_metadata and "eda_markdown_reports" in node_metadata:
                    for dataset_name, report_markdown in node_metadata["eda_markdown_reports"].items():
                        self._safe_log_artifact(
                            content=report_markdown, filename=f"eda_report_{dataset_name}.md", directory=f"{EDA_DIR}"
                        )

            # Log best metrics from model if available
            if info.model:
                model = info.model

                # Log best model metric
                if hasattr(model, "metric") and model.metric:
                    metric = model.metric
                    if hasattr(metric, "name") and hasattr(metric, "value"):
                        mlflow.log_metric(f"best_{metric.name}", float(metric.value))

                # Log best iteration
                mlflow.set_tag("best_iteration", str(info.iteration))

                # Log model artifacts
                if hasattr(model, "artifacts") and model.artifacts:
                    artifact_names = [a.name for a in model.artifacts]
                    mlflow.set_tag("model_artifacts", ", ".join(artifact_names))

                # Log model state
                if hasattr(model, "state"):
                    mlflow.set_tag("final_model_state", str(model.state))

                # Log final model code
                if hasattr(model, "trainer_source") and model.trainer_source:
                    self._safe_log_artifact(
                        content=model.trainer_source, filename="final_trainer.py", directory=f"{ARTIFACT_DIR}/final"
                    )

                if hasattr(model, "predictor_source") and model.predictor_source:
                    self._safe_log_artifact(
                        content=model.predictor_source, filename="final_predictor.py", directory=f"{ARTIFACT_DIR}/final"
                    )

            # End the parent run
            mlflow.end_run()
            logger.debug("✅ Ended MLFlow parent run")

        except Exception as e:
            logger.error(f"❌ Error in on_build_end: {e}")
            # Try to end any active run
            try:
                mlflow.end_run()
            except Exception:
                pass

    @staticmethod
    def _log_metric(metric: Metric, prefix: str = "", step: Optional[int] = None) -> None:
        """
        Safely log a Plexe Metric object to MLFlow.

        Args:
            metric: Plexe Metric object
            prefix: Optional prefix for the metric name
            step: Optional step (iteration) for the metric
        """
        if not mlflow.active_run():
            logger.warning("⚠️ Cannot log metric: No active run")
            return

        if not metric or not hasattr(metric, "name") or not hasattr(metric, "value"):
            logger.warning("⚠️ Cannot log invalid metric")
            return

        try:
            # Clean the metric name to ensure it's valid for MLFlow
            metric_name = re.sub(r"[^a-zA-Z0-9_]", "", f"{prefix}{metric.name}")

            # Convert value to float, or log as tag if not possible
            try:
                value = float(metric.value)

                # Log the metric
                if step is not None:
                    mlflow.log_metric(metric_name, value, step=step)
                else:
                    mlflow.log_metric(metric_name, value)

                logger.debug(f"✅ Logged metric: {metric_name}={value}")

            except (ValueError, TypeError) as e:
                logger.debug(f"⚠️ Could not convert metric {metric.name} to float: {e}")
                mlflow.set_tag(f"metric_{metric_name}", str(metric.value))
                mlflow.set_tag("non_numeric_metrics", "true")

        except Exception as e:
            logger.warning(f"⚠️ Failed to log metric {metric.name}: {e}")
