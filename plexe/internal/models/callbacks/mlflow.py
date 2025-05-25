"""
MLFlow callback for tracking model building process.

This module provides a callback implementation that logs model building
metrics, parameters, and artifacts to MLFlow.
"""

import os
import re
import tempfile
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import mlflow
import logging
import warnings

from plexe.callbacks import Callback, BuildStateInfo
from plexe.internal.models.entities.metric import Metric

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


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
            if mlflow.active_run():
                mlflow.end_run()

            # Configure MLFlow environment
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(self.connect_timeout)
            mlflow.set_tracking_uri(self.tracking_uri)

            # Explicitly set the experiment first to avoid default experiment use for traces
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
                mlflow.set_experiment(experiment_name=self.experiment_name)
            else:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                mlflow.set_experiment(experiment_id=self.experiment_id)

            # Enable autologging for smolagents AFTER setting experiment
            try:
                mlflow.smolagents.autolog()
            except ModuleNotFoundError:
                pass

        except Exception as e:
            logger.error(f"Failed to setup MLFlow: {e}")
            raise RuntimeError(f"Failed to setup MLFlow: {e}") from e

    @staticmethod
    def _timestamp() -> str:
        """Get formatted timestamp for runs and logs."""
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    @staticmethod
    def _safe_get(obj: Any, attrs: List[str], default: Any = None):
        """Safely access nested attributes."""
        if obj is None:
            return default

        current = obj
        for attr in attrs:
            if not hasattr(current, attr):
                return default
            current = getattr(current, attr)
        return current

    def _get_or_create_experiment(self) -> str:
        """
        Get or create the MLFlow experiment and return its ID.
        Reuses experiment_id if already set during initialization.
        """
        # If we already have an experiment ID, ensure it's active and return it
        if self.experiment_id:
            mlflow.set_experiment(experiment_id=self.experiment_id)
            return self.experiment_id

        # Otherwise, look up by name or create
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment:
            self.experiment_id = experiment.experiment_id
        else:
            # Create if not exists
            self.experiment_id = mlflow.create_experiment(self.experiment_name)

        # Set the experiment as active and notify user
        mlflow.set_experiment(experiment_id=self.experiment_id)
        print(f"✅ MLFlow: tracking URI '{self.tracking_uri}', experiment '{self.experiment_name}'")
        return self.experiment_id

    def _ensure_parent_run_active(self) -> bool:
        """Ensure the parent run is active, activating it if needed."""
        if not self.parent_run_id or not self.experiment_id:
            return False

        active_run = mlflow.active_run()

        # If already active and it's the parent run, we're good
        if active_run and active_run.info.run_id == self.parent_run_id:
            return True

        # End any active run and start the parent run
        if active_run:
            mlflow.end_run()

        # Ensure experiment is active before starting the run
        mlflow.set_experiment(experiment_id=self.experiment_id)

        try:
            mlflow.start_run(run_id=self.parent_run_id)
            return True
        except Exception as e:
            logger.warning(f"Could not activate parent run: {e}")
            return False

    @staticmethod
    def _safe_log_artifact(content: str, filename: str) -> None:
        """
        Safely log an artifact by writing to a temporary file first.

        Args:
            content: Content to write to the file
            filename: Name of the file in MLFlow
        """
        if not mlflow.active_run() or not content:
            return

        # Create unique temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=Path(filename).suffix) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Write content and log
            with open(tmp_path, "w") as f:
                f.write(content)

            mlflow.log_artifact(str(tmp_path))

        except Exception as e:
            logger.warning(f"Failed to log artifact '{filename}': {e}")
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()

    def _extract_model_context(self, info: BuildStateInfo) -> Dict[str, Any]:
        """Extract essential model context for logging."""
        context = {"intent": info.intent, "provider": str(info.provider)}

        # Add timing and iteration information
        if info.timeout:
            context["timeout_seconds"] = info.timeout
        if info.run_timeout:
            context["run_timeout_seconds"] = info.run_timeout
        if info.max_iterations:
            context["max_iterations"] = info.max_iterations

        # Add model ID if available
        if info.model_identifier:
            context["model_id"] = info.model_identifier

        # Add basic schema and dataset info
        if info.input_schema:
            context["input_schema_fields"] = len(info.input_schema.model_fields)
        if info.output_schema:
            context["output_schema_fields"] = len(info.output_schema.model_fields)
        if info.datasets:
            context["dataset_count"] = len(info.datasets)

        return context

    def _log_metric(self, metric: Metric, prefix: str = "", step: Optional[int] = None) -> None:
        """Safely log a Plexe Metric object to MLFlow."""
        if not mlflow.active_run() or not metric:
            return

        metric_name = self._safe_get(metric, ["name"])
        metric_value = self._safe_get(metric, ["value"])

        if not metric_name or metric_value is None:
            return

        # Clean metric name and convert value
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "", f"{prefix}{metric_name}")

        try:
            # Try to log as numeric
            value = float(metric_value)
            if step is not None:
                mlflow.log_metric(clean_name, value, step=step)
            else:
                mlflow.log_metric(clean_name, value)
        except (ValueError, TypeError):
            # If not numeric, log as tag
            mlflow.set_tag(f"metric_{clean_name}", str(metric_value))
            mlflow.set_tag("non_numeric_metrics", "true")

    def on_build_start(self, info: BuildStateInfo) -> None:
        """Start MLFlow parent run and log initial parameters."""
        try:
            # Ensure experiment is set and active
            self.experiment_id = self._get_or_create_experiment()

            # Get model info and timestamp
            model_id = (info.model_identifier or "unknown")[0:12] + "..."
            timestamp = self._timestamp()

            # End any active run before starting parent
            if mlflow.active_run():
                mlflow.end_run()

            # Ensure the experiment is active before starting the run
            mlflow.set_experiment(experiment_id=self.experiment_id)

            # Start parent run
            parent_run = mlflow.start_run(
                run_name=f"{model_id}-{timestamp}",
                experiment_id=self.experiment_id,
                description=f"Model building: {info.intent[:100]}...",
            )
            self.parent_run_id = parent_run.info.run_id
            logger.info(f"Started parent run '{parent_run.info.run_id}' in experiment '{self.experiment_name}'")

            # Log common parameters and tags
            mlflow.log_params(self._extract_model_context(info))
            mlflow.set_tags({"provider": str(info.provider), "run_type": "parent", "build_timestamp": timestamp})

            # Log intent
            if info.intent:
                self._safe_log_artifact(content=info.intent, filename="intent.txt")

        except Exception as e:
            logger.error(f"Error starting build in MLFlow: {e}")

    def on_iteration_start(self, info: BuildStateInfo) -> None:
        """Start a new nested child run for this iteration."""
        if not self.parent_run_id:
            return

        try:
            # Ensure experiment is active
            mlflow.set_experiment(experiment_id=self.experiment_id)

            # Create nested run under the parent, letting MLflow decide the name
            mlflow.start_run(
                experiment_id=self.experiment_id,
                nested=True,
                description=f"Iteration {info.iteration}",
            )

            # Log iteration parameters
            mlflow.log_params({"iteration": info.iteration})
            mlflow.set_tags(
                {"run_type": "iteration", "iteration": str(info.iteration), "parent_run_id": self.parent_run_id}
            )

            # Log datasets if available
            if info.datasets:
                for name, data in info.datasets.items():
                    try:
                        mlflow.log_input(mlflow.data.from_pandas(data.to_pandas(), name=name), context="training")
                    except Exception as e:
                        logger.warning(f"Could not log dataset '{name}': {e}")

        except Exception as e:
            logger.error(f"Error starting iteration in MLFlow: {e}")

    def on_iteration_end(self, info: BuildStateInfo) -> None:
        """Log metrics for this iteration and end the child run."""
        if not mlflow.active_run():
            return

        try:
            # Process node data if available
            node = info.node
            if node:
                # Log training code
                training_code = self._safe_get(node, ["training_code"])
                if training_code:
                    self._safe_log_artifact(
                        content=training_code, filename=f"trainer_source_iteration_{info.iteration}.py"
                    )

                # Log performance metrics
                performance = self._safe_get(node, ["performance"])
                if performance:
                    self._log_metric(performance)

                # Log execution time
                execution_time = self._safe_get(node, ["execution_time"])
                if execution_time:
                    mlflow.log_metric("execution_time", execution_time)

                # Log exception information
                exception_raised = self._safe_get(node, ["exception_was_raised"], False)
                if exception_raised:
                    exception_obj = self._safe_get(node, ["exception"])
                    exception_type = type(exception_obj).__name__ if exception_obj else "unknown"

                    mlflow.set_tags({"exception_raised": "true", "exception_type": exception_type})

                    # Log exception details
                    if exception_obj:
                        self._safe_log_artifact(
                            content=str(exception_obj), filename=f"exception-iteration-{info.iteration}.txt"
                        )

                # Log model artifacts
                artifacts = self._safe_get(node, ["model_artifacts"], [])
                for artifact in artifacts:
                    if Path(artifact).exists():
                        try:
                            mlflow.log_artifact(str(artifact))
                        except Exception:
                            pass

            # Determine run status
            status = "FINISHED"
            performance = self._safe_get(node, ["performance"])
            if (
                self._safe_get(node, ["exception_was_raised"], False)
                or performance is None
                or (hasattr(performance, "is_worst") and performance.is_worst)
            ):
                status = "FAILED"

            # End the child run
            mlflow.end_run(status=status)

        except Exception as e:
            logger.error(f"Error ending iteration in MLFlow: {e}")
            try:
                mlflow.end_run(status="FAILED")
            except Exception:
                pass

    def on_build_end(self, info: BuildStateInfo) -> None:
        """Log final model details and end MLFlow parent run."""
        try:
            # End any active child run first
            active_run = mlflow.active_run()
            if active_run and active_run.info.run_id != self.parent_run_id:
                mlflow.end_run()

            # Ensure parent run is active
            if not self._ensure_parent_run_active():
                return

            # Log EDA reports
            node_metadata = self._safe_get(info.node, ["metadata"], {})
            if node_metadata and "eda_markdown_reports" in node_metadata:
                for dataset_name, report_markdown in node_metadata["eda_markdown_reports"].items():
                    self._safe_log_artifact(content=report_markdown, filename=f"eda_report_{dataset_name}.md")

            # Log model information
            if info.final_metric and hasattr(info.final_metric, "name") and hasattr(info.final_metric, "value"):
                mlflow.log_metric(f"best_{info.final_metric.name}", float(info.final_metric.value))

            # Log model artifacts and status
            mlflow.set_tag("best_iteration", str(info.iteration))

            # Log artifact names
            if info.final_artifacts:
                artifact_names = [a.name for a in info.final_artifacts]
                mlflow.set_tag("model_artifacts", ", ".join(artifact_names))

            # Log model state
            if info.model_state:
                mlflow.set_tag("final_model_state", str(info.model_state))

            # Log final model code
            if info.trainer_source:
                self._safe_log_artifact(content=info.trainer_source, filename="final_trainer.py")

            if info.predictor_source:
                self._safe_log_artifact(content=info.predictor_source, filename="final_predictor.py")

            # End the parent run
            mlflow.end_run()

        except Exception as e:
            logger.error(f"Error finalizing build in MLFlow: {e}")
            try:
                mlflow.end_run()
            except Exception:
                pass
