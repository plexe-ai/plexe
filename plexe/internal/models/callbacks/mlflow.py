"""
MLFlow callback for tracking model building process.

This module provides a callback implementation that logs model building
metrics, parameters, and artifacts to MLFlow.
"""

import re
import mlflow
import logging
import warnings
from pathlib import Path

from plexe.callbacks import Callback, BuildStateInfo
from plexe.internal.models.entities.metric import Metric

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


class MLFlowCallback(Callback):
    """
    Callback that logs the model building process to MLFlow.

    This callback hooks into the model building process and logs metrics,
    parameters, and artifacts to MLFlow for tracking and visualization.
    """

    def __init__(self, tracking_uri: str, experiment_name: str, connect_timeout: int = 10):
        """
        Initialize MLFlow callback.

        :param tracking_uri: Optional MLFlow tracking server URI.
        :param experiment_name: Name for the MLFlow experiment. Defaults to "plexe".
        :param connect_timeout: Timeout in seconds for MLFlow server connection. Defaults to 10.
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.experiment_id = None
        self.connect_timeout = connect_timeout

        # Clean up active runs, if any
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception as e:
            raise RuntimeError(f"❌  Error cleaning up active runs: {e}") from e

        # Set up MLFlow tracking URI and experiment
        try:
            # Set connection timeout for API calls
            import os

            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(self.connect_timeout)
            mlflow.set_tracking_uri(tracking_uri)
            logger.debug(f"✅  MLFlow configured with tracking URI '{tracking_uri}'")

        except Exception as e:
            raise RuntimeError(f"❌  Error setting up MLFlow: {e}") from e

    def on_build_start(self, info: BuildStateInfo) -> None:
        """
        Start MLFlow parent run and log initial parameters.

        :param info: Information about the model building process start.
        """
        # Set or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        logger.debug(f"✅  MLFlow configured with experiment '{self.experiment_name}' (ID: {self.experiment_id})")
        print(f"✅  MLFlow: tracking URI '{self.tracking_uri}', experiment '{self.experiment_name}'")
        # TODO: Start an MLFlow parent run

    def on_build_end(self, info: BuildStateInfo) -> None:
        """
        Log final model details and end MLFlow parent run.

        :param info: Information about the model building process end.
        """
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception as e:
            raise RuntimeError(f"❌  Error cleaning up MLFlow run: {e}") from e

    def on_iteration_start(self, info: BuildStateInfo) -> None:
        """
        Start a new child run for this iteration if using nested runs.

        :param info: Information about the iteration start.
        """
        import datetime

        timestamp = datetime.datetime.now().isoformat().replace(":", "-").replace(".", "-")
        run_name = f"run-{timestamp}"
        mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
        )
        logger.debug(f"✅  Started MLFlow run: {run_name}")

        # Log training datasets used
        for name, data in info.datasets.items():
            mlflow.log_input(mlflow.data.from_pandas(data.to_pandas(), name=name), context="training")

        # Log model parameters
        mlflow.log_params(
            {
                "intent": info.intent,
                # "input_schema": str(info.input_schema.model_fields),
                # "output_schema": str(info.output_schema.model_fields),
                "provider": str(info.provider),
                "run_timeout": info.run_timeout,
                "iteration": info.iteration,
            }
        )
        mlflow.set_tags(
            {
                "provider": str(info.provider),
            }
        )

    def on_iteration_end(self, info: BuildStateInfo) -> None:
        """
        Log metrics for this iteration.

        :param info: Information about the iteration end.
        """
        if not mlflow.active_run():
            return

        # Log validation datasets used
        for name, data in info.datasets.items():
            mlflow.log_input(mlflow.data.from_pandas(data.to_pandas(), name=name), context="validation")

        if info.node.training_code:
            try:
                # Save code to a file first, then log it
                code_path = Path("trainer_source.py")
                with open(code_path, "w") as f:
                    f.write(info.node.training_code)
                mlflow.log_artifact(str(code_path))
            except Exception as e:
                logger.warning(f"Could not log trainer source: {e}")

        # Log node performance if available
        if info.node.performance:
            self._log_metric(info.node.performance)

        # Log execution time
        if info.node.execution_time:
            mlflow.log_metric("execution_time", info.node.execution_time)

        # Log whether exception was raised
        if info.node.exception_was_raised:
            mlflow.set_tags({"exception_was_raised": True, "exception": type(info.node.exception)})

        # Log model artifacts if any
        if info.node.model_artifacts:
            for artifact in info.node.model_artifacts:
                if Path(artifact).exists():
                    try:
                        mlflow.log_artifact(str(artifact))
                    except Exception as e:
                        logger.debug(f"Could not log artifact {artifact}: {e}")

        try:
            is_failed = (
                info.node.exception_was_raised or info.node.performance is None or info.node.performance.is_worst
            )
            mlflow.end_run(status="FAILED" if is_failed else "FINISHED")
        except Exception as e:
            logger.debug(f"Error ending MLFlow run: {e}")

    @staticmethod
    def _log_metric(metric: Metric, prefix: str = "", step: int = None) -> None:
        """
        Log a Plexe Metric object to MLFlow.

        :param metric: Plexe Metric object
        :param prefix: Optional prefix for the metric name
        :param step: Optional step (iteration) for the metric
        """
        if mlflow is None or not mlflow.active_run():
            return

        if metric and hasattr(metric, "name") and hasattr(metric, "value"):
            try:
                value = float(metric.value)
                metric_name = re.sub(r"[^a-zA-Z0-9]", "", f"{prefix}{metric.name}")

                if step is not None:
                    mlflow.log_metric(metric_name, value, step=step)
                else:
                    mlflow.log_metric(metric_name, value)
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not convert metric {metric.name} value to float: {e}")
                # Try to log as tag instead
                mlflow.set_tag("performance_is_invalid", True)
