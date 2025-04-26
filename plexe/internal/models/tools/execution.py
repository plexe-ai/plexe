"""
Tools related to code execution, including running training code in isolated environments.

These tools automatically handle model artifact registration through the ArtifactRegistry,
ensuring that artifacts generated during the execution can be retrieved later in the pipeline.
"""

import logging
import uuid
from typing import Dict, List, Callable

from smolagents import tool

from plexe.callbacks import Callback
from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.registries.objects import ObjectRegistry
from plexe.internal.models.entities.code import Code
from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from plexe.internal.models.entities.node import Node
from plexe.internal.models.execution.process_executor import ProcessExecutor
from typing import Type

logger = logging.getLogger(__name__)


def get_executor_tool() -> Callable:
    """Get the executor tool for training code execution."""

    @tool
    def execute_training_code(
        node_id: str,
        code: str,
        working_dir: str,
        dataset_names: List[str],
        timeout: int,
        metric_to_optimise_name: str,
        metric_to_optimise_comparison_method: str,
    ) -> Dict:
        """Executes training code in an isolated environment.

        Args:
            node_id: Unique identifier for this execution
            code: The code to execute
            working_dir: Directory to use for execution
            dataset_names: List of dataset names to retrieve from the registry
            timeout: Maximum execution time in seconds
            metric_to_optimise_name: The name of the metric to optimize for
            metric_to_optimise_comparison_method: The comparison method for the metric

        Returns:
            A dictionary containing execution results with model artifacts and their registry names
        """
        logger.info(f"execute_training_code for node_id={node_id}")

        from plexe.callbacks import BuildStateInfo

        object_registry = ObjectRegistry()

        execution_id = f"{node_id}-{uuid.uuid4()}"
        try:
            # Get actual datasets from registry
            datasets = object_registry.get_multiple(TabularConvertible, dataset_names)

            # Convert string to enum if needed
            if "HIGHER_IS_BETTER" in metric_to_optimise_comparison_method:
                comparison_method = ComparisonMethod.HIGHER_IS_BETTER
            elif "LOWER_IS_BETTER" in metric_to_optimise_comparison_method:
                comparison_method = ComparisonMethod.LOWER_IS_BETTER
            elif "TARGET_IS_BETTER" in metric_to_optimise_comparison_method:
                comparison_method = ComparisonMethod.TARGET_IS_BETTER
            else:
                comparison_method = ComparisonMethod.HIGHER_IS_BETTER

            # Create a node to store execution results
            node = Node(solution_plan="")  # We only need this for execute_node

            # Get callbacks from the registry and notify them
            node.training_code = code
            # Notification for iteration end with all required fields
            for callback in object_registry.get_all(Callback).values():
                try:
                    callback.on_iteration_start(
                        BuildStateInfo(
                            intent="goobers",  # Will be filled by agent context
                            provider="openai/gpt-4o",  # Will be filled by agent context
                            input_schema=None,  # Will be filled by agent context
                            output_schema=None,  # Will be filled by agent context
                            datasets=datasets,
                            iteration=0,  # Default value, no longer used for MLFlow run naming
                            node=node,
                        )
                    )
                except Exception as e:
                    # Log full stack trace at debug level
                    import traceback

                    logger.debug(
                        f"Error in callback {callback.__class__.__name__}.on_iteration_end: {e}\n{traceback.format_exc()}"
                    )

                    # Log a shorter message at warning level
                    logger.warning(f"Error in callback {callback.__class__.__name__}.on_iteration_end: {str(e)[:50]}")

            # Import here to avoid circular imports
            from plexe.config import config

            # Get the appropriate executor class via the factory
            executor_class = _get_executor_class()

            # Create an instance of the executor
            logger.debug(f"Creating {executor_class.__name__} for execution ID: {execution_id}")
            executor = executor_class(
                execution_id=execution_id,
                code=code,
                working_dir=working_dir,
                datasets=datasets,
                timeout=timeout,
                code_execution_file_name=config.execution.runfile_name,
            )

            logger.debug(f"Executing node {node} using executor {executor}")
            result = executor.run()
            logger.debug(f"Execution result: {result}")
            node.execution_time = result.exec_time
            node.execution_stdout = result.term_out
            node.exception_was_raised = result.exception is not None
            node.exception = result.exception or None
            node.model_artifacts = result.model_artifacts

            # Handle the performance metric properly
            performance_value = None
            is_worst = True

            if result.performance is not None and isinstance(result.performance, (int, float)):
                performance_value = result.performance
                is_worst = False

                if (
                    result.performance
                    in [float("inf"), float("-inf")]
                    # or result.performance < 1e-7
                    # or result.performance > 1 - 1e-7
                ):
                    performance_value = None
                    is_worst = True

            # Create a metric object with proper handling of None or invalid values
            node.performance = Metric(
                name=metric_to_optimise_name,
                value=performance_value,
                comparator=MetricComparator(comparison_method=comparison_method),
                is_worst=is_worst,
            )

            node.training_code = code

            # Notify callbacks about the execution end
            # Notification for iteration end with all required fields
            for callback in object_registry.get_all(Callback).values():
                try:
                    # Create build state info with required fields
                    # Some fields like intent, input_schema, etc. will be empty here
                    # but will be filled in by the model builder agent context
                    callback.on_iteration_end(
                        BuildStateInfo(
                            intent="goobers",  # Will be filled by agent context
                            provider="openai/gpt-4o",  # Will be filled by agent context
                            input_schema=None,  # Will be filled by agent context
                            output_schema=None,  # Will be filled by agent context
                            datasets=datasets,
                            iteration=0,  # Default value, no longer used for MLFlow run naming
                            node=node,
                        )
                    )
                except Exception as e:
                    # Log full stack trace at debug level
                    import traceback

                    logger.debug(
                        f"Error in callback {callback.__class__.__name__}.on_iteration_end: {e}\n{traceback.format_exc()}"
                    )

                    # Log a shorter message at warning level
                    logger.warning(f"Error in callback {callback.__class__.__name__}.on_iteration_end: {str(e)[:50]}")

            # Check if the execution failed in any way
            if node.exception is not None:
                raise RuntimeError(f"Execution failed with exception: {node.exception}")
            if (
                result.performance is None
                or not isinstance(result.performance, (int, float))
                or result.performance in [float("inf"), float("-inf")]
                # or result.performance < 1e-7
                # or result.performance > 1 - 1e-7
            ):
                raise RuntimeError(f"Execution failed due to not producing a valid performance: {result.performance}")

            # Register code and artifacts
            artifact_paths = node.model_artifacts if node.model_artifacts else []
            artifacts = [Artifact.from_path(p) for p in artifact_paths]
            object_registry.register_multiple(Artifact, {a.name: a for a in artifacts})
            object_registry.register(Code, execution_id, Code(node.training_code))

            # Return results
            return {
                "success": not node.exception_was_raised,
                "performance": (
                    {
                        "name": node.performance.name if node.performance else None,
                        "value": node.performance.value if node.performance else None,
                        "comparison_method": (
                            str(node.performance.comparator.comparison_method) if node.performance else None
                        ),
                    }
                    if node.performance
                    else None
                ),
                "exception": str(node.exception) if node.exception else None,
                "model_artifact_names": [a.name for a in artifacts],
                "training_code_id": execution_id,
            }
        except Exception as e:
            # Log full stack trace at debug level
            import traceback

            logger.debug(f"Error executing training code: {str(e)}\n{traceback.format_exc()}")

            return {
                "success": False,
                "performance": None,
                "exception": str(e),
                "model_artifact_names": [],
            }

    return execute_training_code


def _get_executor_class() -> Type:
    """Get the appropriate executor class based on Ray availability.

    Returns:
        Executor class (not instance) appropriate for the environment
    """
    # Check if Ray is available
    try:
        import ray
    except ImportError:
        logger.warning("Ray not available, using ProcessExecutor")
        return ProcessExecutor

    # Check if Ray is initialized
    if ray.is_initialized():
        try:
            # Try to import Ray executor
            from plexe.internal.models.execution.ray_executor import RayExecutor

            logger.info("Using Ray for execution (Ray is initialized)")
            return RayExecutor
        except ImportError:
            # Fall back to process executor if Ray executor is not available
            logger.warning("Ray initialized but RayExecutor not found, falling back to ProcessExecutor")
            return ProcessExecutor

    # Fall back to configuration-based decision if Ray is not initialized
    # This maintains backward compatibility
    from plexe.config import config

    ray_configured = hasattr(config, "ray") and (
        getattr(config.ray, "address", None) is not None
        or getattr(config.ray, "num_gpus", None) is not None
        or getattr(config.ray, "num_cpus", None) is not None
    )

    if ray_configured:
        try:
            # Try to import Ray executor
            from plexe.internal.models.execution.ray_executor import RayExecutor

            logger.info("Using Ray for execution based on configuration")
            return RayExecutor
        except ImportError:
            # Fall back to process executor if Ray executor is not available
            logger.warning("Ray configured but RayExecutor not available, falling back to ProcessExecutor")
            return ProcessExecutor

    # Default to ProcessExecutor when Ray is not available
    logger.info("Using ProcessExecutor (Ray not initialized or configured)")
    return ProcessExecutor
