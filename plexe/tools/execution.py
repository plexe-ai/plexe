"""
Tools related to code execution, including running training code in isolated environments and
applying feature transformations to datasets.

These tools automatically handle model artifact registration through the ArtifactRegistry,
ensuring that artifacts generated during the execution can be retrieved later in the pipeline.
"""

import logging
import uuid
import types
import warnings
from typing import Dict, List, Callable, Type

from smolagents import tool

from plexe.callbacks import Callback
from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.datasets.adapter import DatasetAdapter
from plexe.core.object_registry import ObjectRegistry
from plexe.internal.models.entities.code import Code
from plexe.internal.models.entities.artifact import Artifact
from plexe.internal.models.entities.metric import Metric, MetricComparator, ComparisonMethod
from plexe.internal.models.entities.node import Node
from plexe.internal.models.execution.process_executor import ProcessExecutor

logger = logging.getLogger(__name__)


def get_executor_tool(distributed: bool = False) -> Callable:
    """Get the appropriate executor tool based on the distributed flag."""

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
        # Log the distributed flag
        logger.debug(f"execute_training_code called with distributed={distributed}")

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
            # Create state info once for all callbacks
            state_info = BuildStateInfo(
                intent="Unknown",  # Will be filled by agent context
                provider="Unknown",  # Will be filled by agent context
                input_schema=None,  # Will be filled by agent context
                output_schema=None,  # Will be filled by agent context
                datasets=datasets,
                iteration=0,  # Default value, no longer used for MLFlow run naming
                node=node,
            )

            # Notify all callbacks about execution start
            _notify_callbacks(object_registry.get_all(Callback), "start", state_info)

            # Import here to avoid circular imports
            from plexe.config import config

            # Get the appropriate executor class via the factory
            executor_class = _get_executor_class(distributed=distributed)

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

            # Execute and collect results - ProcessExecutor.run() handles cleanup internally
            logger.debug(f"Executing node {node} using executor {executor}")
            result = executor.run()
            logger.debug(f"Execution result: {result}")
            node.execution_time = result.exec_time
            node.execution_stdout = result.term_out
            node.exception_was_raised = result.exception is not None
            node.exception = result.exception or None
            node.model_artifacts = result.model_artifacts

            # Handle the performance metric properly using the consolidated validation logic
            performance_value = None
            is_worst = True

            if result.is_valid_performance():
                performance_value = result.performance
                is_worst = False

            # Create a metric object with proper handling of None or invalid values
            node.performance = Metric(
                name=metric_to_optimise_name,
                value=performance_value,
                comparator=MetricComparator(comparison_method=comparison_method),
                is_worst=is_worst,
            )

            node.training_code = code

            # Notify callbacks about the execution end with the same state_info
            # The node reference in state_info automatically reflects the updates to node
            _notify_callbacks(object_registry.get_all(Callback), "end", state_info)

            # Check if the execution failed in any way
            if node.exception is not None:
                raise RuntimeError(f"Execution failed with exception: {node.exception}")
            if not result.is_valid_performance():
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


def _get_executor_class(distributed: bool = False) -> Type:
    """Get the appropriate executor class based on the distributed flag.

    Args:
        distributed: Whether to use distributed execution if available

    Returns:
        Executor class (not instance) appropriate for the environment
    """
    # Log the distributed flag
    logger.debug(f"get_executor_class using distributed={distributed}")
    if distributed:
        try:
            # Try to import Ray executor
            from plexe.internal.models.execution.ray_executor import RayExecutor

            logger.debug("Using Ray for distributed execution")
            return RayExecutor
        except ImportError:
            # Fall back to process executor if Ray is not available
            logger.warning("Ray not available, falling back to ProcessExecutor")
            return ProcessExecutor

    # Default to ProcessExecutor for non-distributed execution
    logger.debug("Using ProcessExecutor (non-distributed)")
    return ProcessExecutor


def _notify_callbacks(callbacks: Dict, event_type: str, build_state_info) -> None:
    """Helper function to notify callbacks with consistent error handling.

    Args:
        callbacks: Dictionary of callbacks from the registry
        event_type: The event type - either "start" or "end"
        build_state_info: The state info to pass to callbacks
    """
    method_name = f"on_iteration_{event_type}"

    for callback in callbacks.values():
        try:
            getattr(callback, method_name)(build_state_info)
        except Exception as e:
            # Log full stack trace at debug level
            import traceback

            logger.debug(
                f"Error in callback {callback.__class__.__name__}.{method_name}: {e}\n{traceback.format_exc()}"
            )
            # Log a shorter message at warning level
            logger.warning(f"Error in callback {callback.__class__.__name__}.{method_name}: {str(e)[:50]}")


@tool
def apply_feature_transformer(dataset_name: str) -> Dict:
    """
    Applies a feature transformer to datasets and registers the transformed datasets. The name of the
    new transformed dataset is returned in the response.

    Args:
        dataset_name: Name of datasets to transform

    Returns:
        Dictionary with results of transformation:
        - success: Boolean indicating success or failure
        - original_dataset_name: Name of the original dataset
        - new_dataset_name: Name of the transformed dataset
    """
    object_registry = ObjectRegistry()

    try:
        # Get feature transformer code from registry
        code_obj = object_registry.get(Code, "feature_transformations")
        transformer_code = code_obj.code

        # Load code as module
        module = types.ModuleType("feature_transformer_module")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(transformer_code, module.__dict__)

        # Instantiate transformer
        transformer = module.FeatureTransformerImplementation()

        # Get dataset
        dataset = object_registry.get(TabularConvertible, dataset_name)
        df = dataset.to_pandas()

        # Apply transformation
        transformed_df = transformer.transform(df)

        # Register transformed dataset
        transformed_name = f"{dataset_name}_transformed"
        transformed_ds = DatasetAdapter.coerce(transformed_df)
        object_registry.register(TabularConvertible, transformed_name, transformed_ds, overwrite=True, immutable=True)

        logger.debug(f"✅ Applied feature transformer to {dataset_name} → {transformed_name}")

        return {"success": True, "original_dataset_name": dataset_name, "new_dataset_name": transformed_name}
    except Exception as e:
        import traceback

        logger.debug(f"Error applying feature transformer: {str(e)}\n{traceback.format_exc()}")
        return {"success": False, "error": str(e), "transformed_datasets": []}


@tool
def get_model_artifacts() -> List[str]:
    """
    Get all registered model artifact names.

    Returns:
        List of artifact names (e.g., ["model.pkl", "scaler.pkl", ...])
    """
    object_registry = ObjectRegistry()

    try:
        # Get all artifact names from registry
        artifact_names = object_registry.list_by_type(Artifact)
        return artifact_names
    except Exception as e:
        logger.warning(f"⚠️ Error getting model artifacts: {str(e)}")
        return []
