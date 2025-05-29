"""Module: RayExecutor for Distributed Python Code Execution

This module provides an implementation of the `Executor` interface for executing Python code snippets
in a distributed Ray cluster. It leverages Ray to run code remotely, with support for parallel execution.

Classes:
    - RayExecutor: A class to execute Python code snippets in a Ray cluster.

Usage:
    Create an instance of `RayExecutor`, providing the Python code, working directory, and timeout.
    Call the `run` method to execute the code on the Ray cluster and return the results in an
    `ExecutionResult` object.
"""

import logging
import time
import ray
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from typing import Dict, List

from plexe.internal.common.datasets.interface import TabularConvertible
from plexe.internal.common.utils.response import extract_performance
from plexe.internal.models.execution.executor import ExecutionResult, Executor
from plexe.config import config

logger = logging.getLogger(__name__)


@ray.remote
def _run_code(code: str, working_dir: str, dataset_files: List[str], timeout: int) -> dict:
    """Ray remote function that executes the code."""
    import subprocess
    import sys
    from pathlib import Path

    working_dir = Path(working_dir)
    code_file = working_dir / "run.py"

    # Write code to file
    with open(code_file, "w", encoding="utf-8") as f:
        f.write("import os\nimport sys\nfrom pathlib import Path\n\n" + code)

    start_time = time.time()
    process = subprocess.Popen(
        [sys.executable, str(code_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(working_dir),
        text=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
        exec_time = time.time() - start_time

        # Collect model artifacts
        model_artifacts = []
        model_dir = working_dir / "model_files"
        if model_dir.exists() and model_dir.is_dir():
            model_artifacts.append(str(model_dir))
        else:
            for file in working_dir.iterdir():
                if file != code_file and str(file) not in dataset_files:
                    model_artifacts.append(str(file))

        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": process.returncode,
            "exec_time": exec_time,
            "model_artifacts": model_artifacts,
        }
    except subprocess.TimeoutExpired:
        process.kill()
        return {
            "stdout": "",
            "stderr": f"Execution exceeded {timeout}s timeout",
            "returncode": -1,
            "exec_time": timeout,
            "model_artifacts": [],
        }


class RayExecutor(Executor):
    """Execute Python code snippets on a Ray cluster."""

    _ray_was_used = False

    def __init__(
        self,
        execution_id: str,
        code: str,
        working_dir: Path | str,
        datasets: Dict[str, TabularConvertible],
        timeout: int,
        code_execution_file_name: str = config.execution.runfile_name,
    ):
        """Initialize the RayExecutor.

        Args:
            execution_id (str): Unique ID for this execution.
            code (str): The Python code to execute.
            working_dir (Path | str): The working directory for execution.
            datasets (Dict[str, TabularConvertible]): The datasets to be used.
            timeout (int): The maximum allowed execution time in seconds.
            code_execution_file_name (str): The filename to use for the executed script.
        """
        RayExecutor._ray_was_used = True
        super().__init__(code, timeout)
        self.working_dir = Path(working_dir).resolve() / execution_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.code_file_name = code_execution_file_name
        self.dataset = datasets

        # Initialize Ray if not already done
        if not ray.is_initialized():
            from plexe.config import config

            ray.init(
                address=getattr(config.ray, "address", None) if hasattr(config, "ray") else None,
                num_cpus=getattr(config.ray, "num_cpus", None) if hasattr(config, "ray") else None,
                num_gpus=getattr(config.ray, "num_gpus", None) if hasattr(config, "ray") else None,
                ignore_reinit_error=True,
            )

    def run(self) -> ExecutionResult:
        """Execute code using Ray and return results."""
        logger.debug(f"RayExecutor is executing code with working directory: {self.working_dir}")

        # Write datasets to files
        dataset_files = []
        for dataset_name, dataset in self.dataset.items():
            dataset_file = self.working_dir / f"{dataset_name}.parquet"
            pq.write_table(pa.Table.from_pandas(df=dataset.to_pandas()), dataset_file)
            dataset_files.append(str(dataset_file))

        try:
            # Execute the code using Ray
            result_ref = _run_code.remote(self.code, str(self.working_dir), dataset_files, self.timeout)
            # Wait for result with timeout
            ready_refs, remaining_refs = ray.wait([result_ref], timeout=self.timeout)

            # If no ready refs, we hit a timeout
            if not ready_refs:
                ray.cancel(result_ref, force=True)
                return ExecutionResult(
                    term_out=[],
                    exec_time=self.timeout,
                    exception=TimeoutError(f"Execution exceeded {self.timeout}s timeout - Ray timeout reached"),
                )

            # Get the result from the completed task
            result = ray.get(ready_refs[0])

            if result["returncode"] != 0:
                return ExecutionResult(
                    term_out=[result["stdout"]],
                    exec_time=result["exec_time"],
                    exception=RuntimeError(result["stderr"]),
                    model_artifact_paths=result["model_artifacts"],
                )

            return ExecutionResult(
                term_out=[result["stdout"]],
                exec_time=result["exec_time"],
                model_artifact_paths=result["model_artifacts"],
                performance=extract_performance(result["stdout"]),
            )

        except ray.exceptions.GetTimeoutError:
            ray.cancel(result_ref, force=True)
            return ExecutionResult(
                term_out=[],
                exec_time=self.timeout,
                exception=TimeoutError(f"Execution exceeded {self.timeout}s timeout - Ray timeout reached"),
            )

    def cleanup(self) -> None:
        """Clean up Ray resources if needed."""
        pass
