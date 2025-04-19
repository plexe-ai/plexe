"""
Unit tests for the ProcessExecutor class and its associated components.

These tests validate the functionality of the following:
- RedirectQueue: Ensures that stdout and stderr redirection to a queue behaves as expected.
- ProcessExecutor: Tests execution of Python code in an isolated process, including handling of:
  - Successful execution.
  - Timeouts.
  - Exceptions raised during execution.
  - Dataset handling and working directory creation.

The tests use pytest as the test runner and employ mocking to isolate external dependencies.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow
import pytest

from plexe.internal.common.datasets.tabular import TabularDataset
from plexe.internal.models.execution.executor import ExecutionResult
from plexe.internal.models.execution.process_executor import ProcessExecutor


class TestProcessExecutor:
    def setup_method(self):
        self.execution_id = "test_execution"
        self.code = "print('Hello, World!')"
        self.working_dir = Path(os.getcwd()) / self.execution_id
        self.timeout = 5
        self.datasets = {"training_data": TabularDataset(pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}))}
        self.process_executor = ProcessExecutor(
            execution_id=self.execution_id,
            code=self.code,
            working_dir=Path(os.getcwd()),
            datasets=self.datasets,
            timeout=self.timeout,
            code_execution_file_name="run.py",
        )

    def teardown_method(self):
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir, ignore_errors=True)

    def test_constructor_creates_working_directory(self):
        assert self.working_dir.exists()

    @patch("pyarrow.parquet.write_table")
    def test_run_successful_execution(self, mock_write_table):
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("Execution completed", "")
        mock_process.returncode = 0

        with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
            result = self.process_executor.run()

        dataset_file = self.working_dir / "training_data.parquet"
        mock_write_table.assert_called_once_with(
            pyarrow.Table.from_pandas(self.datasets["training_data"].to_pandas()), dataset_file
        )
        mock_popen.assert_called_once_with(
            [sys.executable, str(self.working_dir / "run.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.working_dir),
            text=True,
        )
        assert isinstance(result, ExecutionResult)
        assert "Execution completed" in result.term_out
        assert result.exception is None

    @patch("subprocess.Popen")
    def test_run_timeout(self, mock_popen):
        mock_process = MagicMock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=self.timeout)

        with patch("subprocess.Popen", return_value=mock_process):
            result = self.process_executor.run()

        assert isinstance(result, ExecutionResult)
        assert isinstance(result.exception, TimeoutError)
        assert result.exec_time == self.timeout

    @patch("subprocess.Popen")
    def test_run_exception(self, mock_popen):
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("", "RuntimeError: Something went wrong")
        mock_process.returncode = 1

        with patch("subprocess.Popen", return_value=mock_process):
            result = self.process_executor.run()

        assert isinstance(result, ExecutionResult)
        assert isinstance(result.exception, RuntimeError)
        assert "Something went wrong" in str(result.exception)

    @patch("pyarrow.parquet.write_table")
    def test_dataset_written_to_file(self, mock_write_table):
        self.process_executor.run()
        dataset_file = self.working_dir / "training_data.parquet"
        mock_write_table.assert_called_once_with(
            pyarrow.Table.from_pandas(self.datasets["training_data"].to_pandas()), dataset_file
        )


if __name__ == "__main__":
    pytest.main()
