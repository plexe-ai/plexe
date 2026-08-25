"""Tests for LocalProcessRunner training timeout enforcement."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from plexe.execution.training.local_runner import LocalProcessRunner
from plexe.models import TrainingError


class _SilentProcess:
    """Fake subprocess that produces no output until killed, like a quiet training run."""

    def __init__(self):
        self._killed = threading.Event()
        self.stdout = self
        self.kill_count = 0

    def readline(self):
        # Blocks like a pipe read on a silent child; returns EOF once killed.
        self._killed.wait(timeout=60)
        return ""

    def kill(self):
        self.kill_count += 1
        self._killed.set()

    def wait(self, timeout=None):
        return -9

    def close(self):
        pass


def test_timeout_enforced_when_process_is_silent(tmp_path):
    """run_training must raise TrainingError after timeout even with no subprocess output."""
    proc = _SilentProcess()
    runner = LocalProcessRunner(work_dir=str(tmp_path / "runs"))

    with patch("subprocess.Popen", return_value=proc), pytest.raises(TrainingError, match="timed out"):
        runner.run_training(
            template="train_xgboost",
            model=object(),
            feature_pipeline=MagicMock(),
            train_uri=str(tmp_path / "train.parquet"),
            val_uri=str(tmp_path / "val.parquet"),
            timeout=2,
            target_columns=["target"],
        )

    assert proc.kill_count == 1
