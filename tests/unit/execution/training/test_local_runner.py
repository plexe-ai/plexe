"""Tests for LocalProcessRunner GPU detection and command construction."""

import builtins
import sys
from unittest.mock import MagicMock, patch

import pytest

from plexe.execution.training.local_runner import (
    LocalProcessRunner,
    _detect_gpu_count,
    _detect_tf_gpu_count,
)

torch = pytest.importorskip("torch")
nn = torch.nn


class TestGPUDetection:
    """Tests for framework GPU detection helpers."""

    def test_no_torch(self):
        """Returns 0 when torch is not importable."""
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            assert _detect_gpu_count() == 0

    def test_no_cuda(self):
        """Returns 0 when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            assert _detect_gpu_count() == 0

    def test_with_cuda(self):
        """Returns device count when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=4):
            assert _detect_gpu_count() == 4

    def test_tf_gpu_detection_no_tf(self):
        """Returns 0 when tensorflow is not importable."""
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "tensorflow":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            assert _detect_tf_gpu_count() == 0


def _make_pytorch_model():
    """Create a simple PyTorch model for testing."""
    return nn.Linear(10, 1)


def _make_pytorch_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)


def _make_pytorch_loss():
    return nn.MSELoss()


def _run_and_capture_cmd(runner, template, gpu_count, mixed_precision=True, dataloader_workers=0):
    """Run training with mocked subprocess and return the constructed command."""
    model = _make_pytorch_model()
    optimizer = _make_pytorch_optimizer(model)
    loss = _make_pytorch_loss()

    captured_cmd = None

    def _capture_popen(cmd, **kwargs):
        nonlocal captured_cmd
        captured_cmd = cmd
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_proc.wait.return_value = 0
        return mock_proc

    gpu_patch = "plexe.execution.training.local_runner._detect_gpu_count"
    tf_gpu_patch = "plexe.execution.training.local_runner._detect_tf_gpu_count"

    with (
        patch(gpu_patch, return_value=gpu_count if "pytorch" in template else 0),
        patch(tf_gpu_patch, return_value=gpu_count if "keras" in template else 0),
        patch("subprocess.Popen", side_effect=_capture_popen),
        patch("torch.save"),
    ):
        try:
            runner.run_training(
                template=template,
                model=model,
                feature_pipeline=MagicMock(),
                train_uri="/tmp/train.parquet",
                val_uri="/tmp/val.parquet",
                timeout=300,
                target_columns=["target"],
                optimizer=optimizer,
                loss=loss,
                epochs=10,
                batch_size=32,
                mixed_precision=mixed_precision,
                dataloader_workers=dataloader_workers,
            )
        except Exception:
            pass  # We only care about the command

    return captured_cmd


class TestCommandConstruction:
    """Test that the runner builds the right command for different GPU configurations."""

    def setup_method(self):
        self.runner = LocalProcessRunner(work_dir="/tmp/test_runner")

    def test_pytorch_no_gpu_uses_python(self):
        """PyTorch with 0 GPUs should use the current Python launcher, no GPU flags."""
        cmd = _run_and_capture_cmd(self.runner, "train_pytorch", gpu_count=0)
        assert cmd is not None
        assert cmd[0] == sys.executable
        assert "--ddp" not in cmd
        assert "--mixed-precision" not in cmd

    def test_pytorch_single_gpu_no_ddp(self):
        """PyTorch with 1 GPU should use current Python (no DDP), but get --mixed-precision."""
        cmd = _run_and_capture_cmd(self.runner, "train_pytorch", gpu_count=1)
        assert cmd is not None
        assert cmd[0] == sys.executable
        assert "--ddp" not in cmd
        assert "--mixed-precision" in cmd

    def test_pytorch_multi_gpu_uses_distributed_run(self):
        """PyTorch with >1 GPU should use torch.distributed.run with --ddp and --mixed-precision."""
        cmd = _run_and_capture_cmd(self.runner, "train_pytorch", gpu_count=4)
        assert cmd is not None
        assert cmd[0] == sys.executable
        assert "-m" in cmd
        assert "torch.distributed.run" in cmd
        assert "--nproc_per_node=auto" in cmd
        assert "--standalone" in cmd
        assert "--ddp" in cmd
        assert "--mixed-precision" in cmd

    def test_pytorch_num_workers_passed(self):
        """PyTorch should pass --num-workers when dataloader_workers > 0."""
        cmd = _run_and_capture_cmd(self.runner, "train_pytorch", gpu_count=1, dataloader_workers=4)
        assert cmd is not None
        assert "--num-workers" in cmd
        idx = cmd.index("--num-workers")
        assert cmd[idx + 1] == "4"

    def test_pytorch_no_mixed_precision_when_disabled(self):
        """PyTorch with GPU but mixed_precision=False should not get --mixed-precision."""
        cmd = _run_and_capture_cmd(self.runner, "train_pytorch", gpu_count=1, mixed_precision=False)
        assert cmd is not None
        assert "--mixed-precision" not in cmd
