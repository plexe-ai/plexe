"""
Unit tests for PyTorch model submission.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


from plexe.models import BuildContext
from plexe.tools.submission import get_save_model_fn


def test_save_model_pytorch(tmp_path):
    """Test PyTorch model submission validation and context scratch storage."""
    context = BuildContext(
        user_id="test_user",
        experiment_id="exp_pytorch",
        dataset_uri="file:///tmp/train.parquet",
        work_dir=tmp_path,
        intent="predict something",
    )

    save_model = get_save_model_fn(context, "pytorch", max_epochs=10)

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.MSELoss()

    result = save_model(model, optimizer, loss, epochs=5, batch_size=32)

    assert "pytorch model saved" in result.lower()
    assert context.scratch["_saved_model"] is model
    assert context.scratch["_saved_optimizer"] is optimizer
    assert context.scratch["_saved_loss"] is loss
    assert context.scratch["_nn_epochs"] == 5
    assert context.scratch["_nn_batch_size"] == 32
