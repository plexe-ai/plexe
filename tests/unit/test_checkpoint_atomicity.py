"""Checkpoint publication must preserve the last resumable state on failure."""

import json
from pathlib import Path

import pytest

from plexe import checkpointing
from plexe.models import BuildContext


@pytest.fixture
def context(tmp_path):
    """Build real workflow state without running a model or a provider."""
    return BuildContext("user", "experiment", "data.parquet", tmp_path, "Predict a value")


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", [OSError("disk full"), ValueError("serialization failed"), KeyboardInterrupt()])
def test_failed_save_preserves_previous_checkpoint(tmp_path, context, monkeypatch, existing, failure):
    """A partial write must never replace a valid checkpoint or appear as one."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    destination = checkpoint_dir / "prepare_data.json"
    previous = b'{"phase": "prepare_data", "context": {"intent": "previous"}}'
    if existing:
        destination.write_bytes(previous)

    def failing_dump(data, stream, **kwargs):
        stream.write('{"context": ')
        raise failure

    monkeypatch.setattr(checkpointing.json, "dump", failing_dump)
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            checkpointing.save_checkpoint("experiment", "prepare_data", context, tmp_path)
    else:
        assert checkpointing.save_checkpoint("experiment", "prepare_data", context, tmp_path) is None

    assert list(checkpoint_dir.iterdir()) == ([destination] if existing else [])
    if existing:
        assert destination.read_bytes() == previous
        assert checkpointing.load_checkpoint("prepare_data", tmp_path)["context"]["intent"] == "previous"


def test_successful_save_replaces_previous_checkpoint(tmp_path, context):
    """The public path and persisted feedback remain compatible across saves."""
    first = checkpointing.save_checkpoint("experiment", "prepare_data", context, tmp_path)
    assert first == tmp_path / "checkpoints" / "prepare_data.json"
    context.intent = "Updated intent"
    context.scratch["_user_feedback"] = {"message": "保留 • feedback"}
    second = checkpointing.save_checkpoint("experiment", "prepare_data", context, tmp_path)
    assert second == first
    result = checkpointing.load_checkpoint("prepare_data", tmp_path)
    assert result["context"]["intent"] == "Updated intent"
    assert result["user_feedback"] == context.scratch["_user_feedback"]
    assert json.loads(second.read_text())["version"] == "plexe-v1"
    assert list(second.parent.iterdir()) == [second]


def test_failed_replace_preserves_previous_checkpoint(tmp_path, context, monkeypatch):
    """An OS replacement error must clean the temporary file and preserve state."""
    destination = checkpointing.save_checkpoint("experiment", "prepare_data", context, tmp_path)
    previous = destination.read_bytes()

    def fail_replace(self, target):
        raise PermissionError("destination temporarily unavailable")

    monkeypatch.setattr(Path, "replace", fail_replace)
    context.intent = "Not published"
    assert checkpointing.save_checkpoint("experiment", "prepare_data", context, tmp_path) is None
    assert destination.read_bytes() == previous
    assert list(destination.parent.iterdir()) == [destination]
