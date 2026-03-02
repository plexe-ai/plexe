"""Stage 1 integration tests: build reusable checkpoints through phase 3."""

from __future__ import annotations

import shutil

import pytest

from plexe.constants import PhaseNames
from tests.integration.conftest import (
    DATASET_SPECS,
    REQUIRED_SEED_DATASET_KINDS,
    build_seed_workflow,
    checkpoint_file,
    seed_path,
)


@pytest.mark.integration_seed
@pytest.mark.parametrize("dataset_kind", REQUIRED_SEED_DATASET_KINDS)
def test_build_seed_checkpoint(dataset_kind: str, artifact_root, repo_root) -> None:
    """Build a seed run and pause after baseline creation."""
    dataset_spec = DATASET_SPECS[dataset_kind]
    seed_dir = seed_path(artifact_root, dataset_kind)

    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    result = build_seed_workflow(
        work_dir=seed_dir,
        dataset_input=repo_root / dataset_spec["dataset_relpath"],
        intent=dataset_spec["intent"],
        experiment_id=f"integration_seed_{dataset_kind}",
    )

    # build_model returns None when paused at a configured phase.
    assert result is None
    assert checkpoint_file(seed_dir, PhaseNames.ANALYZE_DATA).exists()
    assert checkpoint_file(seed_dir, PhaseNames.PREPARE_DATA).exists()
    assert checkpoint_file(seed_dir, PhaseNames.BUILD_BASELINES).exists()
    assert not checkpoint_file(seed_dir, PhaseNames.SEARCH_MODELS).exists()
