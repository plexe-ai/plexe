"""Stage 2 integration tests: resume from seeds and pause after phase 4."""

from __future__ import annotations

import pytest

from plexe.constants import PhaseNames
from tests.integration.conftest import (
    MODEL_DATASET_KIND,
    MODEL_TYPE_PARAMS,
    assert_stage_prereqs,
    checkpoint_file,
    copy_seed_to_model_run,
    model_run_path,
    resume_workflow,
    seed_path,
)


@pytest.mark.integration_search
@pytest.mark.parametrize("model_type", MODEL_TYPE_PARAMS)
def test_resume_from_seed_and_run_search_only(model_type: str, artifact_root) -> None:
    """Copy a seed, resume from checkpoints, and pause after search models."""
    assert_stage_prereqs("search", artifact_root)

    dataset_kind = MODEL_DATASET_KIND[model_type]
    source_seed_dir = seed_path(artifact_root, dataset_kind)
    target_run_dir = model_run_path(artifact_root, model_type)

    copy_seed_to_model_run(source_seed_dir, target_run_dir)
    result = resume_workflow(
        work_dir=target_run_dir,
        allowed_model_types=[model_type],
        pause_points=[PhaseNames.SEARCH_MODELS],
        enable_final_evaluation=True,
        max_iterations=1,
    )

    assert result is None
    assert checkpoint_file(target_run_dir, PhaseNames.SEARCH_MODELS).exists()
    assert not checkpoint_file(target_run_dir, PhaseNames.EVALUATE_FINAL).exists()
    assert not checkpoint_file(target_run_dir, PhaseNames.PACKAGE_FINAL_MODEL).exists()
