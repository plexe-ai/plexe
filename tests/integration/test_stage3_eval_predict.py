"""Stage 3 integration tests: run evaluation/packaging and validate predictors."""

from __future__ import annotations

import pytest

from tests.integration.conftest import (
    MODEL_DATASET_KIND,
    MODEL_TYPE_PARAMS,
    assert_stage_prereqs,
    load_prediction_input,
    load_predictor_class,
    model_run_path,
    resume_workflow,
)


@pytest.mark.integration_eval
@pytest.mark.parametrize("model_type", MODEL_TYPE_PARAMS)
def test_resume_and_run_eval_then_predict(model_type: str, artifact_root, repo_root) -> None:
    """Resume from stage 2 checkpoints, run to completion, and validate predictor inference."""
    assert_stage_prereqs("eval", artifact_root)

    run_dir = model_run_path(artifact_root, model_type)
    result = resume_workflow(
        work_dir=run_dir,
        allowed_model_types=[model_type],
        pause_points=None,
        enable_final_evaluation=True,
        max_iterations=1,
    )

    assert result is not None
    best_solution, final_metrics, evaluation_report = result
    assert best_solution is not None
    assert isinstance(final_metrics, dict)
    assert evaluation_report is not None

    model_dir = run_dir / "model"
    assert model_dir.exists()
    assert (run_dir / "model.tar.gz").exists()
    assert (model_dir / "evaluation" / "reports" / "evaluation.json").exists()

    predictor_class = load_predictor_class(model_dir, model_type)
    predictor = predictor_class(str(model_dir))

    dataset_kind = MODEL_DATASET_KIND[model_type]
    prediction_input = load_prediction_input(repo_root, dataset_kind)
    predictions = predictor.predict(prediction_input)

    assert "prediction" in predictions.columns
    assert len(predictions) == len(prediction_input)
