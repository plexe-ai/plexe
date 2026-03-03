"""Prompt-level tests for ModelEvaluatorAgent probability guidance."""

from __future__ import annotations

import inspect

from plexe.agents.model_evaluator import ModelEvaluatorAgent


def test_phase_1_prompt_includes_probability_metric_guidance():
    prompt = ModelEvaluatorAgent._get_phase_1_prompt("Predict churn", "roc_auc")

    assert "predict_proba" in prompt
    assert "roc_auc" in prompt
    assert "Binary: use positive-class scores" in prompt
    assert "Multiclass: use full per-class probability matrix" in prompt


def test_build_agent_instructions_document_predict_proba_interface():
    source = inspect.getsource(ModelEvaluatorAgent._build_agent)

    assert "predict_proba" in source
    assert "probability-based" in source
