"""
Tests for ModelEvaluatorAgent probability phase gating.
"""

from __future__ import annotations

import pandas as pd

from plexe.agents.model_evaluator import ModelEvaluatorAgent
from plexe.config import Config
from plexe.models import BuildContext, Metric, Solution


def _make_context(tmp_path, task_type: str) -> BuildContext:
    context = BuildContext(
        user_id="user",
        experiment_id="exp",
        dataset_uri="file:///tmp/train.parquet",
        work_dir=tmp_path,
        intent="predict label",
    )
    context.metric = Metric(name="accuracy", optimization_direction="higher")
    context.output_targets = ["target"]
    context.task_analysis = {"task_type": task_type}
    return context


def _make_solution() -> Solution:
    return Solution(solution_id=1, feature_pipeline=object(), model=object(), model_type="xgboost")


def test_model_evaluator_runs_probability_phase_when_gate_passes(monkeypatch, tmp_path):
    context = _make_context(tmp_path, task_type="binary_classification")
    agent = ModelEvaluatorAgent(spark=None, context=context, config=Config())
    phases = []

    def fake_run_phase(self, phase_name, phase_prompt, tools, additional_args, registry_key):
        phases.append(phase_name)
        self.context.scratch[registry_key] = {"phase": phase_name}
        return True

    monkeypatch.setattr(ModelEvaluatorAgent, "_run_phase", fake_run_phase)

    class Predictor:
        def predict_proba(self, x):
            return pd.DataFrame({"proba_0": [0.8, 0.2], "proba_1": [0.2, 0.8]})

    test_df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})
    report = agent.run(solution=_make_solution(), test_sample_df=test_df, predictor=Predictor())

    assert report is not None
    assert "ProbabilityAnalysis" in phases


def test_model_evaluator_skips_probability_phase_when_task_not_classification(monkeypatch, tmp_path):
    context = _make_context(tmp_path, task_type="regression")
    agent = ModelEvaluatorAgent(spark=None, context=context, config=Config())
    phases = []

    def fake_run_phase(self, phase_name, phase_prompt, tools, additional_args, registry_key):
        phases.append(phase_name)
        self.context.scratch[registry_key] = {"phase": phase_name}
        return True

    monkeypatch.setattr(ModelEvaluatorAgent, "_run_phase", fake_run_phase)

    class Predictor:
        def predict_proba(self, x):
            return pd.DataFrame({"proba_0": [0.8, 0.2], "proba_1": [0.2, 0.8]})

    test_df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})
    report = agent.run(solution=_make_solution(), test_sample_df=test_df, predictor=Predictor())

    assert report is not None
    assert "ProbabilityAnalysis" not in phases
