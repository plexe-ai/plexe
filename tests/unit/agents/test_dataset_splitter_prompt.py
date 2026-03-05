"""Prompt-level tests for DatasetSplitterAgent split-mode instructions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("dataclasses_json")
pytest.importorskip("smolagents")

from plexe.agents.dataset_splitter import DatasetSplitterAgent
from plexe.models import BuildContext


def _make_context():
    context = BuildContext(
        user_id="user-1",
        experiment_id="exp-1",
        dataset_uri="train.parquet",
        work_dir="/tmp",
        intent="predict churn",
    )
    context.task_analysis = {"task_type": "binary_classification", "data_challenges": [], "recommended_split": {}}
    context.output_targets = ["target"]
    return context


def test_build_task_prompt_for_two_way_split_avoids_test_output():
    agent = DatasetSplitterAgent(
        spark=object(),
        dataset_uri="train.parquet",
        context=_make_context(),
        config=SimpleNamespace(dataset_splitting_llm="test-model"),
    )

    prompt = agent._build_task_prompt({"train": 0.85, "val": 0.15}, "/tmp/splits")

    assert "train/validation sets" in prompt
    assert "save_split_uris(train_path, val_path) with NO test_path" in prompt
    assert "Do NOT create or submit a test split in this run" in prompt


def test_build_task_prompt_for_three_way_split_requires_test_output():
    agent = DatasetSplitterAgent(
        spark=object(),
        dataset_uri="train.parquet",
        context=_make_context(),
        config=SimpleNamespace(dataset_splitting_llm="test-model"),
    )

    prompt = agent._build_task_prompt({"train": 0.7, "val": 0.15, "test": 0.15}, "/tmp/splits")

    assert "train/validation/test sets" in prompt
    assert "save_split_uris(train_path, val_path, test_path)" in prompt
    assert "Do NOT create or submit a test split in this run" not in prompt
