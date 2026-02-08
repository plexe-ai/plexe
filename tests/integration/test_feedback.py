"""
Tests for user feedback integration in agents.

Verifies that agents properly receive and incorporate user feedback into their prompts.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from plexe.agents.utils import format_user_feedback_for_prompt
from plexe.models import BuildContext, Metric
from plexe.config import Config


class TestFeedbackFormatting:
    """Test the format_user_feedback_for_prompt helper function."""

    def test_none_feedback_returns_empty(self):
        """No feedback should return empty string."""
        result = format_user_feedback_for_prompt(None)
        assert result == ""

    def test_empty_dict_returns_empty(self):
        """Empty dict should return empty string."""
        result = format_user_feedback_for_prompt({})
        assert result == ""

    def test_string_feedback(self):
        """String feedback should be formatted with header."""
        feedback = "Focus on recency features"
        result = format_user_feedback_for_prompt(feedback)

        assert "USER FEEDBACK" in result
        assert "Focus on recency features" in result
        assert "IMPORTANT" in result

    def test_dict_with_comments(self):
        """Dict with comments field should extract text."""
        feedback = {
            "approved": True,
            "comments": "Try neural networks instead of tree models",
            "requested_changes": [],
            "enable_auto_mode": False,
        }
        result = format_user_feedback_for_prompt(feedback)

        assert "USER FEEDBACK" in result
        assert "Try neural networks instead of tree models" in result

    def test_dict_with_feedback_field(self):
        """Dict with feedback field (legacy) should extract text."""
        feedback = {"feedback": "Ignore demographic features"}
        result = format_user_feedback_for_prompt(feedback)

        assert "USER FEEDBACK" in result
        assert "Ignore demographic features" in result

    def test_dict_with_requested_changes(self):
        """Dict with requested_changes list should format as bullets."""
        feedback = {
            "comments": "Main guidance",
            "requested_changes": ["Change A", "Change B"],
        }
        result = format_user_feedback_for_prompt(feedback)

        assert "Main guidance" in result
        assert "Requested changes:" in result
        assert "Change A" in result
        assert "Change B" in result

    def test_empty_comments_returns_empty(self):
        """Dict with empty comments should return empty string."""
        feedback = {"comments": "", "approved": True}
        result = format_user_feedback_for_prompt(feedback)

        assert result == ""


class TestAgentFeedbackIntegration:
    """Test that agents properly integrate feedback into their prompts."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock BuildContext for testing."""
        context = BuildContext(
            user_id="test_user",
            experiment_id="test_exp",
            dataset_uri="/fake/data.parquet",
            work_dir=Path("/tmp/test"),
            intent="predict customer churn",
        )
        context.stats = {"num_rows": 1000, "num_columns": 10}
        context.task_analysis = {"task_type": "binary_classification", "num_classes": 2}
        context.metric = Metric(name="f1_score", optimization_direction="higher")
        context.output_targets = ["churned"]
        return context

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config for testing."""
        return Config()

    def test_statistical_analyser_includes_feedback(self, mock_context, mock_config):
        """StatisticalAnalyserAgent should include feedback in instructions."""
        from plexe.agents.statistical_analyser import StatisticalAnalyserAgent

        # Add feedback to context
        mock_context.scratch["_user_feedback"] = {"comments": "Focus on temporal patterns in the data"}

        # Create agent
        agent = StatisticalAnalyserAgent(
            spark=MagicMock(), dataset_uri="/fake/data.parquet", context=mock_context, config=mock_config
        )

        # Build the agent and check instructions
        code_agent = agent._build_agent()
        instructions = code_agent.system_prompt

        assert "USER FEEDBACK" in instructions
        assert "temporal patterns" in instructions
        assert "IMPORTANT" in instructions

    def test_ml_task_analyser_includes_feedback(self, mock_context, mock_config):
        """MLTaskAnalyserAgent should include feedback in instructions."""
        from plexe.agents.ml_task_analyser import MLTaskAnalyserAgent

        mock_context.scratch["_user_feedback"] = {"comments": "This is a time-series problem, treat it accordingly"}

        agent = MLTaskAnalyserAgent(
            spark=MagicMock(),
            dataset_uri="/fake/data.parquet",
            stats_report={},
            context=mock_context,
            config=mock_config,
        )

        code_agent = agent._build_agent()
        instructions = code_agent.system_prompt

        assert "USER FEEDBACK" in instructions
        assert "time-series problem" in instructions

    def test_metric_selector_includes_feedback(self, mock_context, mock_config):
        """MetricSelectorAgent should include feedback in instructions."""
        from plexe.agents.metric_selector import MetricSelectorAgent

        mock_context.scratch["_user_feedback"] = {"comments": "Use ROC-AUC instead of accuracy"}

        agent = MetricSelectorAgent(context=mock_context, config=mock_config)

        code_agent = agent._build_agent()
        instructions = code_agent.system_prompt

        assert "USER FEEDBACK" in instructions
        assert "ROC-AUC" in instructions

    def test_baseline_builder_includes_feedback(self, mock_context, mock_config):
        """BaselineBuilderAgent should include feedback in instructions."""
        from plexe.agents.baseline_builder import BaselineBuilderAgent

        mock_context.scratch["_user_feedback"] = {"comments": "Use mode baseline, not mean"}
        mock_context.train_sample_uri = "/fake/train.parquet"
        mock_context.val_sample_uri = "/fake/val.parquet"

        agent = BaselineBuilderAgent(spark=MagicMock(), context=mock_context, config=mock_config)

        # Need to pass val_sample_df to _build_agent
        val_df = MagicMock()
        code_agent = agent._build_agent(val_df)
        instructions = code_agent.system_prompt

        assert "USER FEEDBACK" in instructions
        assert "mode baseline" in instructions

    def test_hypothesiser_includes_feedback(self, mock_context, mock_config):
        """HypothesiserAgent should include feedback in instructions."""
        from plexe.agents.hypothesiser import HypothesiserAgent
        from plexe.search.journal import SearchJournal

        mock_context.scratch["_user_feedback"] = {"comments": "Try deep learning models, avoid XGBoost"}

        journal = SearchJournal(baseline=MagicMock(performance=0.7))
        agent = HypothesiserAgent(journal=journal, context=mock_context, config=mock_config, expand_solution_id=0)

        code_agent = agent._build_agent()
        instructions = code_agent.system_prompt

        assert "USER FEEDBACK" in instructions
        assert "deep learning" in instructions
        assert "avoid XGBoost" in instructions

    def test_agent_without_feedback_works(self, mock_context, mock_config):
        """Agents should work normally when no feedback is provided."""
        from plexe.agents.statistical_analyser import StatisticalAnalyserAgent

        # No feedback in context
        assert "_user_feedback" not in mock_context.scratch

        agent = StatisticalAnalyserAgent(
            spark=MagicMock(), dataset_uri="/fake/data.parquet", context=mock_context, config=mock_config
        )

        code_agent = agent._build_agent()
        instructions = code_agent.system_prompt

        # Should not contain feedback section
        assert "USER FEEDBACK" not in instructions
        # But should still work
        assert "YOUR ROLE" in instructions


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
