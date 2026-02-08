# Code Index: tests

> Generated on 2026-02-08 21:45:49

Test suite structure and test case documentation.

## `integration/test_feedback.py`
Tests for user feedback integration in agents.

**`TestFeedbackFormatting`** - Test the format_user_feedback_for_prompt helper function.
- `test_none_feedback_returns_empty(self)` - No feedback should return empty string.
- `test_empty_dict_returns_empty(self)` - Empty dict should return empty string.
- `test_string_feedback(self)` - String feedback should be formatted with header.
- `test_dict_with_comments(self)` - Dict with comments field should extract text.
- `test_dict_with_feedback_field(self)` - Dict with feedback field (legacy) should extract text.
- `test_dict_with_requested_changes(self)` - Dict with requested_changes list should format as bullets.
- `test_empty_comments_returns_empty(self)` - Dict with empty comments should return empty string.

**`TestAgentFeedbackIntegration`** - Test that agents properly integrate feedback into their prompts.
- `mock_context(self)` - Create a mock BuildContext for testing.
- `mock_config(self)` - Create a mock Config for testing.
- `test_statistical_analyser_includes_feedback(self, mock_context, mock_config)` - StatisticalAnalyserAgent should include feedback in instructions.
- `test_ml_task_analyser_includes_feedback(self, mock_context, mock_config)` - MLTaskAnalyserAgent should include feedback in instructions.
- `test_metric_selector_includes_feedback(self, mock_context, mock_config)` - MetricSelectorAgent should include feedback in instructions.
- `test_baseline_builder_includes_feedback(self, mock_context, mock_config)` - BaselineBuilderAgent should include feedback in instructions.
- `test_hypothesiser_includes_feedback(self, mock_context, mock_config)` - HypothesiserAgent should include feedback in instructions.
- `test_agent_without_feedback_works(self, mock_context, mock_config)` - Agents should work normally when no feedback is provided.

---
## `unit/search/test_journal.py`
Unit tests for SearchJournal.

**Functions:**
- `test_journal_initialization()` - Test journal initializes correctly.
- `test_journal_initialization_no_baseline()` - Test journal initializes without baseline.
- `test_journal_add_successful_node()` - Test recording a successful solution.
- `test_journal_add_buggy_node()` - Test recording a failed attempt.
- `test_journal_best_node_tracks_best()` - Test best_node returns the highest performing solution.
- `test_journal_failure_rate()` - Test failure rate computation.
- `test_journal_failure_rate_empty()` - Test failure rate on empty journal.
- `test_journal_get_history()` - Test history returns recent entries.
- `test_journal_improvement_trend_improving()` - Test improvement trend with steadily improving solutions.
- `test_journal_improvement_trend_insufficient_data()` - Test improvement trend with fewer than 2 successful solutions.

---
## `unit/test_helpers.py`
Unit tests for workflow helper functions.

**Functions:**
- `test_compute_metric_accuracy()` - Test accuracy computation.
- `test_compute_metric_rmse()` - Test RMSE computation.
- `test_compute_metric_f1_score()` - Test F1 score computation.
- `test_compute_metric_unknown_raises()` - Test unknown metric raises ValueError.

---
## `unit/test_imports.py`
Test that all production modules can be imported without errors.

**Functions:**
- `test_all_modules_importable()` - Import all production modules in the plexe/ package to catch import errors.

---
## `unit/validation/test_validators.py`
Unit tests for validation functions.

**Functions:**
- `test_validate_sklearn_pipeline_success()` - Test valid pipeline passes validation.
- `test_validate_sklearn_pipeline_wrong_type()` - Test non-pipeline fails validation.
- `test_validate_xgboost_params_success()` - Test valid XGBoost params pass validation.
- `test_validate_xgboost_params_invalid_max_depth()` - Test invalid max_depth fails validation.
- `test_validate_xgboost_params_invalid_type()` - Test non-dict fails validation.
- `test_validate_model_definition_xgboost()` - Test XGBoost model definition validation.
- `test_validate_model_definition_unknown_type()` - Test unknown model type fails validation.

---