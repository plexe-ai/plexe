# Code Index: tests

> Generated on 2026-02-26 13:49:04

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
## `unit/search/test_insight_store.py`
Unit tests for InsightStore.

**Functions:**
- `test_insight_store_add_update_serialize_roundtrip()` - Add/update and serialize/deserialize should preserve insights.

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
## `unit/test_config.py`
Unit tests for config helpers.

**Functions:**
- `test_get_routing_for_model_mapping_and_default()` - Mapped models use provider config; others use default.

---
## `unit/test_helpers.py`
Unit tests for workflow helper functions.

**Functions:**
- `test_compute_metric_accuracy()` - Test accuracy computation.
- `test_compute_metric_rmse()` - Test RMSE computation.
- `test_compute_metric_f1_score()` - Test F1 score computation.
- `test_compute_metric_unknown_raises()` - Test unknown metric raises ValueError.
- `test_select_viable_model_types_defaults_image()` - Default model types intersect with IMAGE_PATH.
- `test_select_viable_model_types_no_intersection()` - No compatible frameworks should raise ValueError.
- `test_compute_metric_map_grouped()` - MAP should compute per-group and average.

---
## `unit/test_imports.py`
Test that all production modules can be imported without errors.

**Functions:**
- `test_all_modules_importable()` - Import all production modules in the plexe/ package to catch import errors.

---
## `unit/test_lightgbm_predictor.py`
Unit tests for LightGBM predictor template.

**`DummyModel`** - Minimal model stub with a predict method.
- `predict(self, x)` - No description

**`DummyPipeline`** - Minimal pipeline stub with a transform method.
- `transform(self, x)` - No description

**Functions:**
- `test_lightgbm_predictor_basic(tmp_path: Path) -> None` - No description
- `test_lightgbm_predictor_label_encoder(tmp_path: Path) -> None` - No description

---
## `unit/test_models.py`
Unit tests for core model dataclasses.

**Functions:**
- `test_build_context_update_and_unknown_key()` - Update should set known fields and reject unknown keys.

---
## `unit/test_submission_pytorch.py`
Unit tests for PyTorch model submission.

**Functions:**
- `test_save_model_pytorch(tmp_path)` - Test PyTorch model submission validation and context scratch storage.

---
## `unit/utils/test_reporting.py`
Unit tests for reporting utilities.

**Functions:**
- `test_save_report_converts_numpy_types(tmp_path)` - save_report should serialize numpy types to native Python values.

---
## `unit/utils/test_tooling.py`
Unit tests for tooling utilities.

**Functions:**
- `greet(name: str, times: int)` - greet(name: str, times: int = 1) -> str
- `test_agentinspectable_valid_call()` - Decorator should allow valid calls.
- `test_agentinspectable_invalid_call_raises()` - Invalid calls should raise AgentInvocationError with usage.

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
- `test_validate_metric_function_object_success()` - Callable with correct signature should pass.
- `test_validate_metric_function_object_bad_signature()` - Callable with wrong arg names should fail.

---