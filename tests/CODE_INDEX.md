# Code Index: tests

> Generated on 2026-03-03 00:06:47

Test suite structure and test case documentation.

## `conftest.py`
Shared test fixtures for plexe tests.

**Functions:**
- `synthetic_parquet_classification(tmp_path)` - Create a 200-row binary classification parquet with 2 row groups.
- `synthetic_parquet_regression(tmp_path)` - Create a 200-row regression parquet with 2 row groups.

---
## `integration/conftest.py`
Shared fixtures and helpers for staged integration tests.

**Functions:**
- `repo_root() -> Path` - Return repository root path.
- `run_id() -> str` - Return deterministic run identifier for staged artifacts.
- `artifact_root(repo_root: Path, run_id: str) -> Path` - Return base path for staged integration artifacts.
- `configure_integration_environment(repo_root: Path) -> None` - Set environment variables needed by the integration suite.
- `cleanup_spark_session() -> None` - Stop Spark session after tests complete.
- `seed_path(artifact_root: Path, dataset_kind: str) -> Path` - Return seed directory path for a dataset kind.
- `model_run_path(artifact_root: Path, model_type: str) -> Path` - Return model-specific run directory path.
- `checkpoint_file(work_dir: Path, phase_name: str) -> Path` - Return path to a checkpoint file.
- `copy_seed_to_model_run(seed_dir: Path, model_dir: Path) -> None` - Copy a seed workdir into a model run workdir and rewrite checkpoint paths.
- `assert_stage_prereqs(stage: str, artifact_root: Path) -> None` - Assert required artifacts from prior stages exist.
- `build_seed_workflow(work_dir: Path, dataset_input: Path, intent: str, experiment_id: str) -> Any` - Run stages 1-3 and pause after baseline creation.
- `resume_workflow(work_dir: Path, allowed_model_types: list[str], pause_points: list[str] | None, enable_final_evaluation: bool, max_iterations: int) -> Any` - Resume a staged integration workflow from existing checkpoints.
- `load_predictor_class(model_dir: Path, model_type: str) -> type` - Load predictor class from packaged model/predictor.py.
- `load_prediction_input(repo_root: Path, dataset_kind: str, n_rows: int) -> pd.DataFrame` - Load a small feature sample used for predictor checks.

---
## `integration/test_stage1_seed.py`
Stage 1 integration tests: build reusable checkpoints through phase 3.

**Functions:**
- `test_build_seed_checkpoint(dataset_kind: str, artifact_root, repo_root) -> None` - Build a seed run and pause after baseline creation.

---
## `integration/test_stage2_search.py`
Stage 2 integration tests: resume from seeds and pause after phase 4.

**Functions:**
- `test_resume_from_seed_and_run_search_only(model_type: str, artifact_root) -> None` - Copy a seed, resume from checkpoints, and pause after search models.

---
## `integration/test_stage3_eval_predict.py`
Stage 3 integration tests: run evaluation/packaging and validate predictors.

**Functions:**
- `test_resume_and_run_eval_then_predict(model_type: str, artifact_root, repo_root) -> None` - Resume from stage 2 checkpoints, run to completion, and validate predictor inference.

---
## `unit/agents/test_feedback.py`
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
## `unit/execution/training/test_local_runner.py`
Tests for LocalProcessRunner GPU detection and command construction.

**`TestGPUDetection`** - Tests for framework GPU detection helpers.
- `test_no_torch(self)` - Returns 0 when torch is not importable.
- `test_no_cuda(self)` - Returns 0 when CUDA is not available.
- `test_with_cuda(self)` - Returns device count when CUDA is available.
- `test_tf_gpu_detection_no_tf(self)` - Returns 0 when tensorflow is not importable.

**`TestCommandConstruction`** - Test that the runner builds the right command for different GPU configurations.
- `setup_method(self)` - No description
- `test_pytorch_no_gpu_uses_python(self)` - PyTorch with 0 GPUs should use the current Python launcher, no GPU flags.
- `test_pytorch_single_gpu_no_ddp(self)` - PyTorch with 1 GPU should use current Python (no DDP), but get --mixed-precision.
- `test_pytorch_multi_gpu_uses_distributed_run(self)` - PyTorch with >1 GPU should use torch.distributed.run with --ddp and --mixed-precision.
- `test_pytorch_num_workers_passed(self)` - PyTorch should pass --num-workers when dataloader_workers > 0.
- `test_pytorch_no_mixed_precision_when_disabled(self)` - PyTorch with GPU but mixed_precision=False should not get --mixed-precision.

---
## `unit/search/test_evolutionary_policy_determinism.py`
Determinism tests for EvolutionarySearchPolicy local RNG behavior.

**Functions:**
- `test_evolutionary_policy_determinism(monkeypatch, tmp_path)` - No description

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
- `test_journal_get_history_includes_train_performance()` - get_history should include train_performance when set on a solution.
- `test_journal_get_history_train_performance_none()` - get_history should include train_performance=None when not set.

---
## `unit/search/test_tree_policy_determinism.py`
Determinism tests for TreeSearchPolicy local RNG behavior.

**Functions:**
- `test_tree_policy_determinism(monkeypatch, tmp_path)` - No description

---
## `unit/templates/features/test_pipeline_runner.py`
Unit tests for pipeline_runner feature name resolution.

**`NoFeatureNamesTransformer`** - Transformer without get_feature_names_out.
- `fit(self, x, y)` - No description
- `transform(self, x)` - No description

**`SelectFirstColumnTransformer`** - Transformer that reduces output to a single column.
- `fit(self, x, y)` - No description
- `transform(self, x)` - No description

**Functions:**
- `test_resolve_feature_names_uses_pipeline_minus_last()` - Falls back to pipeline[:-1] when last step lacks get_feature_names_out.
- `test_resolve_feature_names_falls_back_on_mismatch()` - Returns generic names when resolved names don't match output count.
- `test_resolve_feature_names_falls_back_when_unavailable()` - Returns generic names when no get_feature_names_out is available.

---
## `unit/templates/training/test_train_pytorch_worker_fallback.py`
Unit tests for PyTorch DataLoader worker fallback behavior.

**Functions:**
- `test_resolve_num_workers_zero_is_unchanged() -> None` - Requested zero workers should remain zero.
- `test_resolve_num_workers_falls_back_on_darwin_spawn(monkeypatch) -> None` - On macOS spawn, requested workers should fall back to zero.
- `test_resolve_num_workers_uses_context_when_start_method_is_none(monkeypatch) -> None` - When get_start_method returns None, context start method should be used.
- `test_resolve_num_workers_kept_on_non_darwin_spawn(monkeypatch) -> None` - Spawn on non-macOS should keep the requested worker count.

---
## `unit/test_config.py`
Unit tests for config helpers.

**Functions:**
- `test_get_routing_for_model_mapping_and_default()` - Mapped models use provider config; others use default.
- `test_temperature_fields_from_env(monkeypatch)` - No description
- `test_temperature_fields_from_yaml(tmp_path, monkeypatch)` - No description
- `test_get_temperature_resolves_override_and_default()` - No description
- `test_setup_logging_disables_propagation()` - Plexe logger should not propagate to root to avoid duplicate log lines.

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
- `test_solution_train_performance_defaults_to_none()` - New field should default to None for backward compatibility.
- `test_solution_to_dict_includes_train_performance()` - to_dict should serialize train_performance.
- `test_solution_from_dict_backward_compatible()` - Old checkpoints missing train_performance should deserialize cleanly.
- `test_solution_from_dict_with_train_performance()` - Checkpoints with train_performance should round-trip correctly.

---
## `unit/test_submission_pytorch.py`
Unit tests for PyTorch model submission.

**Functions:**
- `test_save_model_pytorch(tmp_path)` - Test PyTorch model submission validation and context scratch storage.

---
## `unit/utils/test_parquet_dataset.py`
Tests for streaming parquet data loading utilities.

**`TestMetadataUtilities`** - Tests for parquet metadata helper functions.
- `test_get_parquet_row_count(self, synthetic_parquet_classification)` - No description
- `test_get_dataset_size_bytes_file(self, synthetic_parquet_classification)` - No description
- `test_get_dataset_size_bytes_directory(self, tmp_path, synthetic_parquet_classification)` - No description
- `test_get_dataset_size_bytes_nonexistent(self)` - No description
- `test_get_parquet_feature_count(self, synthetic_parquet_classification)` - No description
- `test_get_steps_per_epoch(self, synthetic_parquet_classification)` - No description

**`TestParquetIterableDataset`** - Tests for streaming iterable dataset behavior.
- `test_yields_all_rows_classification(self, synthetic_parquet_classification)` - No description
- `test_yields_all_rows_regression(self, synthetic_parquet_regression)` - No description
- `test_yields_all_rows_binary(self, synthetic_parquet_classification)` - No description
- `test_directory_of_parquets(self, tmp_path, synthetic_parquet_classification)` - Test loading from a directory containing multiple parquet files.
- `test_total_rows_property(self, synthetic_parquet_classification)` - No description
- `test_ddp_sharding(self, synthetic_parquet_classification)` - Verify DDP sharding splits row groups across ranks.
- `test_feature_values_match_source(self, synthetic_parquet_classification)` - Verify streamed data matches the original parquet content.

**`TestParquetBatchGenerator`** - Tests for Keras/TensorFlow parquet batch generator.
- `test_yields_all_rows(self, synthetic_parquet_classification)` - No description
- `test_batch_size_respected(self, synthetic_parquet_classification)` - No description
- `test_directory_input(self, tmp_path, synthetic_parquet_classification)` - Test generator with directory of parquet files.
- `test_values_match_source(self, synthetic_parquet_classification)` - Verify batched data matches original parquet content.

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
## `unit/workflow/test_checkpoint_resume_feedback.py`
Tests for checkpoint resume feedback and persisted search journal behavior.

**Functions:**
- `test_search_models_preserves_user_feedback_for_all_variants(monkeypatch, tmp_path)` - No description
- `test_evaluate_final_checkpoint_persists_search_journal(monkeypatch, tmp_path)` - No description
- `test_package_final_checkpoint_persists_search_journal(monkeypatch, tmp_path)` - No description

---
## `unit/workflow/test_column_exclusion.py`
Tests for column exclusion pipeline.

**Functions:**
- `spark()` - No description
- `test_exclude_problematic_columns_drops_columns_and_returns_new_uri(spark, tmp_path)` - No description
- `test_exclude_problematic_columns_noop_when_empty(spark, tmp_path)` - No description
- `test_build_context_round_trip_with_excluded_columns(tmp_path)` - No description
- `test_exclude_problematic_columns_never_drops_target(spark, tmp_path)` - No description
- `test_exclude_problematic_columns_never_drops_primary_input(spark, tmp_path)` - No description

---
## `unit/workflow/test_model_card.py`
Unit tests for model card generation.

**Functions:**
- `test_generate_model_card_full_context(tmp_path: Path) -> None` - No description
- `test_generate_model_card_minimal_context(tmp_path: Path) -> None` - No description

---
## `unit/workflow/test_resume_model_type_filtering.py`
Tests for resume-time model type filtering.

**Functions:**
- `test_filters_checkpoint_model_types_on_resume(tmp_path)` - No description
- `test_uses_allowed_model_types_when_checkpoint_has_none(tmp_path)` - No description
- `test_raises_when_allowed_types_do_not_intersect_checkpoint(tmp_path)` - No description
- `test_does_not_filter_before_phase_one(tmp_path)` - No description

---