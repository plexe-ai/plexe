# Code Index: plexe

> Generated on 2026-02-26 19:02:04

Code structure and public interface documentation for the **plexe** package.

## `agents/baseline_builder.py`
Baseline Builder Agent.

**`BaselineBuilderAgent`** - Agent that creates simple baseline predictors.
- `__init__(self, spark: SparkSession, context: BuildContext, config: Config)`
- `run(self) -> Baseline` - Build baseline predictor using agent.

---
## `agents/dataset_splitter.py`
Dataset Splitter Agent.

**`DatasetSplitterAgent`** - Agent that generates PySpark code for intelligent dataset splitting.
- `__init__(self, spark: SparkSession, dataset_uri: str, context: BuildContext, config: Config)`
- `run(self, split_ratios: dict[str, float], output_dir: str | Path) -> tuple[str, str, str]` - Generate and execute intelligent dataset splitting.

---
## `agents/feature_processor.py`
Feature Processor Agent.

**`FeatureProcessorAgent`** - Agent that designs sklearn Pipeline for feature engineering.
- `__init__(self, spark: SparkSession, train_uri: str, context: BuildContext, config: Config, plan: Any)`
- `run(self) -> tuple[Pipeline, str]` - Design feature engineering pipeline.

---
## `agents/hypothesiser.py`
Hypothesiser Agent.

**`HypothesiserAgent`** - Agent that generates hypotheses for next search direction.
- `__init__(self, journal: SearchJournal, context: BuildContext, config: Config, expand_solution_id: int)`
- `run(self) -> Hypothesis` - Generate hypothesis for next exploration.

---
## `agents/insight_extractor.py`
Insight Extractor Agent.

**`InsightExtractorAgent`** - Agent that extracts structured insights from experiment results.
- `__init__(self, hypothesis: Hypothesis, variant_solutions: list[Solution], insight_store, context: BuildContext, config: Config)`
- `run(self) -> int` - Extract insights from variant results.

---
## `agents/layout_detector.py`
Layout Detection Agent.

**`LayoutDetectionAgent`** - Agent that detects data layout and identifies primary input column.
- `__init__(self, spark: SparkSession, dataset_uri: str, context: BuildContext, config: Config)`
- `run(self) -> dict` - Run layout detection.

---
## `agents/metric_implementer.py`
Metric Implementation Agent.

**`MetricImplementationAgent`** - Agent that generates metric computation function code.
- `__init__(self, context: BuildContext, config: Config)`
- `run(self) -> Any` - Generate metric computation function.

---
## `agents/metric_selector.py`
Metric Selector Agent.

**`MetricSelectorAgent`** - Agent that selects evaluation metric using smolagents with structured submission tool.
- `__init__(self, context: BuildContext, config: Config)`
- `run(self) -> Metric` - Run metric selection with structured submission.

---
## `agents/ml_task_analyser.py`
ML Task Analyser Agent.

**`MLTaskAnalyserAgent`** - Agent that analyzes dataset from ML perspective.
- `__init__(self, spark: SparkSession, dataset_uri: str, stats_report: dict, context: BuildContext, config: Config)`
- `run(self) -> dict` - Run ML task analysis.

---
## `agents/model_definer.py`
Model Definer Agent.

**`ModelDefinerAgent`** - Agent that defines model configuration.
- `__init__(self, model_type: str, context: BuildContext, config: Config, transformed_schema: dict, plan: Any)`
- `run(self) -> tuple[Any, str]` - Create untrained model object from plan.

---
## `agents/model_evaluator.py`
Model Evaluator Agent for comprehensive ML model evaluation.

**`ModelEvaluatorAgent`** - Agent that performs comprehensive model evaluation through structured phases.
- `__init__(self, spark: SparkSession, context: BuildContext, config: Config)`
- `run(self, solution: Solution, test_sample_df: pd.DataFrame, predictor: Any) -> EvaluationReport | None` - Execute multi-phase evaluation.

---
## `agents/planner.py`
Planner Agent.

**`PlannerAgent`** - Agent that creates concrete plan specifications from hypotheses.
- `__init__(self, journal: SearchJournal, context: BuildContext, config: Config, hypothesis: Hypothesis | None, num_bootstrap: int)`
- `run(self) -> list[UnifiedPlan]` - Generate concrete plan specifications.

---
## `agents/sampler.py`
Sampling Agent.

**`SamplingAgent`** - Agent that generates PySpark code for intelligent dataset sampling.
- `__init__(self, spark: SparkSession, context: BuildContext, config: Config)`
- `run(self, train_uri: str, val_uri: str, train_sample_size: int, val_sample_size: int, output_dir: Path) -> tuple[str, str]` - Generate and execute intelligent sampling for both train and val datasets.

---
## `agents/statistical_analyser.py`
Statistical Analyser Agent.

**`StatisticalAnalyserAgent`** - Agent that performs statistical profiling of datasets.
- `__init__(self, spark: SparkSession, dataset_uri: str, context: BuildContext, config: Config)`
- `run(self) -> dict` - Run statistical analysis.

---
## `agents/utils.py`
Shared utilities for agents.

**Functions:**
- `format_user_feedback_for_prompt(user_feedback: dict | str | None) -> str` - Format user feedback for inclusion in agent prompts.

---
## `checkpointing.py`
Checkpointing functionality for plexe.

**Functions:**
- `pickle_to_base64(obj) -> str` - Serialize object to base64-encoded pickle string.
- `base64_to_pickle(b64_string: str)` - Deserialize base64-encoded pickle string to object.
- `save_checkpoint(experiment_id: str, phase_name: str, context: BuildContext, work_dir: Path, search_journal: SearchJournal | None, insight_store: InsightStore | None) -> Path | None` - Save checkpoint to local disk only.
- `load_checkpoint(phase_name: str, work_dir: Path) -> dict | None` - Load checkpoint from local disk.

---
## `config.py`
Configuration for plexe.

**`ModelType`** - Supported model types (architectural decision).

**`StandardMetric`** - Standard metrics with hardcoded implementations.

**`YamlConfigSettingsSource`** - Custom settings source that loads config from YAML file specified by CONFIG_FILE env var.
- `get_field_value(self, field, field_name)` - Not used in Pydantic v2 - use __call__ instead.

**`RoutingProviderConfig`** - Configuration for a single routing provider.

**`RoutingConfig`** - LiteLLM routing configuration for custom API endpoints.
- `validate_model_providers(cls, v: dict[str, str], info) -> dict[str, str]` - Validate that all model provider references exist.

**`Config`** - Configuration for model building workflow.
- `settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings)` - Customize settings source priority.
- `validate_nn_training_settings(self) -> 'Config'` - Ensure neural network defaults do not exceed the configured cap.
- `parse_otel_headers_from_env(self) -> 'Config'` - Parse OTEL_EXPORTER_OTLP_HEADERS (comma-separated key=value pairs).

**Functions:**
- `get_routing_for_model(config: RoutingConfig | None, model_id: str) -> tuple[str | None, dict[str, str]]` - Get routing configuration for a specific model ID.
- `setup_logging(config: Config) -> logging.Logger` - Configure logging for the plexe package.
- `setup_litellm(config: Config) -> None` - Configure LiteLLM global settings.
- `get_config() -> Config` - Get configuration from YAML file (if specified) with environment variable overrides.

---
## `constants.py`
Constants for plexe.

**`ScratchKeys`** - Keys for BuildContext.scratch dictionary.

**`DatasetPatterns`** - Naming patterns for dataset artifacts.
- `transformed_name(base_uri: str, iteration: int) -> str` - Generate name for transformed dataset.

**`SearchDefaults`** - Default values for search configuration.

**`DirNames`** - Standard directory and file names used across the codebase.

**`PhaseNames`** - Standardized phase names for workflow orchestration.

---
## `execution/dataproc/dataset_io.py`
Dataset I/O with format detection and normalization.

**`DatasetFormat`** - Supported dataset formats.

**`FormatDetector`** - Detects dataset format from URI.
- `detect(uri: str) -> DatasetFormat` - Detect format from URI.

**`DatasetReader`** - Reads datasets in any supported format using Spark.
- `__init__(self, spark: SparkSession)`
- `read(self, uri: str, format: DatasetFormat, options: dict | None) -> DataFrame` - Read dataset in specified format.

**`DatasetNormalizer`** - Normalizes datasets to Parquet format.
- `__init__(self, spark: SparkSession)`
- `normalize(self, input_uri: str, output_uri: str, format_hint: DatasetFormat | None, read_options: dict | None) -> tuple[str, DatasetFormat]` - Normalize dataset to Parquet format.

---
## `execution/dataproc/session.py`
Spark session management with singleton pattern.

**Functions:**
- `get_or_create_spark_session(config) -> SparkSession` - Get or create Spark session based on config.
- `stop_spark_session()` - Stop and cleanup Spark session.

---
## `execution/training/local_runner.py`
Local process runner - executes training in subprocess.

**`LocalProcessRunner`** - Runs training in local subprocess.
- `__init__(self, work_dir: str)`
- `run_training(self, template: str, model: Any, feature_pipeline: Pipeline, train_uri: str, val_uri: str, timeout: int, target_columns: list[str], optimizer: Any, loss: Any, epochs: int, batch_size: int, group_column: str | None) -> Path` - Execute training in subprocess.

---
## `execution/training/runner.py`
Training runner abstract base class.

**`TrainingRunner`** - Abstract base class for training execution environments.
- `run_training(self, template: str, model: Any, feature_pipeline: Pipeline, train_uri: str, val_uri: str, timeout: int, target_columns: list[str]) -> Path` - Execute model training and return path to artifacts.

---
## `helpers.py`
Helper functions for workflow.

**Functions:**
- `select_viable_model_types(data_layout: DataLayout, selected_frameworks: list[str] | None) -> list[str]` - Select viable model types using three-tier filtering.
- `evaluate_on_sample(spark: SparkSession, sample_uri: str, model_artifacts_path: Path, model_type: str, metric: str, target_columns: list[str], group_column: str | None) -> float` - Evaluate model on sample (fast).
- `compute_metric_hardcoded(y_true, y_pred, metric_name: str) -> float` - Compute metric using hardcoded sklearn implementations.
- `compute_metric(y_true, y_pred, metric_name: str, group_ids) -> float` - Compute metric value.

---
## `integrations/base.py`
Base integration interface for connecting plexe to external infrastructure.

**`WorkflowIntegration`** - Integration interface for environment-specific infrastructure.
- `prepare_workspace(self, experiment_id: str, work_dir: Path) -> None` - Prepare workspace for a model-building run.
- `get_artifact_location(self, artifact_type: str, dataset_uri: str, experiment_id: str, work_dir: Path) -> str` - Determine where an intermediate artifact should be written.
- `ensure_local(self, uris: list[str], work_dir: Path) -> list[str]` - Ensure remote URIs are available on the local filesystem.
- `prepare_original_model(self, model_reference: str, work_dir: Path) -> str` - Locate and download an existing model for retraining.
- `on_checkpoint(self, experiment_id: str, phase_name: str, checkpoint_path: Path, work_dir: Path) -> None` - Persist checkpoint and work directory after a phase completes.
- `on_completion(self, experiment_id: str, work_dir: Path, final_metrics: dict, evaluation_report: Any) -> None` - Persist final model and update tracking on successful completion.
- `on_failure(self, experiment_id: str, error: Exception) -> None` - Handle workflow failure.
- `on_pause(self, phase_name: str) -> None` - Handle workflow pause for user feedback.

---
## `integrations/standalone.py`
Standalone integration for local development and S3-backed deployments.

**`StandaloneIntegration`** - Standalone integration for local development and testing.
- `__init__(self, external_storage_uri: str | None, user_id: str | None)`
- `prepare_workspace(self, experiment_id: str, work_dir: Path) -> None` - Restore workspace from S3 if a previous run exists.
- `get_artifact_location(self, artifact_type: str, dataset_uri: str, experiment_id: str, work_dir: Path) -> str` - Determine storage location based on dataset location.
- `ensure_local(self, uris: list[str], work_dir: Path) -> list[str]` - Download S3 URIs to local if needed (handles Spark parquet directories).
- `prepare_original_model(self, model_reference: str, work_dir: Path) -> str` - Prepare original model for retraining.
- `on_checkpoint(self, experiment_id: str, phase_name: str, checkpoint_path: Path, work_dir: Path) -> None` - Upload checkpoint and workdir to S3 if configured.
- `on_completion(self, experiment_id: str, work_dir: Path, final_metrics: dict, evaluation_report: Any) -> None` - Upload final model to S3 if configured.

---
## `integrations/storage/__init__.py`
Storage helper interface and implementations.

**`StorageHelper`** - Abstract base for cloud/remote storage helpers.
- `upload_file(self, local_path: Path, key: str) -> str` - Upload a local file to remote storage.
- `download_file(self, key: str, local_path: Path) -> None` - Download a remote file to local filesystem.
- `object_exists(self, key: str) -> bool` - Check whether an object exists in remote storage.
- `download_directory(self, uri: str, local_dir: Path) -> str` - Download a remote directory (e.g., Spark parquet output) to local filesystem.

---
## `integrations/storage/azure.py`
Azure Blob Storage helper (stub).

**`AzureBlobHelper`** - Azure Blob Storage helper (not yet implemented).
- `upload_file(self, local_path: Path, key: str) -> str` - No description
- `download_file(self, key: str, local_path: Path) -> None` - No description
- `object_exists(self, key: str) -> bool` - No description
- `download_directory(self, uri: str, local_dir: Path) -> str` - No description

---
## `integrations/storage/gcs.py`
Google Cloud Storage helper (stub).

**`GCSHelper`** - Google Cloud Storage helper (not yet implemented).
- `upload_file(self, local_path: Path, key: str) -> str` - No description
- `download_file(self, key: str, local_path: Path) -> None` - No description
- `object_exists(self, key: str) -> bool` - No description
- `download_directory(self, uri: str, local_dir: Path) -> str` - No description

---
## `integrations/storage/s3.py`
Amazon S3 storage helper.

**`S3Helper`** - Amazon S3 storage helper.
- `__init__(self, bucket: str, prefix: str, user_id: str | None)`
- `parse_uri(uri: str) -> tuple[str, str]` - Parse s3://bucket/prefix into (bucket, prefix).
- `build_key(self, experiment_id: str) -> str` - Build S3 key with user/experiment scoping.
- `upload_file(self, local_path: Path, key: str) -> str` - Upload file to S3.
- `download_file(self, key: str, local_path: Path) -> None` - Download file from S3.
- `object_exists(self, key: str) -> bool` - Check if S3 object exists.
- `download_directory(self, uri: str, local_dir: Path) -> str` - Download a Spark parquet directory (multiple part files) from S3.
- `tar_and_upload(self, local_dir: Path, s3_key: str) -> str` - Create tarball from directory and upload to S3.
- `download_and_extract_tar(self, s3_key: str, extract_to: Path) -> None` - Download tarball from S3 and extract.
- `handle_download_error(self, error: ClientError, reference: str, context: str) -> None` - Raise appropriate exception for S3 download errors.

---
## `main.py`
Universal entry point for plexe.

**Functions:**
- `main(intent: str, data_refs: list[str], integration: WorkflowIntegration | None, spark_mode: str, user_id: str, experiment_id: str, max_iterations: int, work_dir: Path, test_dataset_uri: str | None, enable_final_evaluation: bool, max_epochs: int | None, allowed_model_types: list[str] | None, is_retrain: bool, original_model_uri: str | None, original_experiment_id: str | None, auto_mode: bool, user_feedback: dict | None, enable_otel: bool, otel_endpoint: str | None, otel_headers: dict[str, str] | None, external_storage_uri: str | None, csv_delimiter: str, csv_header: bool)` - Main model building function.

---
## `models.py`
Simple dataclasses for model building workflow.

**`DataLayout`** - Physical structure of dataset (not semantic meaning).

**`Metric`** - Evaluation metric definition.

**`BuildContext`** - Context passed through workflow phases.
- `add_outer_loop_feedback(self, solution: Optional['Solution'], issue: str)` - Add feedback for outer loop retry.
- `update(self)` - Convenience method to update multiple fields.
- `to_dict(self) -> dict` - Serialize BuildContext to dict for checkpointing.
- `from_dict(d: dict) -> 'BuildContext'` - Deserialize BuildContext from checkpoint dict.

**`Baseline`** - Represents a baseline model result.

**`Solution`** - Represents a solution in the search tree.
- `is_leaf(self) -> bool` - Check if this is a leaf node in the search tree.
- `debug_depth(self) -> int` - Number of consecutive buggy ancestors in lineage.
- `is_successful(self) -> bool` - Check if execution succeeded.
- `to_dict(self) -> dict` - Serialize solution to dict for checkpointing.
- `from_dict(d: dict, all_solutions: dict[int, 'Solution']) -> 'Solution'` - Deserialize solution from checkpoint dict.

**`Insight`** - Structured learning extracted from search experiments.

**`Hypothesis`** - Strategic direction for next exploration.

**`FeaturePlan`** - Feature engineering specification.

**`ModelPlan`** - Model configuration specification (natural language directive).

**`UnifiedPlan`** - Complete solution specification (features + model).

**`CoreMetricsReport`** - Core performance metrics on test set.

**`DiagnosticReport`** - Error analysis and failure pattern detection.

**`RobustnessReport`** - Model reliability under stress conditions.

**`ExplainabilityReport`** - Feature importance and model interpretability.

**`BaselineComparisonReport`** - Model vs. baseline performance comparison.

**`EvaluationReport`** - Final comprehensive evaluation with verdict and recommendations.

**`TrainingError`** - Raised when training fails.

**`ValidationError`** - Raised when validation fails.

---
## `retrain.py`
Model retraining functionality.

**`RetrainingError`** - Raised when retraining fails.

**Functions:**
- `retrain_model(original_model_uri: str, train_dataset_uri: str, experiment_id: str, work_dir: Path, runner, config, on_checkpoint_saved) -> tuple[Solution, dict]` - Retrain existing model with new data using original training pipeline.

---
## `search/evolutionary_search_policy.py`
PiEvolve-inspired evolutionary search policy with adaptive state analysis.

**`EvolutionarySearchPolicy`** - PiEvolve-inspired probabilistic action selection with adaptive search state analysis.
- `__init__(self, num_drafts: int, debug_prob: float, max_debug_depth: int)`
- `decide_next_solution(self, journal: SearchJournal, context: BuildContext, iteration: int, max_iterations: int) -> Solution | None` - PiEvolve-style probabilistic action selection based on search state.
- `should_stop(self, journal: SearchJournal, iteration: int, max_iterations: int) -> bool` - Enhanced stopping criteria with intelligent early stopping.

---
## `search/insight_store.py`
Insight store for accumulating learnings from search.

**`InsightStore`** - Simple store for insights extracted from experiments.
- `__init__(self)`
- `add(self, change: str, effect: str, context: str, confidence: str, supporting_evidence: list[int]) -> Insight` - Add new insight.
- `update(self, insight_id: int) -> bool` - Update insight fields.
- `get_all(self) -> list[Insight]` - Get all insights.
- `to_dict(self) -> dict` - Serialize InsightStore to dict for checkpointing.
- `from_dict(d: dict) -> 'InsightStore'` - Deserialize InsightStore from checkpoint dict.

---
## `search/journal.py`
Search journal for tracking model search tree.

**`SearchJournal`** - Tracks solution search tree.
- `__init__(self, baseline: Baseline | None)`
- `add_node(self, node: Solution) -> None` - Add a solution to the journal.
- `draft_nodes(self) -> list[Solution]` - Get all root nodes (bootstrap solutions without parents).
- `buggy_nodes(self) -> list[Solution]` - Get all buggy nodes that could be debugged.
- `good_nodes(self) -> list[Solution]` - Get all non-buggy nodes with valid performance.
- `best_node(self) -> Solution | None` - Get best performing solution.
- `best_performance(self) -> float` - Get best performance achieved so far.
- `get_history(self, limit: int) -> list[dict]` - Get recent search history for agent consumption.
- `summarize(self) -> str` - Generate text summary of search progress.
- `get_improvement_trend(self, window: int) -> float` - Calculate improvement trend over recent successful iterations.
- `failure_rate(self) -> float` - Calculate overall failure rate.
- `get_tree_depth(self) -> int` - Get maximum depth of the search tree.
- `get_successful_improvements(self, limit: int) -> list[Solution]` - Get recent successful child nodes for learning.
- `to_dict(self) -> dict` - Serialize SearchJournal to dict for checkpointing.
- `from_dict(d: dict) -> 'SearchJournal'` - Deserialize SearchJournal from checkpoint dict.

---
## `search/policy.py`
Search policy abstract base class.

**`SearchPolicy`** - Search strategy for selecting which solution node to expand next.
- `decide_next_solution(self, journal: 'SearchJournal', context: 'BuildContext', iteration: int, max_iterations: int) -> Optional['Solution']` - Select which solution node to expand in the next iteration.
- `should_stop(self, journal: 'SearchJournal', iteration: int, max_iterations: int) -> bool` - Decide if search should terminate early.

---
## `search/tree_policy.py`
Tree-search policy inspired by AIDE.

**`TreeSearchPolicy`** - AIDE-inspired tree-search with draft/debug/improve stages.
- `__init__(self, num_drafts: int, debug_prob: float, max_debug_depth: int)`
- `decide_next_solution(self, journal: SearchJournal, context: BuildContext, iteration: int, max_iterations: int) -> Solution | None` - Decide which solution node to expand next.
- `should_stop(self, journal: SearchJournal, iteration: int, max_iterations: int) -> bool` - Decide if search should terminate early.

---
## `templates/features/pipeline_fitter.py`
Fit sklearn Pipeline on dataset.

**Functions:**
- `fit_pipeline(dataset_uri: str, pipeline: Pipeline, target_columns: list[str], group_column: str | None) -> Pipeline` - Fit sklearn Pipeline on the provided dataset.

---
## `templates/features/pipeline_runner.py`
Apply fitted sklearn Pipeline to full dataset via Spark.

**Functions:**
- `transform_dataset_via_spark(spark: SparkSession, dataset_uri: str, fitted_pipeline: Pipeline, output_uri: str, target_columns: list[str], pipeline_code: str, group_column: str | None) -> str` - Apply fitted sklearn Pipeline to full dataset using Spark UDFs.

---
## `templates/inference/catboost_predictor.py`
Standard CatBoost predictor - NO Plexe dependencies.

**`CatBoostPredictor`** - Standalone CatBoost predictor.
- `__init__(self, model_dir: str)`
- `predict(self, x: pd.DataFrame) -> pd.DataFrame` - Make predictions on input DataFrame.

---
## `templates/inference/keras_predictor.py`
Standard Keras predictor - NO Plexe dependencies.

**`KerasPredictor`** - Standalone Keras predictor.
- `__init__(self, model_dir: str)`
- `predict(self, x: pd.DataFrame) -> pd.DataFrame` - Make predictions on input DataFrame.

---
## `templates/inference/lightgbm_predictor.py`
Standard LightGBM predictor - NO Plexe dependencies.

**`LightGBMPredictor`** - Standalone LightGBM predictor.
- `__init__(self, model_dir: str)`
- `predict(self, x: pd.DataFrame) -> pd.DataFrame` - Make predictions on input DataFrame.

---
## `templates/inference/pytorch_predictor.py`
Standard PyTorch predictor - NO Plexe dependencies.

**`PyTorchPredictor`** - Standalone PyTorch predictor.
- `__init__(self, model_dir: str)`
- `predict(self, x: pd.DataFrame) -> pd.DataFrame` - Make predictions on input DataFrame.

---
## `templates/inference/xgboost_predictor.py`
Standard XGBoost predictor - NO Plexe dependencies.

**`XGBoostPredictor`** - Standalone XGBoost predictor.
- `__init__(self, model_dir: str)`
- `predict(self, x: pd.DataFrame) -> pd.DataFrame` - Make predictions on input DataFrame.

---
## `templates/training/train_catboost.py`
Hardcoded robust CatBoost training loop.

**Functions:**
- `train_catboost(untrained_model_path: Path, train_uri: str, val_uri: str, output_dir: Path, target_column: str) -> dict` - Train CatBoost model directly (no Spark).
- `main()` - No description

---
## `templates/training/train_keras.py`
Hardcoded robust Keras training loop.

**Functions:**
- `train_keras(untrained_model_path: Path, train_uri: str, val_uri: str, output_dir: Path, target_column: str, epochs: int, batch_size: int) -> dict` - Train Keras model directly.

---
## `templates/training/train_lightgbm.py`
Hardcoded robust LightGBM training loop.

**Functions:**
- `train_lightgbm(untrained_model_path: Path, train_uri: str, val_uri: str, output_dir: Path, target_column: str, group_column: str | None) -> dict` - Train LightGBM model directly (no Spark).
- `main()` - No description

---
## `templates/training/train_pytorch.py`
Hardcoded robust PyTorch training loop.

**Functions:**
- `train_pytorch(untrained_model_path: Path, train_uri: str, val_uri: str, output_dir: Path, target_column: str, epochs: int, batch_size: int) -> dict` - Train PyTorch model directly.

---
## `templates/training/train_xgboost.py`
Hardcoded robust XGBoost training loop.

**Functions:**
- `train_xgboost(untrained_model_path: Path, train_uri: str, val_uri: str, output_dir: Path, target_column: str, group_column: str | None) -> dict` - Train XGBoost model directly (no Spark).
- `main()` - No description

---
## `tools/submission.py`
Submission tools for agents.

**Functions:**
- `get_save_pipeline_fn(context: BuildContext, sample_df: pd.DataFrame)` - Factory: Returns pipeline submission function.
- `get_save_pipeline_code_tool(context: BuildContext, train_sample_df: pd.DataFrame, val_sample_df: pd.DataFrame)` - Factory: Returns pipeline code submission tool.
- `get_save_model_fn(context: BuildContext, model_type: str, max_epochs: int)` - Factory: Returns model submission function.
- `get_submit_metric_choice_tool(context: BuildContext)` - Factory: Returns metric choice submission tool for MetricSelectorAgent.
- `get_register_statistical_profile_tool(context: BuildContext)` - Factory: Returns statistical profile submission tool.
- `get_register_layout_tool(context: BuildContext)` - Factory: Returns layout detection submission tool.
- `get_register_eda_report_tool(context: BuildContext)` - Factory: Returns EDA report submission tool.
- `get_save_split_uris_tool(context: BuildContext)` - Factory: Returns split URI submission tool.
- `get_save_sample_uris_tool(context: BuildContext)` - Factory: Returns sample URIs submission tool.
- `get_save_metric_implementation_fn(context: BuildContext)` - Factory: Returns metric implementation submission function.
- `get_validate_baseline_predictor_tool(context: BuildContext, val_sample_df)` - Factory: Returns baseline predictor validation tool.
- `get_save_baseline_code_tool(context: BuildContext, val_sample_df)` - Factory: Returns baseline code saving tool.
- `get_evaluate_baseline_performance_tool(context: BuildContext, val_sample_df)` - Factory: Returns baseline performance evaluation tool.
- `get_save_hypothesis_tool(context: BuildContext, expand_node_id: int)` - Factory: Returns hypothesis submission tool for HypothesiserAgent.
- `get_save_plan_tool(context: BuildContext, hypothesis: 'Hypothesis', allowed_model_types: list[str] | None)` - Factory: Returns plan submission tool for PlannerAgent.
- `get_save_insight_tool(insight_store: InsightStore)` - Factory: Returns insight submission tool for InsightExtractorAgent.
- `get_register_core_metrics_tool(context: BuildContext)` - Factory: Returns core metrics submission tool for ModelEvaluatorAgent.
- `get_register_diagnostic_report_tool(context: BuildContext)` - Factory: Returns diagnostic report submission tool for ModelEvaluatorAgent.
- `get_register_robustness_report_tool(context: BuildContext)` - Factory: Returns robustness report submission tool for ModelEvaluatorAgent.
- `get_register_explainability_report_tool(context: BuildContext)` - Factory: Returns explainability report submission tool for ModelEvaluatorAgent.
- `get_register_baseline_comparison_tool(context: BuildContext)` - Factory: Returns baseline comparison submission tool for ModelEvaluatorAgent.
- `get_register_final_evaluation_tool(context: BuildContext)` - Factory: Returns final evaluation submission tool for ModelEvaluatorAgent.

---
## `utils/dashboard/discovery.py`
Experiment discovery and metadata extraction for dashboard.

**`ExperimentMetadata`** - Metadata for a discovered experiment.

**Functions:**
- `discover_experiments(workdir: Path) -> list[ExperimentMetadata]` - Discover all experiments in workdir.
- `load_experiment_checkpoints(experiment_path: Path) -> dict[str, dict]` - Load all checkpoints for an experiment.

---
## `utils/dashboard/tabs/baselines.py`
Baselines tab: Heuristic baseline info.

**Functions:**
- `render_baselines(checkpoints, exp_path)` - Render baselines tab.

---
## `utils/dashboard/tabs/data_preparation.py`
Data Preparation tab: Splits, sample sizes, data preview.

**Functions:**
- `render_data_preparation(checkpoints, exp_path)` - Render data preparation tab.

---
## `utils/dashboard/tabs/data_understanding.py`
Data Understanding tab: Layout, stats, task analysis, metric.

**Functions:**
- `render_data_understanding(checkpoints, exp_path)` - Render data understanding tab.

---
## `utils/dashboard/tabs/evaluation.py`
Evaluation tab: Final test metrics, baseline comparison, diagnostics.

**Functions:**
- `render_evaluation(checkpoints, exp_path)` - Render evaluation tab.

---
## `utils/dashboard/tabs/model_package.py`
Model Package tab: File structure, metadata.

**Functions:**
- `render_model_package(exp_path)` - Render model package tab.

---
## `utils/dashboard/tabs/overview.py`
Overview tab: Phase progress, timeline, key metrics.

**Functions:**
- `render_overview(exp_meta, checkpoints)` - Render overview tab.

---
## `utils/dashboard/tabs/search_tree.py`
Search Tree tab: Tree visualization, performance chart, insights.

**Functions:**
- `render_search_tree(checkpoints, exp_path)` - Render search tree tab.

---
## `utils/dashboard/theme.py`
Custom theme and styling for dashboard.

**Functions:**
- `apply_custom_theme()` - Apply custom CSS for dense, professional layout.

---
## `utils/dashboard/utils.py`
Utility functions for dashboard data loading.

**Functions:**
- `load_report(exp_path: Path, report_name: str) -> dict | None` - Load YAML report from DirNames.BUILD_DIR/reports/.
- `load_code_file(file_path: Path) -> str | None` - Load Python code file.
- `load_parquet_sample(uri: str, limit: int) -> pd.DataFrame | None` - Load first N rows from parquet file.
- `get_parquet_row_count(uri: str) -> int | None` - Get row count from parquet file.
- `load_json_file(file_path: Path) -> dict | None` - Load JSON file.

---
## `utils/litellm_wrapper.py`
LiteLLM model wrapper with retry logic and optional post-call hook.

**`PlexeLiteLLMModel`** - LiteLLM model wrapper with automatic retries and an optional post-call hook.
- `__init__(self, model_id: str, extra_headers: dict[str, str] | None, on_llm_call: Callable[[str, Any, int], None] | None)`
- `generate(self)` - Generate with automatic retries, header injection, and post-call hook.
- `chat(self)` - Chat with automatic retries, header injection, and post-call hook.

---
## `utils/reporting.py`
Utilities for saving agent reports to disk.

**Functions:**
- `save_report(work_dir: Path, report_name: str, content: Any) -> Path` - Save agent report to workdir/DirNames.BUILD_DIR/reports/ as YAML.

---
## `utils/s3.py`
S3 utilities for downloading datasets.

**Functions:**
- `download_s3_uri(s3_uri: str, local_dir: Path | None) -> str` - Download S3 URI to local directory.

---
## `utils/tooling.py`
This module provides utility functions for defining and managing tools for AI agents.

**`AgentInvocationError`** - Raised when an agent calls an @agentinspectable function with invalid arguments.
- `__init__(self, func_name: str, help_text: str)`

**Functions:**
- `agentinspectable(func)` - Decorator for functions intended to be made available for calling to AI agents using mechanisms

---
## `utils/tracing.py`
OpenTelemetry tracing decorators for agents and tools.

**Functions:**
- `setup_opentelemetry(config: Config)` - Initialize OpenTelemetry tracing with backend-agnostic configuration.
- `agent_span(name: str)` - Wrap an agent call in a named span for observability.
- `tool_span(fn) -> Callable` - Wrap a tool call in a named span with tool metadata, including inputs and outputs.

---
## `validation/validators.py`
Validation functions for pipelines, models, and other agent outputs.

**Functions:**
- `validate_sklearn_pipeline(pipeline: Pipeline, sample_df: pd.DataFrame, target_columns: list[str]) -> tuple[bool, str]` - Validate that an sklearn Pipeline is well-formed and functional.
- `validate_pipeline_consistency(pipeline: Pipeline, train_sample: pd.DataFrame, val_sample: pd.DataFrame, target_columns: list[str]) -> tuple[bool, str]` - Validate pipeline produces consistent output shape on train/val samples.
- `validate_xgboost_params(params: dict[str, Any]) -> tuple[bool, str]` - Validate XGBoost hyperparameters.
- `validate_catboost_params(params: dict[str, Any]) -> tuple[bool, str]` - Validate CatBoost hyperparameters.
- `validate_model_definition(model_type: str, definition: dict[str, Any]) -> tuple[bool, str]` - Validate model definition based on model type.
- `validate_metric_function_object(func) -> tuple[bool, str]` - Validate metric computation function object.
- `validate_dataset_splits(spark, train_uri: str, val_uri: str, test_uri: str | None, expected_ratios: dict[str, float]) -> tuple[bool, str]` - Validate that dataset splits were created correctly.
- `validate_keras_model(model: Any, task_analysis: dict) -> tuple[bool, str]` - Validate Keras 3 model structure.
- `validate_keras_optimizer(optimizer: Any) -> tuple[bool, str]` - Validate Keras 3 optimizer.
- `validate_keras_loss(loss: Any) -> tuple[bool, str]` - Validate Keras 3 loss function.

---
## `viz.py`
Streamlit dashboard for plexe.

**Functions:**
- `main()` - No description

---
## `workflow.py`
Main workflow orchestrator.

**Functions:**
- `build_model(spark: SparkSession, train_dataset_uri: str, test_dataset_uri: str | None, user_id: str, intent: str, experiment_id: str, work_dir: Path, runner: TrainingRunner, search_policy: SearchPolicy, config: Config, integration: WorkflowIntegration, enable_final_evaluation: bool, on_checkpoint_saved: Callable[[str, Path, Path], None] | None, pause_points: list[str] | None, on_pause: Callable[[str], None] | None, user_feedback: dict | None) -> tuple[Solution, dict, EvaluationReport | None] | None` - Main workflow orchestrator.
- `sanitize_dataset_column_names(spark: SparkSession, dataset_uri: str, context: BuildContext) -> str` - Sanitize column names by replacing special characters with underscores.
- `analyze_data(spark: SparkSession, dataset_uri: str, context: BuildContext, config: Config, on_checkpoint_saved: Callable[[str, Path, Path], None] | None)` - Phase 1: Layout detection + Statistical + ML task analysis + metric selection.
- `prepare_data(spark: SparkSession, training_dataset_uri: str, test_dataset_uri: str | None, context: BuildContext, config: Config, integration: WorkflowIntegration, generate_test_set: bool, on_checkpoint_saved: Callable[[str, Path, Path], None] | None)` - Phase 2: Split dataset and extract sample.
- `build_baselines(spark: SparkSession, context: BuildContext, config: Config, on_checkpoint_saved: Callable[[str, Path, Path], None] | None)` - Phase 3: Build baseline models.
- `search_models(spark: SparkSession, context: BuildContext, runner: TrainingRunner, search_policy: SearchPolicy, config: Config, integration: WorkflowIntegration, on_checkpoint_saved: Callable[[str, Path, Path], None] | None, restored_journal: SearchJournal | None, restored_insight_store: InsightStore | None) -> Solution | None` - Phase 4: Iterative tree-search for best model.
- `retrain_on_full_dataset(spark: SparkSession, best_solution: Solution, context: BuildContext, runner: TrainingRunner, config: Config) -> Solution` - Retrain best solution on FULL dataset.
- `evaluate_final(spark: SparkSession, context: BuildContext, solution: Solution, config: Config, on_checkpoint_saved: Callable[[str, Path, Path], None] | None) -> dict` - Phase 5: Final evaluation on test set sample.
- `package_final_model(spark: SparkSession, context: BuildContext, solution: Solution, final_metrics: dict, on_checkpoint_saved: Callable[[str, Path, Path], None] | None) -> Path` - Package all final deliverables into a unified directory.

---