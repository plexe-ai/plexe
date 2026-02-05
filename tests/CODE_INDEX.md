# Code Index: tests

> Generated on 2026-02-05 14:29:56

Test suite structure and test case documentation.

## `benchmark/mle_bench.py`
This script automates the setup, execution, and grading process for the "mle-bench" framework using plexe.

**Functions:**
- `main(cli_args)` - Main entry point for the script

---
## `benchmark/mlebench/core/config.py`
Configuration management for MLE Bench runner.

**`ConfigManager`** - Class to handle configuration loading and generation
- `load_config(config_path)` - Load configuration from YAML file
- `ensure_config_exists(rebuild: bool)` - Check if `mle-bench-config.yaml` exists, and if not, generate it from `mle-bench-config.yaml.jinja`

---
## `benchmark/mlebench/core/models.py`
Data models for MLE Bench runner.

**`TestResult`** - Structured class to store test results

**`SubmissionInfo`** - Structured class to store submission information

---
## `benchmark/mlebench/core/runner.py`
Main runner class for MLE Bench benchmark.

**`MLEBenchRunner`** - Main class to run the MLE-bench benchmarking framework
- `__init__(self)`
- `setup(self, cli_args)` - Set up the MLE-bench environment
- `run(self)` - Run the tests and evaluate the results

---
## `benchmark/mlebench/core/validator.py`
Environment validation for MLE Bench runner.

**`EnvironmentValidator`** - Class to validate environment setup
- `ensure_kaggle_credentials()` - Ensure that Kaggle API credentials are set up
- `check_llm_api_keys()` - Check if required LLM API key environment variables are set

---
## `benchmark/mlebench/runners/grader.py`
Grader for MLE Bench benchmark results.

**`MLEBenchGrader`** - Class to handle grading of model submissions
- `grade_agent(submissions: List[SubmissionInfo])` - Grade the agent's performance based on the test results

---
## `benchmark/mlebench/runners/setup.py`
Setup and preparation for MLE Bench runner.

**`MLEBenchSetup`** - Class to handle setup of MLE-bench framework
- `setup_mle_bench(config, rebuild: bool)` - Set up the MLE-bench framework
- `prepare_datasets(config)` - Prepare datasets listed in the config file

---
## `benchmark/mlebench/runners/test_runner.py`
Test runner for MLE Bench benchmarks.

**`TestRunner`** - Class to run tests using plexe
- `__init__(self, config)`
- `verify_test_files(self, test_name) -> bool` - Verify that all required files for a test exist
- `prepare_test(self, test_name)` - Prepare test data and create output directory
- `build_model(self, test_name, test_data_info)` - Build a model using plexe
- `validate_predictions(self, predictions, expected_columns)` - Validate prediction data has required columns and format
- `generate_predictions(self, model, test_name, test_data_info)` - Generate predictions using the model
- `save_model(self, model, test_name, output_dir)` - Save model for future reference
- `run_tests(self) -> List[SubmissionInfo]` - Run tests from the configuration file using plexe

---
## `benchmark/mlebench/utils/command.py`
Command execution utilities for MLE Bench runner.

**`CommandRunner`** - Class to handle command execution and error handling
- `run(command, error_message, success_message)` - Run a shell command and handle errors

**Functions:**
- `working_directory(path)` - Context manager for changing the current working directory

---
## `benchmark/mlebench/utils/error.py`
Error handling utilities for MLE Bench runner.

**`ErrorHandler`** - Class to handle and format errors
- `handle_error(operation, context, error, exit_on_failure)` - Handle exceptions with consistent formatting

---
## `integration/test_binary_classification.py`
Integration test for binary classification models using plexe.

**Functions:**
- `heart_data()` - Generate synthetic heart disease data for testing.
- `heart_input_schema()` - Define the input schema for heart disease prediction.
- `heart_output_schema()` - Define the output schema for heart disease prediction.
- `model_dir(tmpdir)` - Create and manage a temporary directory for model files.
- `run_around_tests(model_dir)` - Set up and tear down for each test.
- `test_heart_disease_classification(heart_data, heart_input_schema, heart_output_schema)` - Test binary classification for heart disease prediction.

---
## `integration/test_customer_churn.py`
Integration test for customer churn prediction models using plexe.

**Functions:**
- `churn_data()` - Generate synthetic customer churn data for testing.
- `churn_input_schema()` - Define the input schema for churn prediction with field validations.
- `churn_output_schema()` - Define the output schema for churn prediction with probability.
- `model_dir(tmpdir)` - Create and manage a temporary directory for model files.
- `run_around_tests(model_dir)` - Set up and tear down for each test.
- `test_customer_churn_prediction(churn_data, churn_input_schema, churn_output_schema)` - Test customer churn prediction with probability output.

---
## `integration/test_model_description.py`
Integration test for model description functionality in plexe.

**Functions:**
- `iris_data()` - Generate synthetic iris data for testing.
- `iris_input_schema()` - Define the input schema for iris classification.
- `iris_output_schema()` - Define the output schema for iris classification.
- `model_dir(tmpdir)` - Create and manage a temporary directory for model files.
- `run_around_tests(model_dir)` - Set up and tear down for each test.
- `verify_description_format(description, format_type)` - Verify that a description has the expected format and content.
- `test_model_description(iris_data, iris_input_schema, iris_output_schema)` - Test model description generation in various formats and content verification.

---
## `integration/test_multiclass_classification.py`
Integration test for multiclass classification models using plexe.

**Functions:**
- `sentiment_data()` - Generate synthetic sentiment data for testing.
- `sentiment_input_schema()` - Define the input schema for sentiment analysis.
- `sentiment_output_schema()` - Define the output schema for sentiment analysis.
- `model_dir(tmpdir)` - Create and manage a temporary directory for model files.
- `run_around_tests(model_dir)` - Set up and tear down for each test.
- `test_multiclass_classification(sentiment_data, sentiment_input_schema, sentiment_output_schema)` - Test multiclass classification for sentiment analysis.

---
## `integration/test_ray_integration.py`
Integration test for Ray-based distributed training.

**Functions:**
- `sample_dataset()` - Create a simple synthetic dataset for testing.
- `test_model_with_ray(sample_dataset)` - Test building a model with Ray-based distributed execution.

---
## `integration/test_recommendation.py`
Integration test for recommendation models using plexe.

**Functions:**
- `product_data()` - Generate synthetic product recommendation data for testing.
- `recommendation_input_schema()` - Define the input schema for product recommendation.
- `recommendation_output_schema()` - Define the output schema for product recommendation.
- `model_dir(tmpdir)` - Create and manage a temporary directory for model files.
- `run_around_tests(model_dir)` - Set up and tear down for each test.
- `test_product_recommendation(product_data, recommendation_input_schema, recommendation_output_schema)` - Test recommendation model for suggesting related products.

---
## `integration/test_regression.py`
Integration test for regression models using plexe.

**Functions:**
- `house_data()` - Generate synthetic house price data for testing.
- `house_input_schema()` - Define the input schema for house price prediction.
- `house_output_schema()` - Define the output schema for house price prediction.
- `model_dir(tmpdir)` - Create and manage a temporary directory for model files.
- `run_around_tests(model_dir)` - Set up and tear down for each test.
- `test_house_price_regression(house_data, house_input_schema, house_output_schema)` - Test regression for house price prediction.

---
## `integration/test_schema_validation.py`
Integration test for schema validation in plexe.

**Functions:**
- `house_data()` - Generate synthetic house price data for testing.
- `house_data_copy(house_data)` - Create a copy of the house data to avoid mutation issues.
- `validated_input_schema()` - Define the input schema for house price prediction with validation.
- `validated_output_schema()` - Define the output schema for house price prediction with validation.
- `model_dir(tmpdir)` - Create and manage a temporary directory for model files.
- `run_around_tests(model_dir)` - Set up and tear down for each test.
- `test_input_validation(house_data_copy, validated_input_schema, validated_output_schema)` - Test validation of input schema.
- `test_output_validation(house_data_copy, validated_input_schema)` - Test validation of output schema.

---
## `integration/test_time_series.py`
Integration test for time series forecasting models using plexe.

**Functions:**
- `sales_data()` - Generate synthetic time series data for testing.
- `sales_data_copy(sales_data)` - Create a copy of the sales data to avoid mutation issues.
- `sales_input_schema()` - Define the input schema for sales forecasting.
- `sales_output_schema()` - Define the output schema for sales forecasting.
- `model_dir(tmpdir)` - Create and manage a temporary directory for model files.
- `run_around_tests(model_dir)` - Set up and tear down for each test.
- `test_time_series_forecasting(sales_data_copy, sales_input_schema, sales_output_schema)` - Test time series forecasting for sales prediction.

---
## `unit/internal/common/datasets/test_adapter.py`
Tests for the DatasetAdapter class.

**`MockDataset`** - Mock dataset implementation for testing.
- `__init__(self, features)`
- `split(self, train_ratio, val_ratio, test_ratio, stratify_column, random_state)` - No description
- `sample(self, n, frac, replace, random_state)` - No description
- `to_bytes(self)` - No description
- `from_bytes(cls, data)` - No description
- `structure(self)` - No description

**Functions:**
- `test_adapter_coerce_pandas()` - Test that DatasetAdapter.coerce handles pandas DataFrames.
- `test_adapter_coerce_dataset()` - Test that DatasetAdapter.coerce passes through Dataset instances.
- `test_adapter_auto_detect()` - Test the auto_detect functionality.
- `test_adapter_coerce_unsupported()` - Test error handling for unsupported dataset types.
- `test_adapter_features()` - Test the features extraction functionality.

---
## `unit/internal/common/datasets/test_interface.py`
Tests for the dataset interface.

**`MinimalDataset`** - Minimal implementation of Dataset for testing.
- `split(self, train_ratio, val_ratio, test_ratio, stratify_column, random_state)` - No description
- `sample(self, n, frac, replace, random_state)` - No description
- `to_bytes(self)` - No description
- `from_bytes(cls, data)` - No description
- `structure(self)` - No description

**`IncompleteDataset`** - Dataset implementation missing required methods.
- `split(self, train_ratio, val_ratio, test_ratio, stratify_column, random_state)` - No description

**Functions:**
- `test_dataset_structure_creation()` - Test creation of a DatasetStructure with valid parameters.
- `test_dataset_structure_tensor_modality()` - Test creation of a DatasetStructure with tensor modality.
- `test_dataset_structure_other_modality()` - Test creation of a DatasetStructure with 'other' modality.
- `test_dataset_instantiation()` - Test that Dataset can't be instantiated directly.
- `test_minimal_dataset()` - Test that a minimal implementation of Dataset can be instantiated.
- `test_incomplete_dataset()` - Test that a Dataset implementation missing required methods raises errors.

---
## `unit/internal/common/datasets/test_tabular.py`
Tests for the TabularDataset implementation.

**Functions:**
- `test_tabular_dataset_creation()` - Test that TabularDataset can be created from pandas DataFrame.
- `test_tabular_dataset_validation()` - Test validation of input data types.
- `test_tabular_dataset_split_standard()` - Test standard train/val/test split with default ratios.
- `test_tabular_dataset_split_custom_ratios()` - Test train/val/test split with custom ratios.
- `test_tabular_dataset_split_stratified()` - Test stratified splitting.
- `test_tabular_dataset_split_reproducibility()` - Test that splits are reproducible with same random state.
- `test_tabular_dataset_split_edge_cases()` - Test edge cases for splitting.
- `test_tabular_dataset_sample()` - Test sampling functionality.
- `test_tabular_dataset_serialization()` - Test that TabularDataset can be serialized and deserialized.
- `test_tabular_dataset_serialization_error_handling()` - Test error handling during serialization/deserialization.
- `test_tabular_dataset_structure()` - Test structure property.
- `test_tabular_dataset_file_storage(tmp_path)` - Test that TabularDataset can be stored to and loaded from a file.
- `test_tabular_dataset_conversion()` - Test conversion to pandas and numpy.
- `test_tabular_dataset_getitem()` - Test __getitem__ functionality.
- `test_tabular_dataset_len()` - Test __len__ functionality.

---
## `unit/internal/common/utils/test_dataset_storage.py`
Tests for the dataset storage utilities.

**Functions:**
- `test_write_and_read_file(tmp_path)` - Test writing a dataset to a file and reading it back.
- `test_file_storage_error_handling(tmp_path)` - Test error handling for file storage operations.
- `test_shared_memory_error_handling()` - Test error handling in shared memory functions.

---
## `unit/internal/models/callbacks/test_mlflow.py`
Unit tests for the MLFlowCallback class.

**Functions:**
- `setup_env()` - Set up common test environment.
- `test_callback_initialization()` - Test that the MLFlowCallback can be initialized properly.
- `test_build_start(mock_create_experiment, mock_get_experiment, _, setup_env)` - Test on_build_start callback.
- `test_build_start_new_experiment(mock_active_run, mock_set_experiment, mock_create_experiment, mock_get_experiment, mock_set_tracking_uri, setup_env)` - Test on_build_start with a new experiment.
- `test_build_end(setup_env)` - Test on_build_end callback.
- `test_log_metric(setup_env)` - Test _log_metric helper method.

---
## `unit/internal/models/entities/test_metric.py`
Module: test_metric_class

**Functions:**
- `test_comparator_higher_is_better()` - No description
- `test_comparator_lower_is_better()` - No description
- `test_comparator_target_is_better()` - No description
- `test_comparator_invalid_target()` - No description
- `test_comparator_floating_point_precision()` - No description
- `test_metric_higher_is_better()` - No description
- `test_metric_lower_is_better()` - No description
- `test_metric_target_is_better()` - No description
- `test_metric_different_names()` - No description
- `test_metric_invalid_comparison()` - No description
- `test_metric_is_valid()` - No description
- `test_metric_repr_and_str()` - No description
- `test_metric_transitivity()` - No description
- `test_metric_collection_sorting()` - No description

---
## `unit/internal/models/execution/test_factory.py`
Test the executor factory.

**Functions:**
- `test_get_executor_class_non_distributed()` - Test that ProcessExecutor is returned when distributed=False.
- `test_get_executor_class_distributed()` - Test that RayExecutor is returned when distributed=True and Ray is available.
- `test_get_executor_class_distributed_ray_not_available()` - Test that ProcessExecutor is returned as fallback when Ray is not available.

---
## `unit/internal/models/execution/test_process_executor.py`
Unit tests for the ProcessExecutor class and its associated components.

**`TestProcessExecutor`** - No description
- `setup_method(self)` - No description
- `teardown_method(self)` - No description
- `test_constructor_creates_working_directory(self)` - No description
- `test_run_successful_execution(self, mock_write_table)` - No description
- `test_run_timeout(self, mock_popen)` - No description
- `test_run_exception(self, mock_popen)` - No description
- `test_dataset_written_to_file(self, mock_write_table)` - No description

---
## `unit/internal/models/validation/primitives/test_syntax.py`
No description

**Functions:**
- `syntax_validator()` - Fixture to provide an instance of SyntaxValidator.
- `test_valid_code(syntax_validator)` - Test that the validate method correctly identifies valid Python code.
- `test_invalid_code(syntax_validator)` - Test that the validate method correctly identifies invalid Python code.
- `test_empty_code(syntax_validator)` - Test that the validate method handles empty code correctly.
- `test_code_with_comments(syntax_validator)` - Test that the validate method handles code containing only comments.
- `test_code_with_syntax_warning(syntax_validator)` - Test that the validate method handles code with a warning but no syntax error.
- `test_code_with_non_ascii_characters(syntax_validator)` - Test that the validate method handles code with non-ASCII characters.
- `test_code_with_indentation_error(syntax_validator)` - Test that the validate method correctly identifies indentation errors.
- `test_code_with_nested_functions(syntax_validator)` - Test that the validate method handles code with nested functions.
- `test_code_with_large_input(syntax_validator)` - Test that the validate method handles a large amount of valid code.

---
## `unit/test_datasets.py`
No description

**`TestDataGeneration`** - Test suite for data generation with comprehensive mocking
- `setup_mocks(self)` - Setup all required mocks for the test class
- `test_basic_generation(self, sample_schema, mock_generated_data)` - Test basic data generation
- `test_data_augmentation(self, sample_schema, mock_generated_data)` - Test data augmentation with existing dataset

**Functions:**
- `sample_schema()` - Test schema for house price prediction
- `mock_generated_data()` - Mock data generation output

---
## `unit/test_fileio.py`
Unit tests for plexe.fileio module, including backwards compatibility testing.

**`TestFileIO`** - Test cases for fileio module functionality.
- `test_load_model_backwards_compatibility_v0_18_3(self)` - Test loading a model bundle from v0.18.3 for backwards compatibility.
- `test_load_model_backwards_compatibility_v0_23_2(self)` - Test loading a model bundle from v0.23.2 for backwards compatibility.
- `test_load_model_file_not_found(self)` - Test that load_model raises appropriate error for missing files.

---
## `utils/utils.py`
No description

**Functions:**
- `generate_heart_data(n_samples, random_seed)` - Generate synthetic heart disease data for testing.
- `generate_house_prices_data(n_samples, random_seed)` - Generate synthetic house price data for regression testing.
- `generate_customer_churn_data(n_samples, random_seed)` - Generate synthetic customer churn data for classification testing.
- `generate_sentiment_data(n_samples, random_seed)` - Generate synthetic sentiment analysis data for text classification testing.
- `generate_product_recommendation_data(n_samples, random_seed)` - Generate synthetic product recommendation data.
- `generate_time_series_data(n_samples, random_seed)` - Generate synthetic time series data for forecasting testing.
- `verify_prediction(prediction, expected_schema)` - Verify that a prediction matches expected format.
- `verify_model_description(description)` - Verify that a model description contains expected fields.
- `cleanup_files(model_dir)` - Clean up any files created during tests.

---