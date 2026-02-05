# Code Index: plexe

> Generated on 2026-02-05 14:29:56

Code structure and public interface documentation for the **plexe** package.

## `agents/agents.py`
This module defines a multi-agent ML engineering system for building machine learning models.

**`ModelGenerationResult`** - No description

**`PlexeAgent`** - Multi-agent ML engineering system for building machine learning models.
- `__init__(self, orchestrator_model_id: str, ml_researcher_model_id: str, ml_engineer_model_id: str, ml_ops_engineer_model_id: str, tool_model_id: str, verbose: bool, max_steps: int, distributed: bool, chain_of_thought_callable: Optional[Callable], max_solutions: int)`
- `run(self, task, additional_args: dict) -> ModelGenerationResult` - Run the orchestrator agent to generate a machine learning model.

---
## `agents/conversational.py`
Conversational Agent for guiding users through ML model definition and initiation.

**`ConversationalAgent`** - Agent for conversational model definition and build initiation.
- `__init__(self, model_id: str, verbose: bool)`

---
## `agents/dataset_analyser.py`
Exploratory Data Analysis (EDA) Agent for data analysis and insights in ML models.

**`EdaAgent`** - Agent for performing exploratory data analysis on datasets.
- `__init__(self, model_id: str, verbose: bool, chain_of_thought_callable: Callable)`
- `run(self, intent: str, dataset_names: List[str]) -> bool` - Run the EDA agent to analyze datasets and create EDA reports.

---
## `agents/dataset_splitter.py`
Dataset Splitter Agent for partitioning datasets into training, validation, and test sets.

**`DatasetSplitterAgent`** - Agent for intelligently splitting datasets into train, validation, and test sets.
- `__init__(self, model_id: str, verbose: bool, chain_of_thought_callable: Optional[Callable])`

---
## `agents/feature_engineer.py`
Feature Engineering Agent for transforming raw datasets into optimized features for ML models.

**`FeatureEngineeringAgent`** - Agent for creating optimized features from raw datasets for ML models.
- `__init__(self, model_id: str, verbose: bool, chain_of_thought_callable: Optional[Callable])`

---
## `agents/model_packager.py`
Model Packager Agent for creating production-ready inference code for ML models.

**`ModelPackagerAgent`** - Agent for creating production-ready inference code for ML models.
- `__init__(self, model_id: str, tool_model_id: str, verbose: bool, chain_of_thought_callable: Optional[Callable], schema_resolver_agent)`

---
## `agents/model_planner.py`
No description

**`ModelPlannerAgent`** - Agent responsible for planning ML model solutions based on provided requirements.
- `__init__(self, model_id: str, verbose: bool, chain_of_thought_callable: callable, max_solutions: int)`

---
## `agents/model_tester.py`
Model Tester Agent for comprehensive testing and evaluation of finalized ML models.

**`ModelTesterAgent`** - Agent for comprehensive testing and evaluation of finalized ML models.
- `__init__(self, model_id: str, verbose: bool, chain_of_thought_callable: Optional[Callable])`

---
## `agents/model_trainer.py`
Model Trainer Agent for training ML models based on provided plans.

**`ModelTrainerAgent`** - Agent for training ML models based on provided plans.
- `__init__(self, ml_engineer_model_id: str, tool_model_id: str, distributed: bool, verbose: bool, chain_of_thought_callable: callable, schema_resolver_agent)`

---
## `agents/schema_resolver.py`
Schema Resolver Agent for inferring input and output schemas for ML models.

**`SchemaResolverAgent`** - Agent for resolving input and output schemas for ML models.
- `__init__(self, model_id: str, verbose: bool, chain_of_thought_callable: Callable)`

---
## `callbacks.py`
Callbacks for model building process in Plexe.

**`BuildStateInfo`** - Consolidated information about model build state at any point in the process.

**`Callback`** - Abstract base class for callbacks during model building.
- `on_build_start(self, info: BuildStateInfo) -> None` - Called when the model building process starts.
- `on_build_end(self, info: BuildStateInfo) -> None` - Called when the model building process ends.
- `on_iteration_start(self, info: BuildStateInfo) -> None` - Called at the start of each model building iteration.
- `on_iteration_end(self, info: BuildStateInfo) -> None` - Called at the end of each model building iteration.

---
## `config.py`
Configuration for the plexe library.

**Functions:**
- `is_package_available(package_name: str) -> bool` - Check if a Python package is available/installed.
- `configure_logging(level: str | int, file: str) -> None` - No description

---
## `core/entities/solution.py`
This module defines the `Solution` class used to represent complete ML pipelines.

**`Solution`** - Represents a complete ML solution from planning through deployment.

---
## `core/interfaces/feature_transformer.py`
This module defines the FeatureTransformer interface, which all generated feature transformers must implement.

**`FeatureTransformer`** - Abstract base class for all dynamically generated feature transformers.
- `transform(self, inputs: pd.DataFrame) -> pd.DataFrame` - No description

---
## `core/interfaces/predictor.py`
This module defines the Predictor interface, which all dynamically generated inference codes must implement.

**`Predictor`** - Abstract base class for all dynamically generated inference code.
- `__init__(self, artifacts: List[Artifact])`
- `predict(self, inputs: dict) -> dict` - No description

---
## `core/object_registry.py`
This module provides a generic Registry pattern implementation for storing and retrieving objects by name or prefix.

**`Item`** - No description

**`ObjectRegistry`** - Registry for storing and retrieving objects by name.
- `register(self, t: Type[T], name: str, item: T, overwrite: bool, immutable: bool) -> None` - Register an item with a given name.
- `register_multiple(self, t: Type[T], items: Dict[str, T], overwrite: bool, immutable: bool) -> None` - Register multiple items with a given prefix.
- `get(self, t: Type[T], name: str) -> T` - Retrieve an item by name.
- `get_multiple(self, t: Type[T], names: List[str]) -> Dict[str, T]` - Retrieve multiple items by name.
- `get_all(self, t: Type[T]) -> Dict[str, T]` - Retrieve all items for a given prefix.
- `delete(self, t: Type[T], name: str) -> None` - Delete an item by name.
- `clear(self) -> None` - Clear all registered items.
- `list(self) -> List[str]` - List all registered item names.
- `list_by_type(self, t: Type[T]) -> List[str]` - List all registered names for a specific type.
- `get_all_solutions(self) -> List[Dict[str, Any]]` - Get all solutions tracked during model building.

---
## `core/state.py`
Model state definitions for Plexe.

**`ModelState`** - States a model can be in during its lifecycle.

---
## `core/storage.py`
Core storage functions for model and checkpoint persistence.

**`FallbackNoneLoader`** - No description

**Functions:**
- `fallback_to_none(loader, tag_suffix, node)` - No description

---
## `datasets.py`
This module provides the Dataset class, which represents a collection of data that can be real,

**`DatasetGenerator`** - Represents a dataset, which can contain real data, synthetic data, or both.
- `__init__(self, description: str, provider: str, schema: Type[BaseModel] | Dict[str, type], data: pd.DataFrame) -> None`
- `generate(self, num_samples: int)` - Generate synthetic data samples or augment existing data.
- `data(self) -> pd.DataFrame` - Get the dataset as a pandas DataFrame.

---
## `fileio.py`
This module provides file I/O utilities for saving and loading models to and from archive files.

**Functions:**
- `save_model(model: Any, path: str | Path) -> str` - Save a model to a tar archive.
- `load_model(path: str | Path)` - Load a model from a tar archive.
- `save_checkpoint(model: Any, iteration: int, path: Optional[str | Path]) -> str` - Save a model checkpoint to a tar archive.
- `load_checkpoint(checkpoint_path: Optional[str | Path], model_id: Optional[str], latest: bool) -> Any` - Load a model from a checkpoint.
- `list_checkpoints(model_id: Optional[str]) -> List[str]` - List available checkpoints.
- `delete_checkpoint(path: str | Path) -> bool` - Delete a specific checkpoint.
- `clear_checkpoints(model_id: Optional[str], older_than_days: Optional[int]) -> int` - Clear checkpoints based on filter criteria.

---
## `internal/common/datasets/adapter.py`
This module provides the DatasetAdapter class, which converts various dataset formats into standardized Dataset

**`DatasetAdapter`** - A utility class for converting different dataset formats into standardized Dataset objects.
- `coerce(dataset: Any) -> Dataset` - Converts a dataset to a standardized format.
- `auto_detect(cls, data: Any) -> Optional[str]` - Auto-detect the appropriate dataset type for the given data.
- `features(datasets: Dict[str, Dataset]) -> List[str]` - Extracts a flat list of feature names from the given datasets.

---
## `internal/common/datasets/interface.py`
This module defines the core interfaces for dataset handling in plexe.

**`DatasetStructure`** - Descriptor for the dataset structure.

**`Dataset`** - Base interface for all dataset implementations with universal operations.
- `split(self, train_ratio: float, val_ratio: float, test_ratio: float, stratify_column: str, random_state: int) -> Tuple[T, T, T]` - Split dataset into train, validation and test sets.
- `sample(self, n: int, frac: float, replace: bool, random_state: int) -> T` - Sample records from dataset.
- `to_bytes(self) -> bytes` - Serialize dataset to bytes.
- `from_bytes(cls: Type[T], data: bytes) -> T` - Deserialize dataset from bytes.
- `structure(self) -> DatasetStructure` - Return a descriptor of the dataset's structure.

**`TabularConvertible`** - Interface for datasets that can be converted to tabular formats.
- `to_pandas(self) -> pd.DataFrame` - Convert to pandas DataFrame.
- `to_numpy(self) -> np.ndarray` - Convert to numpy array.

**`TorchConvertible`** - Interface for datasets that can be converted to PyTorch formats.
- `to_torch_dataset(self)` - Convert to PyTorch Dataset.
- `to_torch_tensor(self)` - Convert to PyTorch Tensor.

**`TensorflowConvertible`** - Interface for datasets that can be converted to TensorFlow formats.
- `to_tf_dataset(self)` - Convert to TensorFlow Dataset.

---
## `internal/common/datasets/tabular.py`
This module provides the TabularDataset implementation, which handles tabular data like pandas DataFrames

**`TabularDataset`** - Dataset implementation for tabular data.
- `__init__(self, data: pd.DataFrame)`
- `split(self, train_ratio: float, val_ratio: float, test_ratio: float, stratify_column: Optional[str], random_state: Optional[int], is_time_series: bool, time_index_column: Optional[str]) -> Tuple['TabularDataset', 'TabularDataset', 'TabularDataset']` - Split dataset into train, validation and test sets.
- `sample(self, n: int, frac: float, replace: bool, random_state: int) -> 'TabularDataset'` - Sample records from dataset.
- `to_bytes(self) -> bytes` - Serialize the dataset to bytes using Parquet format.
- `from_bytes(cls, data: bytes) -> 'TabularDataset'` - Deserialize bytes back into a TabularDataset.
- `structure(self) -> DatasetStructure` - Return structural metadata for the dataset.
- `to_pandas(self) -> pd.DataFrame` - Return a copy of the dataset as a pandas DataFrame.
- `to_numpy(self) -> np.ndarray` - Convert the dataset to a NumPy array.

---
## `internal/common/provider.py`
This module defines the base class for LLM providers and includes

**`ProviderConfig`** - Configuration class for specifying different LLM providers for various agent roles.
- `__init__(self, default_provider: str, orchestrator_provider: Optional[str], research_provider: Optional[str], engineer_provider: Optional[str], ops_provider: Optional[str], tool_provider: Optional[str])`

**`Provider`** - Base class for LiteLLM provider.
- `__init__(self, model: str)`
- `query(self, system_message: str, user_message: str, response_format: Type[BaseModel], retries: int, backoff: bool) -> str` - Method to query the provider using litellm.completion.

---
## `internal/common/utils/agents.py`
This module provides utilities for working with agents defined using the smolagents library.

**Functions:**
- `get_prompt_templates(base_template_name: str, override_template_name: str) -> dict` - Given the name of a smolagents prompt template (the 'base template') and a plexe prompt template

---
## `internal/common/utils/chain_of_thought/adapters.py`
This module provides adapters for extracting step information from different agent frameworks.

**Functions:**
- `extract_step_summary_from_smolagents(step: Any, agent: Any) -> StepSummary` - Extract step summary from a SmoLAgents step object.

---
## `internal/common/utils/chain_of_thought/callable.py`
This module defines Callables for capturing and formatting agent chain of thought.

**`ChainOfThoughtCallable`** - Callable that captures and formats agent chain of thought.
- `__init__(self, emitter: Optional[ChainOfThoughtEmitter], extractor: StepExtractor)`
- `get_full_chain_of_thought(self) -> List[StepSummary]` - Get the full chain of thought captured so far.
- `clear(self) -> None` - Clear all captured steps.

---
## `internal/common/utils/chain_of_thought/emitters.py`
This module defines Emitters for outputting chain of thought information.

**`ChainOfThoughtEmitter`** - Abstract base class for chain of thought emitters.
- `emit_thought(self, agent_name: str, message: str) -> None` - Emit a thought from an agent.

**`ConsoleEmitter`** - Emitter that outputs chain of thought to the console with rich formatting.
- `__init__(self, output: TextIO)`
- `emit_thought(self, agent_name: str, message: str) -> None` - Emit a thought to the console using Rich tree visualization.

**`LoggingEmitter`** - Emitter that outputs chain of thought to the logging system.
- `__init__(self, level: int)`
- `emit_thought(self, agent_name: str, message: str) -> None` - Emit a thought to the logger.

**`MultiEmitter`** - Emitter that outputs chain of thought to multiple emitters.
- `__init__(self, emitters: List[ChainOfThoughtEmitter])`
- `emit_thought(self, agent_name: str, message: str) -> None` - Emit a thought to all configured emitters.

---
## `internal/common/utils/chain_of_thought/protocol.py`
Defines protocols and data classes for capturing agent reasoning steps.

**`ToolCall`** - Information about a tool called by an agent.

**`StepSummary`** - Framework-agnostic representation of an agent's reasoning step.

**`StepExtractor`** - Protocol for extracting step information from agent frameworks.

---
## `internal/common/utils/dataset_storage.py`
This module provides utilities for dataset storage and transfer across processes.

**Functions:**
- `write_dataset_to_file(dataset: Dataset, path: str) -> None` - Write dataset to a file.
- `read_dataset_from_file(dataset_class: Type[T], path: str) -> T` - Read dataset from a file.
- `dataset_to_shared_memory(dataset: Dataset, name: str) -> None` - Place dataset in shared memory for cross-process access.
- `dataset_from_shared_memory(dataset_class: Type[T], name: str, size: Optional[int]) -> T` - Retrieve dataset from shared memory.

---
## `internal/common/utils/dependency_utils.py`
Utilities for handling optional dependencies.

**Functions:**
- `requires_package(package_name: str, error_message: str) -> Callable[[Callable[..., T]], Callable[..., T]]` - Decorator that checks if a required package is installed before executing a function.

---
## `internal/common/utils/markdown_utils.py`
Utilities for markdown formatting of reports and data.

**Functions:**
- `format_eda_report_markdown(eda_report: Dict[Any, Any]) -> str` - Convert an EDA report dictionary to a well-formatted markdown document.

---
## `internal/common/utils/model_state.py`
Model state definitions for Plexe.

**`ModelState`** - States a model can be in during its lifecycle.

---
## `internal/common/utils/model_utils.py`
This module provides utility functions for working with model descriptions and metadata.

**Functions:**
- `calculate_model_size(artifacts: list) -> Optional[int]` - Calculate the total size of the model artifacts in bytes.
- `format_code_snippet(code: Optional[str]) -> Optional[str]` - Format a code snippet for display, truncating if necessary.

---
## `internal/common/utils/pandas_utils.py`
No description

**Functions:**
- `convert_dtype_to_python(dtype, sample_values) -> str` - Convert a Pandas dtype to a Python type.

---
## `internal/common/utils/prompt_utils.py`
No description

**Functions:**
- `join_task_statement(intent: str, input_schema: Type[BaseModel], output_schema: Type[BaseModel]) -> str` - Join the problem statement into a single string.

---
## `internal/common/utils/pydantic_utils.py`
This module provides utility functions for manipulating Pydantic models.

**Functions:**
- `merge_models(model_name: str, models: List[Type[BaseModel]]) -> Type[BaseModel]` - Merge multiple Pydantic models into a single model. The ordering of the list determines
- `create_model_from_fields(model_name: str, model_fields: dict) -> Type[BaseModel]` - Create a Pydantic model from a dictionary of fields.
- `map_to_basemodel(name: str, schema: dict | Type[BaseModel]) -> Type[BaseModel]` - Ensure that the schema is a Pydantic model or a dictionary, and return the model.
- `format_schema(schema: Type[BaseModel]) -> Dict[str, str]` - Format a schema model into a dictionary representation of field names and types.
- `convert_schema_to_type_dict(schema: Type[BaseModel]) -> Dict[str, type]` - Convert a Pydantic model to a dictionary mapping field names to their Python types.

---
## `internal/common/utils/response.py`
No description

**Functions:**
- `wrap_code(code: str, lang) -> str` - Wraps code with three backticks.
- `is_valid_python_script(script)` - Check if a script is a valid Python script.
- `extract_jsons(text)` - Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects.
- `trim_long_string(string, threshold, k)` - No description
- `extract_code(text)` - Extract python code blocks from the text.
- `extract_text_up_to_code(s)` - Extract (presumed) natural language text up to the start of the first code block.
- `format_code(code) -> str` - Format Python code using Black.
- `extract_performance(output: str) -> float | None` - Extract the performance metric from the output.
- `extract_json_array(text: str) -> str` - Extract a JSON array from an LLM response, handling common formatting issues.
- `json_to_dataframe(text: str) -> 'pd.DataFrame'` - Convert LLM-generated JSON text to a pandas DataFrame.

---
## `internal/datasets/config.py`
This module provides configuration for the data generation service.

**`Config`** - Configuration class for the dataset generation functionality.

---
## `internal/datasets/core/generation/base.py`
This module defines the base class for data generators used in the project.

**`BaseDataGenerator`** - Abstract base class for an object that generates data samples in a given schema.
- `generate(self, intent: str, n_generate: int, schema: Type[BaseModel], existing_data: Optional[pd.DataFrame]) -> pd.DataFrame` - Generate synthetic data for a given problem description.

---
## `internal/datasets/core/generation/simple_llm.py`
No description

**`SimpleLLMDataGenerator`** - Implementation of BaseDataGenerator that uses a straightforward LLM prompting mechanism to generate
- `__init__(self, provider: Provider)`
- `generate(self, intent: str, n_generate: int, schema: Type[BaseModel], existing_data: Optional[pd.DataFrame]) -> pd.DataFrame` - Generate synthetic data based on the given intent, schema, and optionally existing data.

---
## `internal/datasets/core/validation/base.py`
No description

**`BaseDataValidator`** - No description
- `validate(self, report_output_path: str, synthetic_data_path: str, reference_data_path: str, data_schema: dict) -> str` - Validate synthetic data against reference data and schema. The validation results are saved to a report

---
## `internal/datasets/core/validation/eda.py`
No description

**`EdaDataValidator`** - No description
- `validate(self, report_output_path: str, synthetic_data_path: str, reference_data_path: str, data_schema: dict) -> str` - Validates the synthetic data by comparing it to reference data (if available) and producing an

---
## `internal/datasets/generator.py`
No description

**`DatasetGenerator`** - Generate synthetic data based on request parameters.
- `__init__(self, provider: Provider, description: str, schema: Type[BaseModel])`
- `generate(self, n_samples: int, existing_data: pd.DataFrame) -> pd.DataFrame` - Generate synthetic data based on request parameters.

---
## `internal/models/callbacks/chain_of_thought.py`
Chain of Thought model callback for emitting chain of thought information

**`ChainOfThoughtModelCallback`** - Callback that captures and formats the chain of thought for model building.
- `__init__(self, emitter)`
- `on_build_start(self, info: BuildStateInfo) -> None` - Reset the chain of thought at the beginning of the build process.
- `on_build_end(self, info: BuildStateInfo) -> None` - Emit completion message at the end of the build process.
- `on_iteration_start(self, info: BuildStateInfo) -> None` - Emit iteration start message.
- `on_iteration_end(self, info: BuildStateInfo) -> None` - Emit iteration end message with performance metrics.
- `get_chain_of_thought_callable(self)` - Get the underlying chain of thought callable.
- `get_full_chain_of_thought(self) -> List` - Get the full chain of thought captured during model building.

---
## `internal/models/callbacks/checkpoint.py`
Checkpoint callback for model building process in Plexe.

**`ModelCheckpointCallback`** - Callback that saves model state checkpoints during the build process.
- `__init__(self, keep_n_latest: Optional[int], checkpoint_dir: Optional[str], delete_on_success: Optional[bool])`
- `on_build_start(self, info: BuildStateInfo) -> None` - Store reference to the model on build start.
- `on_iteration_end(self, info: BuildStateInfo) -> None` - Create a checkpoint after each iteration.
- `on_build_end(self, info: BuildStateInfo) -> None` - Optionally clean up checkpoints when build completes successfully.

---
## `internal/models/callbacks/mlflow.py`
MLFlow callback for tracking model building process.

**`MLFlowCallback`** - Callback that logs the model building process to MLFlow with hierarchical run organization.
- `__init__(self, tracking_uri: str, experiment_name: str, connect_timeout: int)`
- `on_build_start(self, info: BuildStateInfo) -> None` - Start MLFlow parent run and log initial parameters.
- `on_iteration_start(self, info: BuildStateInfo) -> None` - Start a new nested child run for this iteration.
- `on_iteration_end(self, info: BuildStateInfo) -> None` - Log metrics for this iteration and end the child run.
- `on_build_end(self, info: BuildStateInfo) -> None` - Log final model details and end MLFlow parent run.

---
## `internal/models/entities/artifact.py`
This module defines the "Artifact" dataclass, a simple representation of an external artifact.

**`Artifact`** - Represents a model artifact, which can either be a file path or raw text/binary data.
- `__init__(self, name: str, path: Path, handle: BinaryIO, data: bytes)`
- `is_path(self) -> bool` - True if the artifact is a file path.
- `is_handle(self) -> bool` - True if the artifact is file path or file-like object.
- `is_data(self) -> bool` - True if the artifact is a string or bytes object loaded in memory.
- `get_as_handle(self) -> BinaryIO` - Get the artifact as a file-like object.
- `from_path(path: Union[str, Path])` - Create an Artifact instance from a file path.
- `from_data(name: str, data: bytes)` - Create an Artifact instance from an in-memory sequence of bytes.

---
## `internal/models/entities/code.py`
This module defines a Code dataclass for representing code objects passed around by agents.

**`Code`** - Represents a code object.

---
## `internal/models/entities/description.py`
This module defines dataclasses for structured model descriptions.

**`SchemaInfo`** - Information about the model's input and output schemas.

**`ImplementationInfo`** - Technical information about the model implementation.

**`PerformanceInfo`** - Performance metrics and training data information.

**`CodeInfo`** - Information about the model's source code.

**`ModelDescription`** - A comprehensive description of a model.
- `as_text(self) -> str` - Convert the model description to a formatted text string.
- `as_markdown(self) -> str` - Convert the model description to a markdown string.

---
## `internal/models/entities/metric.py`
Module: plexe/internal/common/dataclasses/metric

**`ComparisonMethod`** - Defines methods for comparing metrics.

**`MetricComparator`** - Encapsulates comparison logic for metrics.
- `__init__(self, comparison_method: ComparisonMethod, target: float, epsilon: float)`
- `compare(self, value1: float, value2: float) -> int` - Compare two metric values based on the defined comparison method.

**`Metric`** - Represents a metric with a name, a value, and a comparator for determining which metric is better.
- `__init__(self, name: str, value: float, comparator: MetricComparator, is_worst: bool)`
- `is_valid(self) -> bool` - Check if the metric value is valid (i.e., not None or NaN).

---
## `internal/models/execution/docker_executor.py`
No description

**`DockerExecutor`** - Execute Python code snippets in an isolated Docker container.
- `__init__(self, code: str, timeout: int) -> None`
- `run(self) -> ExecutionResult` - No description
- `cleanup(self) -> None` - No description

---
## `internal/models/execution/executor.py`
No description

**`ExecutionResult`** - Result of executing code in an environment.
- `is_valid_performance(self) -> bool` - Validate if performance metric is usable.

**`Executor`** - Abstract base class for code execution environments.
- `__init__(self, code: str, timeout: int) -> None`
- `run(self) -> ExecutionResult` - Execute the code in the defined environment.
- `cleanup(self) -> None` - Perform any necessary cleanup (e.g., terminate processes, remove temporary files).

---
## `internal/models/execution/process_executor.py`
Module: ProcessExecutor for Isolated Python Code Execution

**`ProcessExecutor`** - Execute Python code snippets in an isolated process.
- `__init__(self, execution_id: str, code: str, working_dir: Path | str, datasets: Dict[str, TabularConvertible], timeout: int, code_execution_file_name: str)`
- `run(self) -> ExecutionResult` - Execute code in a subprocess and return results.
- `cleanup(self)` - Clean up resources after execution while preserving model artifacts.

---
## `internal/models/execution/ray_executor.py`
Module: RayExecutor for Distributed Python Code Execution

**`RayExecutor`** - Execute Python code snippets on a Ray cluster.
- `__init__(self, execution_id: str, code: str, working_dir: Path | str, datasets: Dict[str, TabularConvertible], timeout: int, code_execution_file_name: str)`
- `run(self) -> ExecutionResult` - Execute code using Ray and return results.
- `cleanup(self) -> None` - Clean up Ray resources if needed.

---
## `internal/models/generation/planning.py`
This module provides functions and classes for generating and planning solutions for machine learning problems.

**`SolutionPlanGenerator`** - A class to generate solution plans for given problem statements.
- `__init__(self, provider: Provider)`
- `select_target_metric(self, problem_statement: str) -> Metric` - Selects the metric to optimise for the given problem statement and dataset.

---
## `internal/models/generation/review.py`
This module provides functionality for reviewing and analyzing generated models.

**`ModelReviewResponse`** - Response model for the model review operation.

**`ModelReviewer`** - A class for analyzing and reviewing generated models.
- `__init__(self, provider: Provider)`
- `review_model(self, intent: str, input_schema: Dict[str, str], output_schema: Dict[str, str], solution_plan: str, training_code: str, inference_code: str) -> Dict[str, str]` - Review a generated model to extract metadata, explanations and insights about the trained model.

---
## `internal/models/generation/training.py`
This module provides functions and classes for generating, fixing, and reviewing machine learning model training code.

**`TrainingCodeGenerator`** - A class to generate, fix, and review machine learning model training code.
- `__init__(self, provider: Provider)`
- `generate_training_code(self, problem_statement: str, plan: str, train_dataset_names: list[str], validation_dataset_names: list[str]) -> str` - Generates machine learning model training code based on the given problem statement and solution plan.
- `fix_training_code(self, training_code: str, plan: str, review: str, train_dataset_names: list[str], validation_dataset_names: list[str], problems: str) -> str` - Fixes the machine learning model training code based on the review and identified problems.
- `review_training_code(self, training_code: str, problem_statement: str, plan: str, problems: str) -> str` - Reviews the machine learning model training code to identify improvements and fix issues.
- `generate_training_tests(self, problem_statement: str, plan: str, training_code: str) -> str` - No description
- `fix_training_tests(self, training_tests: str, training_code: str, review: str, problems: str) -> str` - No description
- `review_training_tests(self, training_tests: str, training_code: str, problem_statement: str, plan: str) -> str` - No description

---
## `internal/models/validation/composite.py`
This module defines the `CompositeValidator` class, which chains multiple validators together in a workflow.

**`CompositeValidator`** - A validator that chains multiple validators together in a workflow.
- `__init__(self, name: str, validators: list[Validator])`
- `validate(self, code: str) -> ValidationResult` - Validates the given code by running it through each validator in the pipeline.

---
## `internal/models/validation/composites/inference.py`
This module defines a composite validator for validating the correctness of prediction code.

**`InferenceCodeValidator`** - A validator class that validates the correctness of prediction code.
- `__init__(self, input_schema: Type[BaseModel], output_schema: Type[BaseModel], input_sample: List[Dict[str, Any]])`

---
## `internal/models/validation/composites/training.py`
This module defines a composite validator for validating the correctness of training code.

**`TrainingCodeValidator`** - A validator class that validates the correctness of training code.
- `__init__(self)`

---
## `internal/models/validation/primitives/predict.py`
This module defines the `PredictorValidator` class, which validates that a predictor behaves as expected.

**`PredictorValidator`** - A validator class that checks that a predictor behaves as expected.
- `__init__(self, input_schema: Type[BaseModel], output_schema: Type[BaseModel], sample: List[Dict[str, Any]]) -> None`
- `validate(self, code: str, model_artifacts) -> ValidationResult` - Validates that the given code for a predictor behaves as expected.

---
## `internal/models/validation/primitives/security.py`
This module defines the SecurityValidator class, which is responsible for validating the security

**`SecurityValidator`** - A validator class that checks the security of Python code using the Bandit tool.
- `__init__(self)`
- `validate(self, code: str) -> ValidationResult` - Validate the generated code for security vulnerabilities using the Bandit tool.

---
## `internal/models/validation/primitives/syntax.py`
This module defines the SyntaxValidator class, which is responsible for validating the syntax

**`SyntaxValidator`** - A validator class that checks the syntax of Python code using the AST module.
- `__init__(self)`
- `validate(self, code: str) -> ValidationResult` - Validate Python code using AST.

---
## `internal/models/validation/validator.py`
This module defines the `Validator` abstract base class and the `ValidationResult` data class.

**`ValidationResult`** - Represents the result of a validation.

**`Validator`** - Abstract base class for validators.
- `__init__(self, name: str)`
- `validate(self, code: str) -> ValidationResult` - Validates the given code.

---
## `internal/schemas/resolver.py`
Module for schema generation and handling.

**`SchemaResolver`** - A utility class for resolving input and output schemas for a given intent and dataset.
- `__init__(self, provider: Provider, intent: str, input_schema: Type[BaseModel], output_schema: Type[BaseModel])`
- `resolve(self, datasets: Dict[str, TabularConvertible]) -> Tuple[Type[BaseModel], Type[BaseModel]]` - Resolve the input and output schemas for a given intent and dataset.

---
## `main.py`
Application entry point for using the plexe package as a conversational agent.

**Functions:**
- `main()` - Launch the Plexe assistant with a web UI.

---
## `model_builder.py`
ModelBuilder for creating ML models through agentic workflows.

**`ModelBuilder`** - Factory for creating ML models through agentic workflows.
- `__init__(self, provider: str | ProviderConfig, verbose: bool, distributed: bool, working_dir: Optional[str])`
- `build(self, intent: str, datasets: List[pd.DataFrame | DatasetGenerator], input_schema: Type[BaseModel] | Dict[str, type], output_schema: Type[BaseModel] | Dict[str, type], timeout: int, max_iterations: int, run_timeout: int, callbacks: List[Callback], enable_checkpointing: bool)` - Build a complete ML model using the agentic workflow.

---
## `models.py`
This module defines the `Model` class, which represents a machine learning model.

**`Model`** - Represents a model that transforms inputs to outputs according to a specified intent.
- `__init__(self, intent: str, input_schema: Type[BaseModel] | Dict[str, type], output_schema: Type[BaseModel] | Dict[str, type], distributed: bool)`
- `build(self, datasets: List[pd.DataFrame | DatasetGenerator], provider: str | ProviderConfig, timeout: int, max_iterations: int, run_timeout: int, callbacks: List[Callback], verbose: bool, enable_checkpointing: bool) -> None` - Build the model using the provided dataset and optional data generation configuration.
- `predict(self, x: Dict[str, Any], validate_input: bool, validate_output: bool) -> Dict[str, Any]` - Call the model with input x and return the output.
- `get_state(self) -> ModelState` - Return the current state of the model.
- `get_metadata(self) -> dict` - Return metadata about the model.
- `get_metrics(self) -> dict` - Return metrics about the model.
- `describe(self) -> ModelDescription` - Return a structured description of the model.

---
## `server.py`
FastAPI server for the Plexe conversational agent.

**Functions:**
- `async root()` - Serve the main HTML page.
- `async websocket_endpoint(websocket: WebSocket)` - WebSocket endpoint for real-time chat communication.
- `async health_check()` - Health check endpoint.

---
## `templates/models/feature_transformer.tmpl.py`
No description

**`FeatureTransformerImplementation`** - No description
- `transform(self, inputs: pd.DataFrame) -> pd.DataFrame` - Given a DataFrame representing a raw dataset, applies feature transformations to the

---
## `templates/models/predictor.tmpl.py`
No description

**`PredictorImplementation`** - No description
- `__init__(self, artifacts: List[Artifact])`
- `predict(self, inputs: dict) -> dict` - Given an input conforming to the input schema, return the model's prediction

---
## `tools/code_analysis.py`
Tools for analyzing and inspecting code.

**Functions:**
- `read_training_code(training_code_id: str) -> str` - Retrieves the training code from the registry for analysis. Use this tool to understand the
- `get_feature_transformer_code() -> Optional[str]` - Get the feature transformation code that was used to transform the raw input dataset into the

---
## `tools/context.py`
Tools for providing context to agents for code generation tasks.

**Functions:**
- `get_inference_context_tool(llm_to_use: str) -> Callable` - Returns a tool function to get inference context with the model ID pre-filled.

---
## `tools/conversation.py`
Tools for conversational model definition and build initiation.

**Functions:**
- `validate_dataset_files(file_paths: List[str]) -> Dict[str, Dict]` - Check if specified file paths can be read as datasets using pandas.
- `initiate_model_build(intent: str, dataset_file_paths: List[str], input_schema: Optional[Dict], output_schema: Optional[Dict], n_solutions_to_try: int) -> Dict[str, str]` - Initiate a model build by loading datasets from file paths and starting the build process.

---
## `tools/datasets.py`
Tools for dataset manipulation, splitting, and registration.

**Functions:**
- `register_split_datasets(dataset_name: str, train_dataset: pd.DataFrame, validation_dataset: pd.DataFrame, test_dataset: pd.DataFrame, splitting_code: str) -> Dict[str, str]` - Register train, validation, and test datasets in the object registry after custom splitting.
- `create_input_sample(n_samples: int) -> bool` - Create and register a synthetic sample input dataset that matches the model's input schema.
- `drop_null_columns(dataset_name: str) -> str` - Drop all columns from the dataset that are completely null and register the modified dataset.
- `get_dataset_preview(dataset_name: str) -> Dict[str, Any]` - Generate a concise preview of a dataset with statistical information to help agents understand the data.
- `register_eda_report(dataset_name: str, overview: Dict[str, Any], feature_engineering_opportunities: Dict[str, Any], data_quality_challenges: Dict[str, Any], data_preprocessing_requirements: Dict[str, Any], feature_importance: Dict[str, Any], insights: List[str], recommendations: List[str]) -> str` - Register an exploratory data analysis (EDA) report for a dataset in the Object Registry.
- `register_feature_engineering_report(dataset_name: str, overview: Dict[str, Any], feature_catalog: Dict[str, Any], feature_importance: Dict[str, Any], insights: List[str], recommendations: List[str]) -> str` - Register a feature engineering report for a transformed dataset. This tool registers a structured report with
- `get_latest_datasets() -> Dict[str, str]` - Get the most recent version of each dataset in the pipeline. Automatically detects transformed
- `get_dataset_for_splitting() -> str` - Get the most appropriate dataset for splitting. Returns transformed version if available,
- `get_training_datasets() -> Dict[str, str]` - Get datasets ready for model training.
- `get_test_dataset() -> str` - Get the name of the test dataset for final model evaluation.
- `get_dataset_reports() -> Dict[str, Dict]` - Get all available data analysis reports, including EDA for raw datasets and feature engineering reports

---
## `tools/evaluation.py`
This module defines agent tools for evaluating the properties and performance of models.

**Functions:**
- `get_review_finalised_model(llm_to_use: str) -> Callable` - Returns a tool function to review finalized models with the model ID pre-filled.
- `get_solution_performances() -> Dict[str, float]` - Returns the performance of all successfully trained solutions so far. The performances are returned as a dictionary

---
## `tools/execution.py`
Tools related to code execution, including running training code in isolated environments and

**Functions:**
- `get_executor_tool(distributed: bool) -> Callable` - Get the appropriate executor tool based on the distributed flag.
- `apply_feature_transformer(dataset_name: str) -> Dict` - Applies a feature transformer to datasets and registers the transformed datasets. The name of the
- `get_model_artifacts() -> List[str]` - Get all registered model artifact names.

---
## `tools/metrics.py`
Tools related to metrics selection and model review/metadata extraction.

**Functions:**
- `get_select_target_metric(llm_to_use: str) -> Callable` - Returns a tool function to select target metrics with the model ID pre-filled.

---
## `tools/response_formatting.py`
This module provides tools for forcing an agent to return its response in a specific format.

**Functions:**
- `format_final_orchestrator_agent_response(best_solution_id: str, performance_metric_name: str, performance_metric_value: float, performance_metric_comparison_method: str, model_review_output: Dict[str, str]) -> dict` - Returns a dictionary containing the exact fields that the agent must return in its final response. The purpose
- `format_final_mle_agent_response(solution_id: str, execution_success: bool, performance_value: Optional[float], exception: Optional[str], model_artifact_names: Optional[List[str]]) -> dict` - Returns a dictionary containing the exact fields that the agent must return in its final response. The fields
- `format_final_mlops_agent_response(inference_code_id: str) -> dict` - Returns a dictionary containing the exact fields that the agent must return in its final response.

---
## `tools/schemas.py`
Tools for schema inference, definition, and validation.

**Functions:**
- `register_global_schemas(input_schema: Dict[str, str], output_schema: Dict[str, str], reasoning: str) -> Dict[str, str]` - Register input and output schemas that should be used by all models built for all solutions.
- `get_dataset_schema(dataset_name: str) -> Dict[str, Any]` - Extract the schema (column names and types) from a dataset. This is useful for understanding the structure
- `get_global_schemas() -> Dict[str, Dict[str, str]]` - Get global input and output schemas that should apply to a model.
- `register_solution_schemas(solution_id: str, input_schema: Dict[str, str], output_schema: Dict[str, str], reasoning: str) -> Dict[str, str]` - Register input and output schemas for a specific solution.
- `get_solution_schemas(solution_id: str) -> Dict[str, Dict[str, str]]` - Get schemas for a specific solution, with fallback to global schemas.

---
## `tools/solutions.py`
Tools for creating and managing Solution objects in the ML workflow.

**Functions:**
- `get_solution_creation_tool(max_solutions: int)` - Returns a tool function to create a new Solution object with a plan.
- `get_solution_plan_by_id(solution_id: str) -> str` - Retrieves a model solution plan by its ID.
- `list_solutions() -> List[str]` - Lists all Solution IDs currently available. Use this tool to see all available solutions if you run into

---
## `tools/testing.py`
Tools for model testing and evaluation.

**Functions:**
- `register_testing_code(solution_id: str, testing_code: str) -> str` - Register the testing/evaluation code in the object registry and update the Solution object. The testing code
- `register_evaluation_report(solution_id: str, model_performance_summary: Dict, detailed_metrics: Dict, quality_analysis: Dict, recommendations: List[str], testing_insights: List[str]) -> str` - Register comprehensive evaluation report in the object registry and link to Solution.

---
## `tools/training.py`
Tools related to code generation, including solution planning, training code,

**Functions:**
- `register_best_solution(best_solution_id: str) -> str` - Register the solution with the best performance as the final selected solution in the object
- `get_training_code_generation_tool(llm_to_use: str) -> Callable` - Returns a tool function to generate training code with the model ID pre-filled.
- `get_training_code_fixing_tool(llm_to_use: str) -> Callable` - Returns a tool function to fix training code with the model ID pre-filled.

---
## `tools/validation.py`
Tools related to code validation, including syntax and security checks.

**Functions:**
- `validate_training_code(training_code: str) -> Dict` - Validates training code for syntax and security issues.
- `validate_inference_code(solution_id: str, inference_code: str) -> Dict` - Validates inference code for syntax, security, and correctness, and updates the Solution object.
- `validate_feature_transformations(transformation_code: str) -> Dict` - Validates feature transformation code for syntax correctness and implementation

---