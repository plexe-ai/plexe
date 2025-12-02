# Complete Model Building Walkthrough: House Price Prediction Example

This document provides a step-by-step walkthrough of how the Plexe system builds a machine learning model, tracing every file and code path from user input to final prediction.

## Example Scenario
**User Goal**: Build a model to predict house prices based on features like bedrooms, bathrooms, and square footage.

**User Code**:
```python
from plexe import ModelBuilder
import pandas as pd

# Load dataset
df = pd.read_csv("houses.csv")  # Contains: bedrooms, bathrooms, square_footage, price

# Build model
builder = ModelBuilder(provider="openai/gpt-4o-mini", verbose=True)
model = builder.build(
    intent="Given a dataset of house features, predict the house price.",
    datasets=[df],
    input_schema={"bedrooms": int, "bathrooms": int, "square_footage": float},
    output_schema={"price": float},
    max_iterations=3
)

# Make prediction
prediction = model.predict({"bedrooms": 3, "bathrooms": 2, "square_footage": 1500.0})
print(prediction)  # {"price": 250000.0}
```

---

## Phase 1: User Initialization

### Step 1.1: User Creates ModelBuilder
**File**: `plexe/model_builder.py` (lines 37-56)

**What Happens**:
- User calls `ModelBuilder(provider="openai/gpt-4o-mini", verbose=True)`
- `__init__` method executes:
  - Creates `ProviderConfig` from provider string (line 53)
  - Sets `self.verbose = True`
  - Sets `self.distributed = False`
  - Calls `_create_working_dir()` which creates `./workdir/run-{timestamp}/` directory
  - Working directory path stored in `self.working_dir`

**Output**: ModelBuilder instance ready with provider config and working directory

---

### Step 1.2: User Calls build() Method
**File**: `plexe/model_builder.py` (lines 66-254)

**What Happens**:
1. **Registry Initialization** (lines 95-97):
   - Gets singleton `ObjectRegistry()` instance
   - Clears registry: `object_registry.clear()`
   - Registry is now empty and ready

2. **Parameter Validation** (lines 99-103):
   - Checks `timeout` or `max_iterations` is set
   - Validates `run_timeout <= timeout` if both provided

3. **Schema Processing** (lines 105-110):
   - Calls `map_to_basemodel("in", input_schema)` from `plexe/internal/common/utils/pydantic_utils.py`
   - **File**: `pydantic_utils.py` (lines 51-102):
     - Converts dict `{"bedrooms": int, ...}` to Pydantic `BaseModel` class
     - Creates model named "in" with fields: bedrooms (int), bathrooms (int), square_footage (float)
     - Validates types are allowed (int, float, str, bool, List variants)
   - Same process for `output_schema` → creates "out" model with `price: float`
   - Registers schema locks in registry:
     - `object_registry.register(bool, "input_schema_is_locked", True, immutable=True)`
     - `object_registry.register(bool, "output_schema_is_locked", True, immutable=True)`

4. **Callback Setup** (lines 112-122):
   - Initializes callbacks list
   - Adds `ModelCheckpointCallback` if `enable_checkpointing=True`
   - Creates `ChainOfThoughtModelCallback` with `ConsoleEmitter`
   - Registers all callbacks in ObjectRegistry

5. **Dataset Registration** (lines 125-136):
   - Converts pandas DataFrame to `DatasetAdapter` via `DatasetAdapter.coerce(df)`
   - **File**: `plexe/internal/common/datasets/adapter.py`
   - Creates `TabularDataset` wrapper around DataFrame
   - Registers as `dataset_0` in ObjectRegistry with `immutable=True`
   - Formats schemas using `format_schema()` and registers in registry:
     - `object_registry.register(dict, "input_schema", {"bedrooms": "int", ...})`
     - `object_registry.register(dict, "output_schema", {"price": "float"})`

6. **Model Identifier Generation** (line 139):
   - Creates unique ID: `model-{timestamp}`

7. **Callback Notification - Build Start** (lines 141-151):
   - Calls `_notify_callbacks()` with event="build_start"
   - All registered callbacks receive `BuildStateInfo` with intent, schemas, datasets

8. **Agent Creation** (lines 154-165):
   - Creates `PlexeAgent` instance (this is the multi-agent orchestrator)
   - **File**: `plexe/agents/agents.py` (lines 63-194)

---

## Phase 2: Multi-Agent System Initialization

### Step 2.1: PlexeAgent Initialization
**File**: `plexe/agents/agents.py` (lines 63-194)

**What Happens**:
The `PlexeAgent.__init__` creates all specialist agents:

1. **ML Research Agent** (lines 105-110):
   - **File**: `plexe/agents/model_planner.py`
   - Creates `ModelPlannerAgent` with `ml_researcher_model_id`
   - This agent will plan ML solution approaches

2. **Schema Resolver Agent** (lines 113-117):
   - **File**: `plexe/agents/schema_resolver.py`
   - Creates `SchemaResolverAgent` with `orchestrator_model_id`
   - Will infer schemas if not provided (already provided in our case)

3. **EDA Agent** (lines 120-124):
   - **File**: `plexe/agents/dataset_analyser.py`
   - Creates `EdaAgent` with `orchestrator_model_id`
   - Will analyze dataset structure and generate insights

4. **Feature Engineering Agent** (lines 127-131):
   - **File**: `plexe/agents/feature_engineer.py`
   - Creates `FeatureEngineeringAgent` with `ml_engineer_model_id`
   - Will transform raw data into better features

5. **Dataset Splitter Agent** (lines 134-138):
   - **File**: `plexe/agents/dataset_splitter.py`
   - Creates `DatasetSplitterAgent` with `orchestrator_model_id`
   - Will split data into train/validation/test sets

6. **Model Trainer Agent** (lines 141-148):
   - **File**: `plexe/agents/model_trainer.py`
   - Creates `ModelTrainerAgent` with `ml_engineer_model_id` and `tool_model_id`
   - Will implement and execute training code

7. **Model Packager Agent** (lines 151-157):
   - **File**: `plexe/agents/model_packager.py`
   - Creates `ModelPackagerAgent` with `ml_ops_engineer_model_id`
   - Will generate inference/prediction code

8. **Model Tester Agent** (lines 160-164):
   - **File**: `plexe/agents/model_tester.py`
   - Creates `ModelTesterAgent` with `ml_engineer_model_id`
   - Will test and evaluate the final model

9. **Orchestrator Agent** (lines 167-194):
   - Creates `CodeAgent` (from smolagents library) named "Orchestrator"
   - Configures with tools:
     - `get_select_target_metric()` - selects evaluation metric
     - `get_review_finalised_model()` - reviews final model
     - `get_latest_datasets` - retrieves datasets from registry
     - `get_solution_performances` - gets performance metrics
     - `register_best_solution` - marks best solution
     - `format_final_orchestrator_agent_response` - formats final response
   - Registers all specialist agents as `managed_agents`
   - Sets `max_steps=30`, `planning_interval=7`

**Output**: Fully initialized multi-agent system with 8 specialist agents + 1 orchestrator

---

### Step 2.2: Agent Prompt Generation
**File**: `plexe/model_builder.py` (lines 167-185)

**What Happens**:
- Calls `prompt_templates.agent_builder_prompt()` with:
  - intent: "Given a dataset of house features, predict the house price."
  - input_schema: JSON string of schema dict
  - output_schema: JSON string of schema dict
  - datasets: ["dataset_0"]
  - working_dir: "./workdir/run-{timestamp}/"
  - max_iterations: 3
- **File**: `plexe/config/prompt_templates.py` (or similar)
- Generates comprehensive prompt instructing orchestrator on the task
- Creates `additional_args` dict with all context

---

## Phase 3: Orchestrator Agent Execution

### Step 3.1: Orchestrator Starts Workflow
**File**: `plexe/agents/agents.py` (lines 196-204)

**What Happens**:
- Calls `agent.run(agent_prompt, additional_args=additional_args)`
- Orchestrator (CodeAgent) begins executing steps
- Uses planning every 7 steps to decide next actions
- Can call tools or delegate to specialist agents

**Typical Orchestrator Workflow**:
1. Select target metric (e.g., RMSE for regression)
2. Analyze dataset (delegate to EDA agent)
3. Resolve schemas if needed (delegate to SchemaResolver - skipped if provided)
4. Plan solutions (delegate to ML Research agent)
5. Split dataset (delegate to Dataset Splitter agent)
6. Train models (delegate to ML Engineer agent)
7. Package models (delegate to ML Ops agent)
8. Test models (delegate to Model Tester agent)
9. Select best solution

---

## Phase 4: Specialist Agent Execution

### Step 4.1: Target Metric Selection
**Tool**: `get_select_target_metric()` from `plexe/tools/metrics.py`

**What Happens**:
- Orchestrator calls this tool
- Tool analyzes intent and output schema
- Determines this is a regression problem (output is float)
- Selects appropriate metric (e.g., RMSE, MAE)
- Registers metric in ObjectRegistry

---

### Step 4.2: EDA Agent Execution
**File**: `plexe/agents/dataset_analyser.py`

**What Happens**:
1. Orchestrator delegates task: "Analyze dataset_0"
2. EDA Agent receives task
3. Agent calls `get_latest_datasets()` tool:
   - **File**: `plexe/tools/datasets.py` (lines 394-451)
   - Retrieves `dataset_0` from ObjectRegistry
   - Returns dict: `{"raw": "dataset_0"}`
4. Agent calls `get_dataset_preview()` tool:
   - **File**: `plexe/tools/datasets.py` (lines 224-283)
   - Gets dataset from registry: `object_registry.get(TabularConvertible, "dataset_0")`
   - Converts to pandas: `dataset.to_pandas()`
   - Generates preview with shape, dtypes, sample rows, statistics, missing values
   - Returns preview dict
5. Agent analyzes data using LLM (via Provider):
   - **File**: `plexe/internal/common/provider.py`
   - Makes LLM call with system prompt + data preview
   - LLM generates EDA insights
6. Agent calls `register_eda_report()` tool:
   - **File**: `plexe/tools/datasets.py` (lines 287-341)
   - Creates structured EDA report dict with:
     - overview: dataset stats, target variable analysis
     - feature_engineering_opportunities: transformation needs
     - data_quality_challenges: data issues
     - data_preprocessing_requirements: preprocessing steps
     - feature_importance: predictive potential
     - insights: key findings
     - recommendations: actionable steps
   - Registers in ObjectRegistry: `object_registry.register(dict, "eda_report_dataset_0", report)`

**Output**: EDA report registered in ObjectRegistry

---

### Step 4.3: Schema Resolution (Skipped)
**File**: `plexe/agents/schema_resolver.py`

**What Happens**:
- Orchestrator checks if schemas are locked
- Finds `input_schema_is_locked = True` in registry
- Skips schema resolution (schemas already provided)
- If schemas weren't provided, this agent would:
  - Analyze intent and sample data
  - Infer input/output schemas
  - Register them in ObjectRegistry

---

### Step 4.4: ML Research Agent - Solution Planning
**File**: `plexe/agents/model_planner.py`

**What Happens**:
1. Orchestrator delegates: "Plan ML solutions for house price prediction"
2. Agent retrieves context:
   - Gets EDA report: `object_registry.get(dict, "eda_report_dataset_0")`
   - Gets schemas from registry
   - Gets datasets: `get_latest_datasets()`
3. Agent uses LLM to generate solution plans:
   - Analyzes problem type (regression)
   - Considers dataset characteristics from EDA
   - Generates multiple solution approaches:
     - Solution 1: Linear Regression
     - Solution 2: Random Forest
     - Solution 3: Gradient Boosting (XGBoost)
   - Each solution includes:
     - Model type and framework
     - Rationale
     - Expected performance
     - Implementation approach
4. Agent registers solution plans in ObjectRegistry
5. **File**: `plexe/core/entities/solution.py` - Solution objects created

**Output**: Multiple solution plans registered

---

### Step 4.5: Feature Engineering Agent (Optional)
**File**: `plexe/agents/feature_engineer.py`

**What Happens** (if orchestrator decides feature engineering is needed):
1. Agent retrieves EDA report and raw dataset
2. Analyzes feature engineering opportunities from EDA
3. Generates transformation code (e.g., normalization, encoding)
4. Applies transformations to create new dataset
5. Registers transformed dataset: `dataset_0_transformed`
6. Registers transformation code in ObjectRegistry
7. Calls `register_feature_engineering_report()`:
   - **File**: `plexe/tools/datasets.py` (lines 344-390)
   - Creates feature engineering report
   - Registers in ObjectRegistry

**Output**: Transformed dataset and feature engineering code (if applicable)

---

### Step 4.6: Dataset Splitter Agent
**File**: `plexe/agents/dataset_splitter.py`

**What Happens**:
1. Orchestrator delegates: "Split dataset into train/validation/test"
2. Agent calls `get_dataset_for_splitting()`:
   - **File**: `plexe/tools/datasets.py` (lines 455-507)
   - Finds best dataset (prefers transformed, falls back to raw)
   - Returns "dataset_0" (or "dataset_0_transformed" if exists)
3. Agent retrieves dataset from registry
4. Agent generates splitting strategy using LLM:
   - Analyzes dataset size and characteristics
   - Decides split ratios (e.g., 70/15/15 or 80/10/10)
   - Considers time-series vs random split needs
5. Agent generates splitting code (Python)
6. Agent executes splitting code to create splits
7. Agent calls `register_split_datasets()`:
   - **File**: `plexe/tools/datasets.py` (lines 27-92)
   - Registers three datasets:
     - `dataset_0_train` (TabularConvertible)
     - `dataset_0_val` (TabularConvertible)
     - `dataset_0_test` (TabularConvertible)
   - Registers splitting code: `object_registry.register(Code, "dataset_splitting_code", Code(code))`

**Output**: Train/validation/test datasets registered in ObjectRegistry

---

### Step 4.7: Model Trainer Agent - Training Implementation
**File**: `plexe/agents/model_trainer.py`

**What Happens** (for each solution plan):
1. Orchestrator delegates: "Train Solution 1: Linear Regression"
2. Agent retrieves:
   - Training datasets: `get_training_datasets()` → `{"train": "dataset_0_train", "validation": "dataset_0_val"}`
   - Solution plan from registry
   - EDA report and feature engineering report (if available)
3. Agent generates training code using LLM:
   - **File**: Uses `tool_model_id` for code generation
   - Creates Python code that:
     - Loads train/val datasets
     - Implements model (e.g., sklearn LinearRegression)
     - Trains model on training set
     - Evaluates on validation set
     - Saves model artifacts
   - Code is validated and formatted
4. Agent executes training code:
   - **File**: `plexe/tools/execution.py`
   - Code runs in isolated environment
   - Model trains on `dataset_0_train`
   - Validates on `dataset_0_val`
   - Generates performance metric (e.g., RMSE = 25000.0)
   - Saves model artifacts (pickle files, weights, etc.)
5. Agent registers results:
   - **File**: `plexe/tools/training.py`
   - Creates `Solution` object with:
     - training_code: generated Python code string
     - performance: Metric object (name="RMSE", value=25000.0)
     - model_artifacts: List[Artifact] (file paths to saved models)
   - Registers Solution in ObjectRegistry
6. Process repeats for each solution (Solution 2: Random Forest, Solution 3: XGBoost)

**Output**: Multiple trained solutions with performance metrics in ObjectRegistry

---

### Step 4.8: Model Packager Agent - Inference Code Generation
**File**: `plexe/agents/model_packager.py`

**What Happens** (for each solution):
1. Orchestrator delegates: "Generate inference code for Solution 1"
2. Agent retrieves:
   - Solution object with training code and artifacts
   - Input/output schemas from registry
   - Feature transformer code (if exists)
3. Agent generates inference code using LLM:
   - Creates Python class `PredictorImplementation` that:
     - Inherits from `Predictor` interface
     - **File**: `plexe/core/interfaces/predictor.py`
     - Implements `__init__(self, artifacts: List[Artifact])`
     - Implements `predict(self, inputs: dict) -> dict`
   - Code loads model artifacts
   - Applies feature transformations (if any)
   - Makes predictions
   - Returns dict matching output schema
4. Agent validates inference code:
   - **File**: `plexe/tools/validation.py`
   - Creates sample inputs using `create_input_sample()`
   - **File**: `plexe/tools/datasets.py` (lines 97-151)
   - Tests prediction on sample inputs
   - Validates output matches schema
5. Agent registers inference code:
   - Updates Solution object with `inference_code` attribute
   - Code stored as string in Solution

**Output**: Inference code for each solution registered

---

### Step 4.9: Model Tester Agent - Final Evaluation
**File**: `plexe/agents/model_tester.py`

**What Happens**:
1. Orchestrator delegates: "Test all solutions on test set"
2. Agent retrieves:
   - Test dataset: `get_test_dataset()` → "dataset_0_test"
   - All solution objects from registry
3. For each solution:
   - Agent generates testing code
   - Code loads model and inference code
   - Runs predictions on entire test set
   - Calculates final performance metrics
   - Generates evaluation report (confusion matrix, feature importance, etc.)
4. Agent updates Solution objects with:
   - `testing_code`: testing code string
   - `model_evaluation_report`: detailed evaluation dict
   - Test performance metrics

**Output**: All solutions tested and evaluated

---

## Phase 5: Solution Selection and Finalization

### Step 5.1: Best Solution Selection
**Tool**: `get_solution_performances()` from `plexe/tools/evaluation.py`

**What Happens**:
1. Orchestrator calls tool to get all solution performances
2. Tool retrieves all Solution objects from ObjectRegistry
3. Extracts performance metrics from each
4. Returns comparison of all solutions

**Tool**: `register_best_solution()` from `plexe/tools/training.py`

**What Happens**:
1. Orchestrator compares all solutions
2. Selects best based on metric (e.g., lowest RMSE)
3. Registers best solution: `object_registry.register(Solution, "best_performing_solution", best_solution)`

---

### Step 5.2: Final Response Formatting
**Tool**: `format_final_orchestrator_agent_response()` from `plexe/tools/response_formatting.py`

**What Happens**:
1. Orchestrator calls tool to format final response
2. Tool collects:
   - Best solution details
   - Performance metrics
   - Model metadata
3. Formats into structured response dict
4. Returns to orchestrator

---

### Step 5.3: PlexeAgent.run() Completion
**File**: `plexe/agents/agents.py` (lines 196-305)

**What Happens**:
1. Orchestrator finishes execution
2. `agent.run()` returns result (AgentText or dict)
3. Code extracts best solution (lines 217-219):
   - `best_solution = object_registry.get(Solution, "best_performing_solution")`
   - `training_code = best_solution.training_code`
   - `inference_code = best_solution.inference_code`
4. Extracts performance metrics (lines 222-245):
   - Gets metric from result or solution
   - Creates `Metric` object with name, value, comparator
5. Compiles inference code (lines 251-255):
   - Creates new module: `types.ModuleType("predictor")`
   - Executes inference code string in module namespace
   - Gets `PredictorImplementation` class from module
   - Instantiates: `predictor = predictor_class(best_solution.model_artifacts)`
6. Extracts additional code (lines 258-291):
   - Feature transformer code (if exists)
   - Dataset split code (if exists)
   - Testing code (if exists)
   - Evaluation report (if exists)
7. Creates `ModelGenerationResult` (lines 293-305):
   - training_source_code: Python code string
   - inference_source_code: Python code string
   - feature_transformer_source_code: Python code string (or None)
   - dataset_split_code: Python code string (or None)
   - predictor: Predictor instance (ready to use)
   - model_artifacts: List[Artifact]
   - performance: Metric object
   - test_performance: Metric object
   - testing_source_code: Python code string (or None)
   - evaluation_report: Dict (or None)
   - metadata: Dict with model info

**Output**: `ModelGenerationResult` object returned to ModelBuilder

---

## Phase 6: ModelBuilder Finalization

### Step 6.1: Extract Final Schemas
**File**: `plexe/model_builder.py` (lines 189-192)

**What Happens**:
- Calls `get_solution_schemas("best_performing_solution")`
- **File**: `plexe/tools/schemas.py`
- Retrieves final input/output schemas (may have been refined)
- Converts to Pydantic models using `map_to_basemodel()`

---

### Step 6.2: Build Metadata
**File**: `plexe/model_builder.py` (lines 194-217)

**What Happens**:
1. Creates metadata dict with provider info
2. Updates with metadata from `ModelGenerationResult`
3. Extracts EDA reports from ObjectRegistry:
   - Gets `eda_report_dataset_0` from registry
   - Formats as markdown using `format_eda_report_markdown()`
   - Adds to metadata

---

### Step 6.3: Create Model Instance
**File**: `plexe/model_builder.py` (lines 219-236)

**What Happens**:
1. Imports `Model` class: `from plexe.models import Model`
2. Creates Model instance:
   ```python
   model = Model(
       intent=intent,
       input_schema=final_input_schema,
       output_schema=final_output_schema
   )
   ```
3. **File**: `plexe/models.py` (lines 93-138)
   - `__init__` executes:
     - Stores intent, schemas
     - Initializes `state = ModelState.DRAFT`
     - Creates unique identifier
     - Creates working directory
     - Initializes empty predictor, artifacts, metrics
4. Populates model with results:
   - `model.identifier = model_identifier`
   - `model.predictor = generated.predictor` (Predictor instance)
   - `model.trainer_source = generated.training_source_code`
   - `model.predictor_source = generated.inference_source_code`
   - `model.feature_transformer_source = generated.feature_transformer_source_code`
   - `model.dataset_splitter_source = generated.dataset_split_code`
   - `model.testing_source = generated.testing_source_code`
   - `model.artifacts = generated.model_artifacts`
   - `model.metric = generated.test_performance`
   - `model.evaluation_report = generated.evaluation_report`
   - `model.metadata.update(metadata)`
   - `model.training_data = training_data` (actual Dataset objects)
   - `model.state = ModelState.READY`

---

### Step 6.4: Final Callback Notification
**File**: `plexe/model_builder.py` (lines 238-252)

**What Happens**:
- Calls `_notify_callbacks()` with event="build_end"
- All callbacks receive final BuildStateInfo with:
  - Completed model
  - Final metrics
  - Source code
  - Artifacts

---

### Step 6.5: Return Model
**File**: `plexe/model_builder.py` (line 254)

**What Happens**:
- Returns completed `Model` instance to user
- Model is in `ModelState.READY` state
- All code, artifacts, and metadata populated

---

## Phase 7: User Makes Prediction

### Step 7.1: User Calls model.predict()
**File**: `plexe/models.py` (lines 227-245)

**What Happens**:
1. User calls: `model.predict({"bedrooms": 3, "bathrooms": 2, "square_footage": 1500.0})`
2. `predict()` method executes:
   - Checks `self.state == ModelState.READY` (line 235)
   - If `validate_input=True`, validates input against `self.input_schema` (line 239)
   - Calls `self.predictor.predict(x)` (line 240)
     - **File**: `plexe/core/interfaces/predictor.py`
     - This is the `PredictorImplementation` class compiled earlier
     - Executes inference code:
       - Loads model artifacts
       - Applies feature transformations (if any)
       - Runs model forward pass
       - Returns prediction dict
   - If `validate_output=True`, validates output against `self.output_schema` (line 242)
   - Returns prediction dict: `{"price": 250000.0}`

**Output**: Prediction dictionary matching output schema

---

## Complete File Trace Summary

### User-Facing Files:
1. `plexe/model_builder.py` - Main entry point
2. `plexe/models.py` - Model class definition

### Core System Files:
3. `plexe/agents/agents.py` - Multi-agent orchestrator
4. `plexe/core/object_registry.py` - Shared state registry
5. `plexe/core/state.py` - Model state definitions
6. `plexe/core/interfaces/predictor.py` - Predictor interface

### Agent Files:
7. `plexe/agents/model_planner.py` - Solution planning
8. `plexe/agents/dataset_analyser.py` - EDA analysis
9. `plexe/agents/schema_resolver.py` - Schema inference
10. `plexe/agents/feature_engineer.py` - Feature engineering
11. `plexe/agents/dataset_splitter.py` - Dataset splitting
12. `plexe/agents/model_trainer.py` - Model training
13. `plexe/agents/model_packager.py` - Inference code generation
14. `plexe/agents/model_tester.py` - Model testing

### Tool Files:
15. `plexe/tools/datasets.py` - Dataset operations
16. `plexe/tools/metrics.py` - Metric selection
17. `plexe/tools/training.py` - Training operations
18. `plexe/tools/execution.py` - Code execution
19. `plexe/tools/validation.py` - Code validation
20. `plexe/tools/evaluation.py` - Model evaluation
21. `plexe/tools/schemas.py` - Schema operations
22. `plexe/tools/response_formatting.py` - Response formatting

### Utility Files:
23. `plexe/internal/common/utils/pydantic_utils.py` - Schema conversion
24. `plexe/internal/common/provider.py` - LLM provider interface
25. `plexe/internal/common/datasets/adapter.py` - Dataset adapters
26. `plexe/core/entities/solution.py` - Solution data structures
27. `plexe/internal/models/entities/artifact.py` - Artifact definitions
28. `plexe/internal/models/entities/metric.py` - Metric definitions

---

## Data Flow Diagram

```
User Input
    ↓
ModelBuilder.build()
    ↓
ObjectRegistry (cleared, then populated)
    ↓
PlexeAgent (orchestrator + 8 specialists)
    ↓
Orchestrator Agent (coordinates workflow)
    ├─→ EDA Agent → EDA Report → Registry
    ├─→ Schema Resolver → Schemas → Registry (if needed)
    ├─→ ML Research Agent → Solution Plans → Registry
    ├─→ Feature Engineer → Transformed Data → Registry (optional)
    ├─→ Dataset Splitter → Train/Val/Test → Registry
    ├─→ Model Trainer → Trained Models → Registry
    ├─→ Model Packager → Inference Code → Registry
    └─→ Model Tester → Evaluation Reports → Registry
    ↓
Best Solution Selected → Registry
    ↓
ModelGenerationResult (extracted from registry)
    ↓
Model Instance (populated with results)
    ↓
User receives Model (ready for predictions)
    ↓
model.predict() → Predictor.predict() → Prediction
```

---

This walkthrough covers every major file and code path in the system, showing how data flows from user input through the multi-agent system to final predictions.

