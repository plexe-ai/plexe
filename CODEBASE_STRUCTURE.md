# Plexe Codebase Structure - Complete A-Z Documentation

## Overview
Plexe is an agentic framework for building ML models from natural language. It uses a multi-agent system where specialized AI agents collaborate to analyze data, plan solutions, train models, and generate production-ready inference code.

## Architecture Summary
- **Entry Points**: `main.py` (CLI), `server.py` (Web UI), `model_builder.py` (Core API)
- **Core System**: Object registry, state management, storage, interfaces
- **Agents**: 8 specialized agents for different ML tasks
- **Tools**: 14 tool modules providing agent capabilities
- **Internal**: Common utilities, models, datasets, schemas
- **Testing**: Unit, integration, and benchmark tests

---

## A-Z File Structure and Dependencies

### **A - Agents** (`plexe/agents/`)
**Purpose**: Multi-agent system for ML model building

#### `agents.py` - Main Orchestrator
- **What it does**: Coordinates 8 specialized agents in ML workflow
- **Key classes**: `PlexeAgent`, `ModelGenerationResult`
- **Dependencies**: All other agent modules, tools, core modules
- **Flow**: Creates agents → Runs orchestrator → Extracts results

#### `conversational.py` - User Interface Agent
- **What it does**: Guides users through ML requirements via chat
- **Key classes**: `ConversationalAgent`
- **Dependencies**: `tools.conversation`, `tools.datasets`
- **Flow**: User input → Validation → Model build initiation

#### `dataset_analyser.py` - EDA Agent
- **What it does**: Performs exploratory data analysis on datasets
- **Key classes**: `EdaAgent`
- **Dependencies**: `tools.datasets`, `tools.schemas`
- **Flow**: Dataset → Analysis → EDA report registration

#### `dataset_splitter.py` - Data Splitting Agent
- **What it does**: Intelligently splits datasets into train/val/test
- **Key classes**: `DatasetSplitterAgent`
- **Dependencies**: `tools.datasets`
- **Flow**: Dataset → Split strategy → Registered splits

#### `feature_engineer.py` - Feature Engineering Agent
- **What it does**: Transforms raw data into optimized features
- **Key classes**: `FeatureEngineeringAgent`
- **Dependencies**: `tools.datasets`, `tools.execution`, `tools.validation`
- **Flow**: Raw data → Transformations → Enhanced datasets

#### `model_packager.py` - MLOps Agent
- **What it does**: Creates production-ready inference code
- **Key classes**: `ModelPackagerAgent`
- **Dependencies**: `tools.context`, `tools.validation`, `tools.solutions`
- **Flow**: Training code → Inference code → Production package

#### `model_planner.py` - ML Research Agent
- **What it does**: Plans ML approaches and solution strategies
- **Key classes**: `ModelPlannerAgent`
- **Dependencies**: `tools.datasets`, `tools.schemas`, `tools.solutions`
- **Flow**: Requirements → Analysis → Solution plans

#### `model_trainer.py` - ML Engineer Agent
- **What it does**: Implements and executes training code
- **Key classes**: `ModelTrainerAgent`
- **Dependencies**: `tools.execution`, `tools.training`, `tools.validation`
- **Flow**: Plan → Code generation → Training → Validation

#### `model_tester.py` - Model Testing Agent
- **What it does**: Tests and evaluates finalized models
- **Key classes**: `ModelTesterAgent`
- **Dependencies**: `tools.testing`, `tools.datasets`, `tools.schemas`
- **Flow**: Model → Testing → Evaluation report

#### `schema_resolver.py` - Schema Resolution Agent
- **What it does**: Resolves and validates input/output schemas
- **Key classes**: `SchemaResolverAgent`
- **Dependencies**: `tools.schemas`
- **Flow**: Data analysis → Schema inference → Validation

---

### **B - Core System** (`plexe/core/`)
**Purpose**: Fundamental types and functionality used across Plexe

#### `state.py` - Model State Management
- **What it does**: Defines model lifecycle states
- **Key classes**: `ModelState` (Enum)
- **States**: DRAFT, BUILDING, READY, ERROR
- **Dependencies**: None (base enum)

#### `storage.py` - Model Persistence
- **What it does**: Core implementation for saving/loading models and checkpoints
- **Key functions**: `_save_model_to_tar()`, `_load_model_data_from_tar()`, `_save_checkpoint_to_tar()`
- **Dependencies**: `core.state`, `config`, `internal.common.utils.pydantic_utils`
- **Flow**: Model data → Tar archive → File system

#### `object_registry.py` - Global Object Registry
- **What it does**: Singleton registry for storing/retrieving objects by type and name
- **Key classes**: `ObjectRegistry`, `Item`
- **Dependencies**: None (core functionality)
- **Flow**: Register objects → Retrieve by type/name → Manage lifecycle

#### `entities/solution.py` - Solution Container
- **What it does**: Represents complete ML solutions from planning to deployment
- **Key classes**: `Solution`
- **Dependencies**: `internal.models.entities.artifact`, `internal.models.entities.metric`
- **Flow**: Plan → Code → Execution → Results → Artifacts

#### `interfaces/` - Core Interfaces
- **`predictor.py`**: Abstract base for inference code (`Predictor`)
- **`feature_transformer.py`**: Abstract base for feature transformers (`FeatureTransformer`)
- **Dependencies**: `internal.models.entities.artifact`, `pandas`
- **Purpose**: Define contracts for generated code

---

### **C - Configuration** (`plexe/config.py`)
**Purpose**: Central configuration management

- **What it does**: Loads configuration from YAML files and environment variables
- **Key components**: `config` object, `prompt_templates` object
- **Dependencies**: `yaml`, `pathlib`
- **Flow**: YAML files → Configuration object → Application settings

---

### **D - Datasets** (`plexe/datasets.py`)
**Purpose**: Dataset generation and management

- **What it does**: Provides `DatasetGenerator` class for synthetic data creation
- **Key classes**: `DatasetGenerator`
- **Dependencies**: `pandas`, `numpy`, `internal.common.datasets.interface`
- **Flow**: Schema → Synthetic data → TabularConvertible dataset

---

### **E - Entry Points**
**Purpose**: Application entry points and main interfaces

#### `main.py` - CLI Entry Point
- **What it does**: Launches Plexe assistant with web UI
- **Dependencies**: `server`, `uvicorn`, `webbrowser`
- **Flow**: Start server → Open browser → Keep running

#### `server.py` - Web Server
- **What it does**: FastAPI server for conversational agent
- **Dependencies**: `agents.conversational`, `fastapi`, `websockets`
- **Flow**: WebSocket connection → Agent processing → Response

#### `model_builder.py` - Core API
- **What it does**: Main API for building ML models programmatically
- **Key classes**: `ModelBuilder`
- **Dependencies**: All agents, core modules, tools, callbacks
- **Flow**: Intent + Data → Agent orchestration → Complete model

---

### **F - File I/O** (`plexe/fileio.py`)
**Purpose**: High-level file operations for models and checkpoints

- **What it does**: Provides user-friendly functions for model persistence
- **Key functions**: `save_model()`, `load_model()`, `save_checkpoint()`, `load_checkpoint()`
- **Dependencies**: `core.storage`, `models`
- **Flow**: Model object → Storage functions → File system

---

### **G - Internal Common** (`plexe/internal/common/`)
**Purpose**: Shared utilities and common functionality

#### `provider.py` - LLM Provider Management
- **What it does**: Manages different LLM providers with retry logic
- **Key classes**: `ProviderConfig`, `Provider`
- **Dependencies**: `litellm`, `tenacity`, `pydantic`
- **Flow**: Provider config → API calls → Response with retries

#### `datasets/` - Dataset Interfaces and Adapters
- **`interface.py`**: Core dataset interfaces (`Dataset`, `TabularConvertible`, etc.)
- **`adapter.py`**: Adapter for converting between dataset types
- **`tabular.py`**: Tabular dataset implementation
- **Dependencies**: `pandas`, `numpy`, `abc`
- **Flow**: Data → Interface implementation → Convertible formats

#### `utils/` - Utility Functions
- **`agents.py`**: Agent utility functions
- **`chain_of_thought/`**: Chain of thought logging utilities
- **`dataset_storage.py`**: Dataset storage utilities
- **`dependency_utils.py`**: Dependency management
- **`markdown_utils.py`**: Markdown formatting
- **`model_state.py`**: Model state utilities
- **`model_utils.py`**: Model utility functions
- **`pandas_utils.py`**: Pandas utilities
- **`prompt_utils.py`**: Prompt template utilities
- **`pydantic_utils.py`**: Pydantic utilities
- **`response.py`**: Response formatting utilities

---

### **H - Internal Models** (`plexe/internal/models/`)
**Purpose**: Internal model entities and execution logic

#### `entities/` - Model Entities
- **`artifact.py`**: Represents model artifacts (files, data, handles)
- **`code.py`**: Represents code objects with performance metrics
- **`description.py`**: Model description entities
- **`metric.py`**: Metric comparison and evaluation logic
- **Dependencies**: `pathlib`, `io`, `dataclasses`

#### `execution/` - Code Execution
- **`executor.py`**: Base executor interface
- **`process_executor.py`**: Process-based execution
- **`docker_executor.py`**: Docker-based execution
- **`ray_executor.py`**: Ray-based distributed execution
- **Dependencies**: `subprocess`, `docker`, `ray`

#### `generation/` - Code Generation
- **`planning.py`**: Solution planning logic
- **`review.py`**: Code review functionality
- **`training.py`**: Training code generation
- **Dependencies**: `internal.common.provider`, `internal.common.utils`

#### `validation/` - Code Validation
- **`validator.py`**: Main validation orchestrator
- **`composite.py`**: Composite validators
- **`primitives/`**: Basic validation primitives
- **Dependencies**: `ast`, `importlib`, `subprocess`

#### `callbacks/` - Execution Callbacks
- **`chain_of_thought.py`**: Chain of thought logging
- **`checkpoint.py`**: Model checkpointing
- **`mlflow.py`**: MLflow integration
- **Dependencies**: `mlflow`, `plexe.core.object_registry`

---

### **I - Internal Datasets** (`plexe/internal/datasets/`)
**Purpose**: Dataset generation and validation

#### `core/` - Core Dataset Logic
- **`generation/`**: Dataset generation utilities
- **`validation/`**: Dataset validation logic
- **Dependencies**: `pandas`, `numpy`, `sklearn`

#### `generator.py` - Dataset Generator
- **What it does**: Main dataset generation interface
- **Dependencies**: `core.generation`, `core.validation`

#### `config.py` - Dataset Configuration
- **What it does**: Dataset generation configuration
- **Dependencies**: `pydantic`

---

### **J - Internal Schemas** (`plexe/internal/schemas/`)
**Purpose**: Schema resolution and management

#### `resolver.py` - Schema Resolver
- **What it does**: Resolves and validates schemas from data
- **Dependencies**: `pandas`, `pydantic`, `internal.common.utils`

---

### **K - Models** (`plexe/models.py`)
**Purpose**: Main Model class and related functionality

- **What it does**: Defines the main `Model` class representing a complete ML model
- **Key classes**: `Model`
- **Dependencies**: `core.state`, `core.interfaces`, `internal.models.entities`
- **Flow**: Model creation → Training → Inference → Persistence

---

### **L - Tools** (`plexe/tools/`)
**Purpose**: Tool functions used by agents

#### `code_analysis.py` - Code Analysis Tools
- **What it does**: Analyzes and extracts code from solutions
- **Dependencies**: `core.object_registry`, `internal.models.entities.code`

#### `context.py` - Context Tools
- **What it does**: Provides context for code generation
- **Dependencies**: `core.object_registry`, `internal.common.provider`

#### `conversation.py` - Conversation Tools
- **What it does**: Tools for conversational agent
- **Dependencies**: `core.object_registry`, `model_builder`

#### `datasets.py` - Dataset Tools
- **What it does**: Dataset manipulation and registration tools
- **Dependencies**: `core.object_registry`, `internal.common.datasets`

#### `evaluation.py` - Evaluation Tools
- **What it does**: Model evaluation and review tools
- **Dependencies**: `core.object_registry`, `internal.common.provider`

#### `execution.py` - Execution Tools
- **What it does**: Code execution and training tools
- **Dependencies**: `core.object_registry`, `internal.models.execution`

#### `metrics.py` - Metrics Tools
- **What it does**: Metric selection and comparison tools
- **Dependencies**: `core.object_registry`, `internal.common.provider`

#### `response_formatting.py` - Response Formatting
- **What it does**: Formats agent responses
- **Dependencies**: `core.object_registry`, `internal.common.provider`

#### `schemas.py` - Schema Tools
- **What it does**: Schema inference and validation tools
- **Dependencies**: `core.object_registry`, `internal.common.utils`

#### `solutions.py` - Solution Management
- **What it does**: Solution creation and management tools
- **Dependencies**: `core.object_registry`, `core.entities.solution`

#### `testing.py` - Testing Tools
- **What it does**: Model testing and evaluation tools
- **Dependencies**: `core.object_registry`, `internal.common.provider`

#### `training.py` - Training Tools
- **What it does**: Training code generation and management
- **Dependencies**: `core.object_registry`, `internal.common.provider`

#### `validation.py` - Validation Tools
- **What it does**: Code and model validation tools
- **Dependencies**: `core.object_registry`, `internal.models.validation`

---

### **M - Templates** (`plexe/templates/`)
**Purpose**: Jinja2 templates for prompts and code generation

#### `models/` - Model Templates
- **`feature_transformer.tmpl.py`**: Feature transformer code template
- **`predictor.tmpl.py`**: Predictor code template

#### `prompts/` - Prompt Templates
- **`agent/`**: Agent-specific prompt templates
- **`planning/`**: Planning prompt templates
- **`review/`**: Review prompt templates
- **`schemas/`**: Schema prompt templates
- **`training/`**: Training prompt templates
- **`utils/`**: Utility prompt templates

---

### **N - UI** (`plexe/ui/`)
**Purpose**: Web user interface

#### `index.html` - Web Interface
- **What it does**: HTML interface for conversational agent
- **Dependencies**: WebSocket connection to server
- **Flow**: User input → WebSocket → Agent → Response

---

### **O - Callbacks** (`plexe/callbacks.py`)
**Purpose**: Callback system for model building lifecycle

- **What it does**: Defines callback interface and implementations
- **Key classes**: `Callback`, `MLFlowCallback`, `ModelCheckpointCallback`
- **Dependencies**: `mlflow`, `core.object_registry`
- **Flow**: Model events → Callback execution → External systems

---

### **P - Tests** (`tests/`)
**Purpose**: Comprehensive test suite

#### `unit/` - Unit Tests
- **What it does**: Tests individual components in isolation
- **Coverage**: Core modules, internal utilities, models
- **Dependencies**: `pytest`, individual modules

#### `integration/` - Integration Tests
- **What it does**: End-to-end tests for complete ML workflows
- **Coverage**: Binary classification, regression, time series, etc.
- **Dependencies**: `pytest`, full plexe system

#### `benchmark/` - Benchmark Tests
- **What it does**: Performance benchmarks using MLE-Bench
- **Coverage**: Kaggle competitions, model performance
- **Dependencies**: `mle-bench`, `kaggle`

#### `fixtures/` - Test Fixtures
- **What it does**: Test data and model artifacts
- **Contents**: Legacy model files for compatibility testing

---

## Dependency Flow Diagram

```
Entry Points (main.py, server.py, model_builder.py)
    ↓
Core System (state.py, storage.py, object_registry.py)
    ↓
Agents (8 specialized agents)
    ↓
Tools (14 tool modules)
    ↓
Internal Modules (common, models, datasets, schemas)
    ↓
External Dependencies (pandas, sklearn, litellm, etc.)
```

## Key Dependencies

### External Dependencies
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **litellm**: LLM provider abstraction
- **smolagents**: Agent framework
- **pydantic**: Data validation
- **fastapi**: Web server
- **mlflow**: Experiment tracking

### Internal Dependencies
- **ObjectRegistry**: Central object storage
- **Provider**: LLM provider management
- **Solution**: ML solution container
- **Artifact**: Model artifact representation
- **Metric**: Performance metric handling

## Data Flow

1. **Input**: User intent + datasets
2. **Analysis**: EDA agent analyzes data
3. **Planning**: Research agent creates solution plans
4. **Feature Engineering**: Feature agent transforms data
5. **Splitting**: Splitter agent creates train/val/test sets
6. **Training**: Engineer agent implements and trains models
7. **Packaging**: Ops agent creates inference code
8. **Testing**: Tester agent evaluates final model
9. **Output**: Complete model with artifacts and code

## Architecture Patterns

- **Multi-Agent System**: Specialized agents for different tasks
- **Object Registry**: Centralized object storage and retrieval
- **Tool-Based Architecture**: Agents use tools for capabilities
- **Template System**: Jinja2 templates for code generation
- **Callback System**: Lifecycle event handling
- **Interface Segregation**: Clear contracts between components
- **Dependency Injection**: Loose coupling between modules
