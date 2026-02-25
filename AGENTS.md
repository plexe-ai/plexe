# AGENTS.md

This file provides guidance to coding agents when working with code in this repository (for example, OpenAI Codex and Claude Code).

`AGENTS.md` is the canonical version of this document. `CLAUDE.md` is kept as a compatibility symlink that points here, so if you opened this file via `CLAUDE.md`, you are in the right place.

## Overview

Plexe is an agentic framework for building ML models from natural language. It employs a multi-agent architecture where 
specialized AI agents collaborate to analyze data, generate solutions, and build functional ML models through an 
autonomous 6-phase workflow.

**Entry point:** `python -m plexe.main --train-dataset-uri <uri> --user-id <id> --intent "<task>" --spark-mode <local|databricks>`

**Docker:**
- `docker build .` — default PySpark image (local Spark execution)
- `docker build --target databricks .` — Databricks Connect image (remote execution)

## Architecture

### 6-Phase Workflow (`plexe/workflow.py`)

1. **Data Understanding**: Statistical analysis → ML task identification → metric selection (with optional custom metric generation)
2. **Data Preparation**: Dataset splitting → intelligent sampling (default: 30k train, 10k val samples)
3. **Baseline Models**: Heuristic baseline with retry logic
4. **Model Search**: Hypothesis-driven tree search on samples (iterative improvement)
5. **Final Evaluation**: Optional test evaluation (default: disabled, uses validation performance)
6. **Packaging**: Consolidates artifacts into `work_dir/model/` (schemas/, config/, artifacts/, src/, evaluation/)

### Multi-Agent System (`plexe/agents/`)

14 specialized agents orchestrate the workflow:
- **LayoutDetectionAgent** → **StatisticalAnalyserAgent** → **MLTaskAnalyserAgent** → **MetricSelectorAgent** (Phase 1)
- **MetricImplementationAgent**: Generates custom metric code if not in `StandardMetric` enum
- **DatasetSplitterAgent** → **SamplingAgent** (Phase 2)
- **BaselineBuilderAgent** (Phase 3)
- **HypothesiserAgent** → **PlannerAgent** → **FeatureProcessorAgent** + **ModelDefinerAgent** (Phase 4 loop)
- **InsightExtractorAgent**: Analyzes variant results, populates `InsightStore` for future hypotheses
- **ModelEvaluatorAgent**: Multi-phase evaluation (Phase 5)

### Tree Search (`plexe/search/`)

Three-stage tree expansion strategy:
1. **Bootstrap**: Create diverse initial solutions from scratch (no parent)
2. **Debug**: Probabilistically fix buggy leaf nodes (max depth: 2)
3. **Improve**: Greedily expand best-performing solutions

Each iteration is hypothesis-driven: `HypothesiserAgent` analyzes the journal + accumulated insights to decide 
what to try next, `PlannerAgent` turns that into concrete plans (FeaturePlan + ModelPlan), and `InsightExtractorAgent` 
distills learnings from results back into the `InsightStore` to inform future iterations.

Search runs on **samples** (fast), best solution retrained on **full dataset** (accurate).

### Workflow Integration (`plexe/integrations/`)

Pluggable interface for connecting plexe to external infrastructure:
- **`WorkflowIntegration`** (`base.py`): ABC defining the contract (8 methods)
- **`StandaloneIntegration`** (`standalone.py`): Default implementation (local + optional S3)
- **`storage/`**: Composable storage helpers (`S3Helper`, Azure/GCS stubs)

Custom integrations implement `WorkflowIntegration` and pass the instance to `main(integration=MyIntegration(...))`.

### State Management

- **BuildContext** (`models.py`): Central state object passed through workflow
- **SearchJournal** (`search/journal.py`): DAG of all solutions, tracks ancestry and performance
- **InsightStore** (`search/insight_store.py`): Accumulates learnings across search iterations

## Commands

```bash
# Install
poetry install

# Run locally
python -m plexe.main --train-dataset-uri data.parquet --user-id user123 --intent "predict churn" --spark-mode local

# Build Docker images
make build                    # PySpark (default)
make build-databricks         # Databricks Connect

# Run tests
poetry run pytest tests/unit/

# Format and lint
poetry run black .
poetry run ruff check . --fix

# Quick integration test via Docker
make test-quick
```

## Key Files

- `plexe/workflow.py`: Main orchestrator (6 phases)
- `plexe/main.py`: CLI entry point
- `plexe/config.py`: Config dataclass + StandardMetric enum + LLM routing
- `plexe/models.py`: Data models (BuildContext, Solution, Baseline, Hypothesis, Plan)
- `plexe/helpers.py`: Metric computation and model type selection
- `plexe/integrations/base.py`: WorkflowIntegration ABC
- `plexe/integrations/standalone.py`: Default integration (local + S3)
- `plexe/integrations/storage/`: Storage helper ABCs and implementations
- `plexe/search/journal.py`: Solution DAG tracking
- `plexe/search/tree_policy.py`: Search strategy
- `plexe/agents/*.py`: 14 specialized agents
- `plexe/templates/`: Code generation templates (training, inference, features, packaging)
- `plexe/utils/litellm_wrapper.py`: LLM wrapper with retries and optional `on_llm_call` hook
- `plexe/utils/tooling.py`: `@agentinspectable` decorator for agent-callable functions

## Code Style

- **Functions**: Max 50 lines (excluding docstrings)
- **Formatting**: Black with 120 char line length
- **Linting**: Ruff with E203/E501/E402 ignored
- **Typing**: Type hints and Pydantic models required
- **Imports**: ALWAYS at top level in order: stdlib, third-party, local; NEVER inside functions
- **__init__.py**: No implementation code except in `__init__.py` files
- **Docstrings**: Required for public APIs; Sphinx style
- **Testing**: Write pytest tests in `tests/unit/` mirroring `plexe/` package structure
- **Elegance**: Write the simplest solution possible; avoid over-engineering; prefer deleting code over adding code
