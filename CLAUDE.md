# CLAUDE.md: Plexe Coding Reference

## Project Overview
Plexe is a framework for building ML models using natural language. It employs a multi-agent architecture where 
specialized AI agents collaborate to analyze requirements, generate solutions, and build functional ML models.

The core architecture is as follows: agents go in `plexe/agents/*`, tools in `plexe/tools/*`, prompt templates in
`plexe/templates/prompts/*`, and the main model code in `plexe/models.py`. This structure must be followed.

## Key Components
- `plexe/models.py`: Core `Model` class with build/predict functionality
- `plexe/agents/schema_resolver.py`: Agent inferring input/output schemas
- `plexe/internal/agents.py`: Multi-agent system implementation (`PlexeAgent` class)
- `plexe/tools/`: Tools for code generation, execution, validation
- `plexe/config.py`: Configuration management and prompt templates
- `plexe/internal/common/registries/objects.py`: Shared object registry for agents
- `plexe/datasets.py`: Dataset handling and synthetic data generation
- `docs/architecture/multi-agent-system.md`: Architectural documentation
- `plexe/templates/prompts/`: Prompt templates for agents and LLM calls

## Build/Run Commands
- Install deps: `poetry install`
- Format code: `poetry run black .`
- Lint code: `poetry run ruff check . --fix`
- Run tests: `poetry run pytest tests/`
- Run with coverage: `poetry run pytest --cov=plexe tests/`

## Code Style
- **Functions**: Max 50 lines (excluding docstrings)
- **Formatting**: Black with 120 char line length
- **Linting**: Ruff with E203/E501/E402 ignored
- **Typing**: Type hints and Pydantic models required
- **Imports**: ALWAYS at top level in order: stdlib, third-party, local; NEVER inside functions
- **__init__.py**: No implementation code except in `plexe/__init__.py`
- **Docstrings**: Required for public APIs; Sphinx style
- **Testing**: Write pytest tests for all new functionality
- **Elegance**: Write the simplest solution possible; avoid over-engineering; prefer deleting code over adding code
