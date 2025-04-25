# CLAUDE.md: Plexe Coding Reference

## Project Structure
- `plexe/`: Main package directory
- `plexe/models.py`: Implemented the top-level `Model` class
- `plexe/fileio.py`: Saving and loading models
- `plexe/config.py`: Configuration for model building, including LLM prompts
- `plexe/internal/common/`: Package containing common utilities and functions
- `plexe/internal/models/`: Package containing model building and training logic
- `plexe/internal/datasets`: Package containing synthetic data generation logic
- `plexe/internal/schemas/`: Package containing schema validation and inference logic
- `plexe/internal/models/generators.py`: Main implementation of the model building and training logic

## Build/Run Commands
- Install deps: `poetry install`
- Format code: `poetry run black .`
- Lint code: `poetry run ruff check . --fix`
- Run all tests: `poetry run pytest tests/`
- Run single test: `poetry run pytest tests/path/to/test_file.py::test_function_name`
- Run unit tests: `poetry run pytest tests/unit/`
- Run integration tests: `poetry run pytest tests/integration/`
- Run with coverage: `poetry run pytest --cov=plexe tests/`

## Code Style
- **Paradigm**: object-oriented structure, functional implementations where appropriate
- **Functions**: 50 lines max (not including docstrings)
- **Formatting**: Black with 120 char line length
- **Linting**: Ruff with E203/E501/E402 ignored
- **Typing**: Use type hints and Pydantic models
- **Naming**: snake_case (functions/vars), CamelCase (classes)
- **Imports**: Group stdlib, third-party, then local imports; NO LOCAL IMPORTS, always import at the top of the file
- **__init__.py**: No code in __init__.py files except in plexe/__init__.py for convenience
- **Docstrings**: Required for public modules/classes/functions; Sphinx style without type hints
- **Testing**: Write pytest tests for all new functionality

## Commit Messages
- Format: `<type>: <subject>`
- Types: feat, fix, docs, style, refactor, test, chore
- Example: `feat: add support for deepseek`
