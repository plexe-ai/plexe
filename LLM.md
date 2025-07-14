# Plexe Library API Documentation

**Plexe** is a Python library that lets you build machine learning models using natural language descriptions. It uses a multi-agent AI system to automatically generate, train, and package ML models.

## Installation

```bash
pip install plexe                  # Standard installation
pip install plexe[transformers]    # With transformers support
pip install plexe[chatui]          # With chat UI
pip install plexe[all]             # All optional dependencies
```

## Quick Start

```python
import plexe

# Create model with natural language intent
model = plexe.Model(
    intent="Predict sentiment from news articles",
    input_schema={"headline": str, "content": str},
    output_schema={"sentiment": str}
)

# Build model using ModelBuilder (recommended)
builder = plexe.ModelBuilder(provider="openai/gpt-4o-mini")
model = builder.build(
    intent="Predict house prices",
    datasets=[your_dataframe],
    timeout=1800
)

# Make predictions
prediction = model.predict({
    "headline": "New breakthrough in renewable energy",
    "content": "Scientists announced..."
})

# Save and load models
plexe.save_model(model, "my-model")
loaded_model = plexe.load_model("my-model.tar.gz")
```

## Core API

### Model Class

**Import:** `from plexe import Model`

```python
Model(
    intent: str,
    input_schema: Type[BaseModel] | Dict[str, type] = None,
    output_schema: Type[BaseModel] | Dict[str, type] = None,
    distributed: bool = False
)
```

Represents a machine learning model with natural language intent and structured schemas.

**Key Methods:**
- `predict(x: Dict[str, Any]) -> Dict[str, Any]` - Make predictions
- `get_state() -> ModelState` - Get current model state
- `get_metadata() -> dict` - Get model metadata
- `get_metrics() -> dict` - Get performance metrics
- `describe() -> ModelDescription` - Get structured model description

**Key Attributes:**
- `intent` - Natural language description
- `input_schema` - Input structure definition
- `output_schema` - Output structure definition
- `state` - Current model state
- `predictor` - Underlying predictor instance

### ModelBuilder Class (Recommended)

**Import:** `from plexe import ModelBuilder`

```python
ModelBuilder(
    provider: str | ProviderConfig = "openai/gpt-4o-mini",
    verbose: bool = False,
    distributed: bool = False,
    working_dir: Optional[str] = None
)
```

Factory for creating ML models through agentic workflows.

**Key Method:**
```python
build(
    intent: str,
    datasets: List[pd.DataFrame | DatasetGenerator],
    input_schema: Type[BaseModel] | Dict[str, type] = None,
    output_schema: Type[BaseModel] | Dict[str, type] = None,
    timeout: int = None,
    max_iterations: int = None,
    run_timeout: int = 1800,
    callbacks: List[Callback] = None,
    enable_checkpointing: bool = False
) -> Model
```

### DatasetGenerator Class

**Import:** `from plexe import DatasetGenerator`

```python
DatasetGenerator(
    description: str,
    provider: str,
    schema: Type[BaseModel] | Dict[str, type] = None,
    data: pd.DataFrame = None
)
```

Manages datasets with synthetic data generation capabilities.

**Key Methods:**
- `generate(num_samples: int)` - Generate synthetic data
- `data` (property) - Access dataset as DataFrame

## File I/O

**Import:** `from plexe import save_model, load_model`

### Model Persistence
- `save_model(model: Any, path: str | Path) -> str` - Save model to archive
- `load_model(path: str | Path) -> Model` - Load model from archive

### Checkpoint Management
- `save_checkpoint(model: Any, iteration: int, path: Optional[str | Path] = None) -> str`
- `load_checkpoint(checkpoint_path: Optional[str | Path] = None, model_id: Optional[str] = None, latest: bool = False) -> Model`
- `list_checkpoints(model_id: Optional[str] = None) -> List[str]`
- `clear_checkpoints(model_id: Optional[str] = None, older_than_days: Optional[int] = None) -> int`

## Callbacks

**Import:** `from plexe import Callback, MLFlowCallback, ModelCheckpointCallback`

### Base Callback
```python
class Callback:
    def on_build_start(info: BuildStateInfo) -> None
    def on_build_end(info: BuildStateInfo) -> None
    def on_iteration_start(info: BuildStateInfo) -> None
    def on_iteration_end(info: BuildStateInfo) -> None
```

### MLFlowCallback
```python
MLFlowCallback(
    tracking_uri: str,
    experiment_name: str,
    connect_timeout: int = 10
)
```
Tracks model building to MLFlow with hierarchical runs.

### ModelCheckpointCallback
```python
ModelCheckpointCallback(
    keep_n_latest: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    delete_on_success: Optional[bool] = None
)
```
Saves model checkpoints during building.

## Configuration

**Import:** `from plexe.config import config`

### Key Configuration Areas
- `config.file_storage` - File paths and storage settings
- `config.model_search` - Model search parameters
- `config.code_generation` - Code generation settings
- `config.ray` - Distributed computing configuration
- `config.logging` - Logging configuration

### Configuration Functions
- `configure_logging(level, file=None)` - Configure logging
- `is_package_available(package_name)` - Check package availability

## Provider Support

Plexe supports multiple LLM providers through LiteLLM:

```python
# OpenAI
model.build(provider="openai/gpt-4o-mini")

# Anthropic
model.build(provider="anthropic/claude-3-opus")

# Ollama
model.build(provider="ollama/llama2")

# Hugging Face
model.build(provider="huggingface/meta-llama/...")
```

## Distributed Training

Enable distributed training with Ray:

```python
from plexe import ModelBuilder
from plexe.config import config

# Optional: Configure Ray cluster
config.ray.address = "ray://10.1.2.3:10001"

# Enable distributed training
builder = ModelBuilder(distributed=True)
model = builder.build(
    intent="Predict house prices",
    datasets=[df],
    distributed=True
)
```

## Advanced Usage

### Custom Schemas with Pydantic
```python
from pydantic import BaseModel, create_model

class HouseInput(BaseModel):
    bedrooms: int
    bathrooms: int
    square_footage: float

class HouseOutput(BaseModel):
    price: float

model = Model(
    intent="Predict house prices",
    input_schema=HouseInput,
    output_schema=HouseOutput
)
```

### Using Callbacks
```python
from plexe.callbacks import MLFlowCallback, ModelCheckpointCallback

callbacks = [
    MLFlowCallback(
        tracking_uri="http://localhost:5000",
        experiment_name="house_prices"
    ),
    ModelCheckpointCallback(keep_n_latest=3)
]

builder = ModelBuilder()
model = builder.build(
    intent="Predict house prices",
    datasets=[df],
    callbacks=callbacks,
    enable_checkpointing=True
)
```

### Data Generation
```python
# Generate synthetic data
dataset = DatasetGenerator(
    description="House price dataset with features",
    provider="openai/gpt-4o-mini",
    schema={"bedrooms": int, "price": float}
)
dataset.generate(1000)

# Use generated data
model = builder.build(
    intent="Predict house prices",
    datasets=[dataset]
)
```

## Error Handling

Plexe provides comprehensive error handling:

```python
try:
    model = builder.build(
        intent="Predict house prices",
        datasets=[df],
        timeout=1800
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except TimeoutError as e:
    print(f"Build timeout: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Environment Variables

Set API keys for your preferred provider:

```bash
export OPENAI_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
export GEMINI_API_KEY=<your-key>
```

## Model States

Models progress through these states:
- `DRAFT` - Initial state
- `BUILDING` - Currently being built
- `READY` - Built and ready for predictions
- `ERROR` - Build failed

## Best Practices

1. **Use ModelBuilder**: Preferred over deprecated `Model.build()`
2. **Set Timeouts**: Always specify `timeout` or `max_iterations`
3. **Enable Checkpointing**: For long-running builds
4. **Use Callbacks**: For monitoring and logging
5. **Validate Schemas**: Define clear input/output schemas
6. **Handle Errors**: Implement proper error handling
7. **Save Models**: Persist trained models for reuse

## Examples

See the `examples/` directory for complete examples:
- `house_prices.py` - Regression example
- `dataset_generation.py` - Synthetic data generation
- `santander_transactions.py` - Classification example
- `spaceship_titanic.py` - Multi-class classification

