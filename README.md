<div align="center">

# plexe ✨

[![PyPI version](https://img.shields.io/pypi/v/plexe.svg)](https://pypi.org/project/plexe/)
[![Discord](https://img.shields.io/discord/1300920499886358529?logo=discord&logoColor=white)](https://discord.gg/SefZDepGMv)

<img src="resources/backed-by-yc.png" alt="backed-by-yc" width="20%">


Build machine learning models using natural language.

[Quickstart](#1-quickstart) |
[Features](#2-features) |
[Installation](#3-installation) |
[Documentation](#4-documentation)

<br>

**plexe** lets you create machine learning models by describing them in plain language. Simply explain what you want, 
and the AI-powered system builds a fully functional model through an automated agentic approach. Also available as a 
[managed cloud service](https://plexe.ai).

<br>

Watch the demo on YouTube:
[![Building an ML model with Plexe](resources/demo-thumbnail.png)](https://www.youtube.com/watch?v=bUwCSglhcXY)
</div>

## 1. Quickstart

### Installation
```bash
pip install plexe
```

### Using plexe

You can use plexe as a Python library to build and train machine learning models:

```python
import plexe

# Define the model
model = plexe.Model(
    intent="Predict sentiment from news articles",
    input_schema={"headline": str, "content": str},
    output_schema={"sentiment": str}
)

# Build and train the model
model.build(
    datasets=[your_dataset],
    provider="openai/gpt-4o-mini",
    max_iterations=10
)

# Use the model
prediction = model.predict({
    "headline": "New breakthrough in renewable energy",
    "content": "Scientists announced a major advancement..."
})

# Save for later use
plexe.save_model(model, "sentiment-model")
loaded_model = plexe.load_model("sentiment-model.tar.gz")
```

## 2. Features

### 2.1. 💬 Natural Language Model Definition
Define models using plain English descriptions:

```python
model = plexe.Model(
    intent="Predict housing prices based on features like size, location, etc.",
    input_schema={"square_feet": int, "bedrooms": int, "location": str},
    output_schema={"price": float}
)
```

### 2.2. 🤖 Multi-Agent Architecture
The system uses a team of specialized AI agents to:
- Analyze your requirements and data
- Plan the optimal model solution
- Generate and improve model code
- Test and evaluate performance
- Package the model for deployment

### 2.3. 🎯 Automated Model Building
Build complete models with a single method call:

```python
model.build(
    datasets=[dataset_a, dataset_b],
    provider="openai/gpt-4o-mini",  # LLM provider
    max_iterations=10,              # Max solutions to explore
    timeout=1800                    # Optional time limit in seconds
)
```

### 2.4. 🚀 Distributed Training with Ray

Plexe supports distributed model training and evaluation with Ray for faster parallel processing:

```python
from plexe import Model

# Optional: Configure Ray cluster address if using remote Ray
# from plexe import config
# config.ray.address = "ray://10.1.2.3:10001"

model = Model(
    intent="Predict house prices based on various features",
    distributed=True  # Enable distributed execution
)

model.build(
    datasets=[df],
    provider="openai/gpt-4o-mini"
)
```

Ray distributes your workload across available CPU cores, significantly speeding up model generation and evaluation when exploring multiple model variants.

### 2.5. 🎲 Data Generation & Schema Inference
Generate synthetic data or infer schemas automatically:

```python
# Generate synthetic data
dataset = plexe.DatasetGenerator(
    description="Example dataset with features and target",
    provider="openai/gpt-4o-mini",
    schema={"features": str, "target": int}
)
dataset.generate(500)  # Generate 500 samples

# Infer schema from intent
model = plexe.Model(intent="Predict customer churn based on usage patterns")
model.build(provider="openai/gpt-4o-mini")  # Schema inferred automatically
```

### 2.6. 🌐 Multi-Provider Support
Use your preferred LLM provider, for example:
```python
model.build(provider="openai/gpt-4o-mini")          # OpenAI
model.build(provider="anthropic/claude-3-opus")     # Anthropic
model.build(provider="ollama/llama2")               # Ollama
model.build(provider="huggingface/meta-llama/...")  # Hugging Face    
```
See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for instructions and available providers.

> [!NOTE]
> Plexe *should* work with most LiteLLM providers, but we actively test only with `openai/*` and `anthropic/*`
> models. If you encounter issues with other providers, please let us know.


## 3. Installation

### 3.1. Installation Options
```bash
pip install plexe                  # Standard installation
pip install plexe[lightweight]     # Minimal dependencies
pip install plexe[all]             # With deep learning support
```

### 3.2. API Keys
```bash
# Set your preferred provider's API key
export OPENAI_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
export GEMINI_API_KEY=<your-key>
```
See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for environment variable names.

## 4. Documentation
For full documentation, visit [docs.plexe.ai](https://docs.plexe.ai).

## 5. Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Join our [Discord](https://discord.gg/SefZDepGMv) to connect with the team.

## 6. License
[Apache-2.0 License](LICENSE)

## 7. Citation
If you use Plexe in your research, please cite it as follows:

```bibtex
@software{plexe2025,
  author = {De Bernardi, Marcello AND Dubey, Vaibhav},
  title = {Plexe: Build machine learning models using natural language.},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/plexe-ai/plexe}},
}
