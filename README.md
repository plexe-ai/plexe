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
provide a dataset, and the AI-powered system builds a fully functional model through an automated agentic approach.
Also available as a [managed cloud service](https://plexe.ai).

<br>

Watch the demo on YouTube:
[![Building an ML model with Plexe](resources/demo-thumbnail.png)](https://www.youtube.com/watch?v=bUwCSglhcXY)
</div>

## 1. Quickstart

### Installation
```bash
pip install plexe
export OPENAI_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
```

### Using plexe

Provide a tabular dataset (Parquet, CSV, ORC, or Avro) and a natural language intent:

```bash
python -m plexe.main \
    --train-dataset-uri data.parquet \
    --intent "predict whether a passenger was transported" \
    --max-iterations 5
```

```python
from plexe.main import main
from pathlib import Path

best_solution, metrics, report = main(
    intent="predict whether a passenger was transported",
    data_refs=["train.parquet"],
    max_iterations=5,
    work_dir=Path("./workdir"),
)
print(f"Performance: {best_solution.performance:.4f}")
```

## 2. Features

### 2.1. 🤖 Multi-Agent Architecture
The system uses 14 specialized AI agents across a 6-phase workflow to:
- Analyze your data and identify the ML task
- Select the right evaluation metric
- Search for the best model through hypothesis-driven iteration
- Evaluate model performance and robustness
- Package the model for deployment

### 2.2. 🎯 Automated Model Building
Build complete models with a single call. Plexe supports **XGBoost**, **CatBoost**, **LightGBM**, **Keras**, and **PyTorch** for tabular data:

```python
best_solution, metrics, report = main(
    intent="predict house prices based on property features",
    data_refs=["housing.parquet"],
    max_iterations=10,                    # Search iterations
    allowed_model_types=["xgboost"],      # Or let plexe choose
    enable_final_evaluation=True,         # Evaluate on held-out test set
)
```

Run `python -m plexe.main --help` for all CLI options.

The output is a self-contained model package at `work_dir/model/` (also archived as `model.tar.gz`).
The package has no dependency on `plexe` — build the model with plexe, deploy it anywhere:

```
model/
├── artifacts/          # Trained model + feature pipeline (pickle)
├── src/                # Inference predictor, pipeline code, training template
├── schemas/            # Input/output JSON schemas
├── config/             # Hyperparameters
├── evaluation/         # Metrics and detailed analysis reports
├── model.yaml          # Model metadata
└── README.md           # Usage instructions with example code
```

### 2.3. 🐳 Batteries-Included Docker Images
Run plexe with everything pre-configured — PySpark, Java, and all dependencies included.
A `Makefile` is provided for common workflows:

```bash
make build          # Build the Docker image
make test-quick     # Fast sanity check (~1 iteration)
make run-titanic    # Run on Spaceship Titanic dataset
```

Or run directly:

```bash
docker run --rm \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    -v $(pwd)/data:/data -v $(pwd)/workdir:/workdir \
    plexe:py3.12 python -m plexe.main \
        --train-dataset-uri /data/dataset.parquet \
        --intent "predict customer churn" \
        --work-dir /workdir \
        --spark-mode local
```

A `config.yaml` in the project root is automatically mounted. A Databricks Connect image
is also available: `docker build --target databricks .`

### 2.4. ⚙️ YAML Configuration
Customize LLM routing, search parameters, Spark settings, and more via a config file:

```yaml
# config.yaml
max_search_iterations: 5
allowed_model_types: [xgboost, catboost]
spark_driver_memory: "4g"
hypothesiser_llm: "openai/gpt-5-mini"
feature_processor_llm: "anthropic/claude-sonnet-4-5-20250929"
```

```bash
CONFIG_FILE=config.yaml python -m plexe.main ...
```

See [`config.yaml.template`](config.yaml.template) for all available options.

### 2.5. 🌐 Multi-Provider LLM Support
Plexe uses LLMs via [LiteLLM](https://docs.litellm.ai/docs/providers), so you can use any supported provider:

```yaml
# Route different agents to different providers
hypothesiser_llm: "openai/gpt-5-mini"
feature_processor_llm: "anthropic/claude-sonnet-4-5-20250929"
model_definer_llm: "ollama/llama3"
```

> [!NOTE]
> Plexe *should* work with most LiteLLM providers, but we actively test only with `openai/*` and `anthropic/*`
> models. If you encounter issues with other providers, please let us know.

### 2.6. 📊 Experiment Dashboard
Visualize experiment results, search trees, and evaluation reports with the built-in Streamlit dashboard:

```bash
python -m plexe.viz --work-dir ./workdir
```

### 2.7. 🔌 Extensibility
Connect plexe to custom storage, tracking, and deployment infrastructure via the `WorkflowIntegration` interface:

```python
main(intent="...", data_refs=[...], integration=MyCustomIntegration())
```

See [`plexe/integrations/base.py`](plexe/integrations/base.py) for the full interface.

## 3. Installation

### 3.1. Installation Options
```bash
pip install plexe                    # Core (XGBoost, CatBoost, LightGBM, Keras, PyTorch, scikit-learn)
pip install plexe[pyspark]           # + Local PySpark execution
pip install plexe[aws]               # + S3 storage support (boto3)
```

Requires Python >= 3.10, < 3.13.

### 3.2. API Keys
```bash
export OPENAI_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
```
See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for all supported providers.

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
```
