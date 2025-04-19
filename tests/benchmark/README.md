# MLE-Bench Runner for Plexe

This script benchmarks `plexe` on the `mle-bench` test suite released by OpenAI. It automates the process of running the Plexe library on Kaggle competitions and evaluating the results using the MLE-Bench framework. See the [mle bench repository](https://github.com/openai/mle-bench) or the [mle-bench paper](https://openai.com/index/mle-bench/) for more information.

## Prerequisites

To run the benchmark, you need to complete these steps:

1. Clone the `plexe` repository: `git clone https://github.com/plexe-ai/plexe.git`
2. Install `git lfs` on your machine ([installation instructions](https://git-lfs.com/))
3. Create a Kaggle account, create an API key, and save it in `~/.kaggle/kaggle.json` ([Kaggle API instructions](https://www.kaggle.com/docs/api))
4. Install Python `3.11.0` or later
5. Install poetry: `pip install poetry`
6. Set up the project: `poetry install` in the project root directory
7. Configure API keys for your LLM provider as described in the main README.md:
   ```
   # For OpenAI
   export OPENAI_API_KEY=<your-API-key>
   # For Anthropic
   export ANTHROPIC_API_KEY=<your-API-key>
   # For Gemini
   export GEMINI_API_KEY=<your-API-key>
   ```

## Usage

Run the benchmark with:

```bash
poetry run python tests/benchmark/mle_bench.py
```

When you run the script for the first time, you will be prompted to:
1. Specify a directory where the `mle-bench` repository will be cloned
2. Set LLM provider details (provider, max iterations, timeout)

The script will then:
1. Clone the `mle-bench` repository
2. Download the Kaggle datasets (by default, only the spaceship-titanic challenge)
3. Run Plexe on the datasets
4. Generate predictions and submission files
5. Evaluate the results using MLE-Bench

### Command Line Options

- `--config PATH`: Specify a custom config file path (default: mle-bench-config.yaml)
- `--rebuild`: Force re-clone the MLE-bench repository and regenerate the config file

## Configuration

The first time you run the script, it creates a file called `mle-bench-config.yaml` with the following structure:

```yaml
repo_url: https://github.com/openai/mle-bench.git
repo_dir: /path/to/mle-bench
datasets:
  - spaceship-titanic
provider: openai/gpt-4o
max_iterations: 3
timeout: 3600
```

Configuration options:

- `repo_url`: The GitHub repository URL for MLE-bench
- `repo_dir`: Local directory to clone MLE-bench into
- `datasets`: List of Kaggle competitions to run (for initial testing, only spaceship-titanic is used)
- `provider`: LLM provider to use with Plexe (format: "provider/model")
- `max_iterations`: Maximum number of model solutions to explore
- `timeout`: Maximum time in seconds for model building

The available datasets match the competitions in the [MLE-bench competitions directory](https://github.com/openai/mle-bench/tree/main/mlebench/competitions).

## Output

After running the benchmark, the results are stored in:
- `workdir/{dataset_name}/`: Model outputs and submissions for each dataset
- `grades/`: Evaluation results from MLE-bench scoring

You can find submission CSV files and saved models in the `workdir/{dataset_name}/` directory.