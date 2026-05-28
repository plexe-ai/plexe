# Plexe MLE-bench adapter

This directory contains a minimal MLE-bench agent adapter for running Plexe
against the OpenAI MLE-bench harness.

It follows the agent contract used by `openai/mle-bench`:

- `config.yaml` registers the agent id as `plexe`.
- `Dockerfile` builds an agent image from `mlebench-env`.
- `start.sh` is the container entrypoint called by MLE-bench.
- `plexe/run_mlebench.py` reads `/home/data`, runs Plexe, writes
  `/home/submission/submission.csv`, and records diagnostics in `/home/logs`.

## Build inside an MLE-bench checkout

Copy or symlink this directory into `openai/mle-bench/agents/plexe`, then build:

```bash
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

docker build --platform=linux/amd64 -t plexe \
  agents/plexe/ \
  --build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
  --build-arg LOGS_DIR=$LOGS_DIR \
  --build-arg CODE_DIR=$CODE_DIR \
  --build-arg AGENT_DIR=$AGENT_DIR
```

## Run a smoke competition

```bash
python run_agent.py \
  --agent-id plexe \
  --competition-set experiments/splits/spaceship-titanic.txt \
  --n-seeds 1 \
  --n-workers 1
```

Then compile and grade the run group:

```bash
python experiments/make_submission.py \
  --metadata runs/<run-group>/metadata.json \
  --output runs/<run-group>/submission.jsonl

mlebench grade \
  --submission runs/<run-group>/submission.jsonl \
  --output-dir runs/<run-group>
```

## Environment

The adapter expects the benchmark-provided paths:

- `DATA_DIR`, default `/home/data`
- `SUBMISSION_DIR`, default `/home/submission`
- `LOGS_DIR`, default `/home/logs`
- `CODE_DIR`, default `/home/code`
- `PLEXE_WORK_DIR`, default `/home/code/plexe-work`
- `PLEXE_MAX_ITERATIONS`, default `10`
- `PLEXE_PROVIDER`, optional provider string for documentation/logging

The Dockerfile has a `PLEXE_PACKAGE` build arg, defaulting to
`plexe[pyspark,tabular]`. Override it when you need to benchmark a specific
branch or wheel.

The actual LLM credentials are read by Plexe/LiteLLM from the normal provider
environment variables.

## Scope

This adapter does not include benchmark results. It is intentionally structured
so results can be produced separately with the official MLE-bench runner and
submitted without mixing code changes with private API-key or Kaggle execution
state.
