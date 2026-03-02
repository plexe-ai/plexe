#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CATBOOST_INFO_DIR="$ROOT_DIR/catboost_info"

cleanup_catboost_info() {
  if [[ "${PLEXE_IT_KEEP_CATBOOST_INFO:-0}" == "1" ]]; then
    return
  fi
  rm -rf "$CATBOOST_INFO_DIR"
}

# Remove stale CatBoost local artifacts from previous runs.
cleanup_catboost_info
# Keep repo clean even if a stage fails midway.
trap cleanup_catboost_info EXIT

if [[ -z "${PLEXE_IT_RUN_ID:-}" ]]; then
  PLEXE_IT_RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi
export PLEXE_IT_RUN_ID

ARTIFACT_ROOT="$ROOT_DIR/.pytest_cache/integration/$PLEXE_IT_RUN_ID"
mkdir -p "$ARTIFACT_ROOT"

if ! poetry run python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('xdist') else 1)"; then
  echo "ERROR: pytest-xdist is required for staged integration tests."
  echo "Install dependencies with: poetry install"
  echo "Then verify with: poetry run pytest --help | grep -E '(^| )-n( |$)'"
  exit 2
fi

if [[ -n "${PLEXE_IT_WORKERS:-}" ]]; then
  WORKERS="${PLEXE_IT_WORKERS}"
elif [[ "${PLEXE_IT_VERBOSE:-0}" == "1" ]]; then
  # In verbose mode, default to main-process execution for reliable live logs.
  WORKERS="0"
else
  WORKERS="auto"
fi
PYTEST_PARALLEL_ARGS=(-n "$WORKERS")
PYTEST_LOG_DISABLE_ARGS=(
  --log-disable=LiteLLM
  --log-disable=litellm
  --log-disable=httpx
  --log-disable=httpcore
  --log-disable=urllib3
  --log-disable=py4j
  --log-disable=py4j.clientserver
  --log-disable=py4j.java_gateway
)

run_stage() {
  local marker="$1"
  local cmd=(poetry run pytest tests/integration -m "$marker" "${PYTEST_PARALLEL_ARGS[@]}" --maxfail=1)

  if [[ "${PLEXE_IT_VERBOSE:-0}" == "1" ]]; then
    cmd+=(-s -vv -o log_cli=true -o log_cli_level=INFO --capture=tee-sys "${PYTEST_LOG_DISABLE_ARGS[@]}")
  fi

  "${cmd[@]}"
}

echo "Running staged integration tests with run id: $PLEXE_IT_RUN_ID"
echo "Artifacts: $ARTIFACT_ROOT"
echo "Workers: $WORKERS"
if [[ "${PLEXE_IT_VERBOSE:-0}" == "1" ]]; then
  echo "Verbose mode: enabled (live logs and test output)"
fi

echo ""
echo "Stage 1/3: building reusable seeds through phase 3"
run_stage "integration_seed"

echo ""
echo "Stage 2/3: resuming from seeds through phase 4"
run_stage "integration_search"

echo ""
echo "Stage 3/3: final evaluation, packaging, and predictor checks"
run_stage "integration_eval"

echo ""
echo "Staged integration suite completed successfully."
