#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate agent

python "${AGENT_DIR}/run_mlebench.py"

bash /home/validate_submission.sh "${SUBMISSION_DIR}/submission.csv"
