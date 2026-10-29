#!/bin/bash
# Outcome table for all steering conditions -> results/steer/outcomes.{log,json}
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
cd "$THESIS"
python code/steer_outcomes.py --root runs/qwen3_14b/steer --out results/steer/outcomes
echo "steer outcomes COMPLETE $(date -Is)"
