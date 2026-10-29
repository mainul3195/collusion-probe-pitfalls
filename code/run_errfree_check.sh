#!/bin/bash
# Robustness: headline 14B analysis without the episodes that hold "[ERROR]" placeholder messages. CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/e2v2b"; mkdir -p "$O"
cd "$THESIS/code"
python list_error_runs.py "$R"/e2v2_*_c? "$R"/e2{sys,ben}_{a,b} "$R"/e2v2aw_*[0-9] > "$O/error_runs.txt"
echo "error runs: $(wc -l < "$O/error_runs.txt")"
python -u e2v2_analysis.py --signal act --exclude-runs "$O/error_runs.txt" --core "$R/activations/core" --v2 "$R"/e2v2_*_c?_act \
  --v1-controls "$R"/e2{sys,ben}_{a,b}_act --aware "$R"/e2v2aw_*_act --workers 5 --out "$O/act_errfree.json" > "$O/act_errfree.log" 2>&1
echo "errfree check COMPLETE $(date -Is)"
