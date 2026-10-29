#!/bin/bash
# E8b: Qwen3-32B confirmation, reduced E2 v2 (500 episodes). Layers 26/29/32 of 64 = same relative depth as 16/18/20 of 40.
# The Core (paper) probe is not run: there are no Core activations for this model here. CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_32b"; O="$THESIS/results/e8b"; mkdir -p "$O"
cd "$THESIS/code"
run() {  # name, extra args...
  local name="$1"; shift
  python -u e2v2_analysis.py "$@" --core none --v2 "$R"/e2v2_*_c?_act --workers 5 --out "$O/$name.json" > "$O/$name.log" 2>&1
  echo "done $name $(date -Is)"
}
run act --signal act --layers 26,29,32
run act_tmpl --signal act --layers 26,29,32 --calib tmpl
run act_deep --signal act --layers 35,38,42
run text --signal text --layers 26,29,32
run vote --signal vote --layers 26,29,32
echo "e8b analysis COMPLETE $(date -Is)"
