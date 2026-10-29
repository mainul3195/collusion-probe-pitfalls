#!/bin/bash
# Monitor-aware slice (PLAN step 3): probe, normalisation and thresholds from the non-aware E2 v2 runs,
# applied unchanged to the aware runs (gen_e2v2.py --aware). CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/aware"; mkdir -p "$O"
cd "$THESIS/code"
run() {  # name, extra args...
  local name="$1"; shift
  python -u e2v2_analysis.py "$@" --core "$R/activations/core" --v2 "$R"/e2v2_*_c?_act \
    --v1-controls "$R"/e2{sys,ben}_{a,b}_act --aware "$R"/e2v2aw_*_act --workers 5 --out "$O/$name.json" > "$O/$name.log" 2>&1
  echo "done $name $(date -Is)"
}
run act --signal act
run act_tmpl --signal act --calib tmpl
run core --signal core
run text --signal text
run vote --signal vote
echo "aware analysis COMPLETE $(date -Is)"
