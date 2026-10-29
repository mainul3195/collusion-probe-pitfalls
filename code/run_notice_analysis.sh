#!/bin/bash
# Notice-only control (PLAN, after the monitor-aware slice): same notice to every agent, NO evasion line for
# colluders. Control and benign_coord runs of the aware slice get the identical treatment (the evasion line is part
# of the collusion briefing only), so they are reused as the notice-only negatives. CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/aware"; mkdir -p "$O"
cd "$THESIS/code"
python -u e2v2_analysis.py --signal act --core "$R/activations/core" --v2 "$R"/e2v2_*_c?_act \
  --v1-controls "$R"/e2{sys,ben}_{a,b}_act --aware "$R"/e2v2an_*_act "$R"/e2v2aw_ctl0_act "$R"/e2v2aw_ben?_act \
  --workers 5 --out "$O/notice_only_act.json" > "$O/notice_only_act.log" 2>&1
echo "notice analysis COMPLETE $(date -Is)"
