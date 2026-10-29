#!/bin/bash
# Paper 1 A7: probe seed variance (seeds 1, 2; seed 0 = reported run results/e2v2b/act.json). CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/paper1/seeds"; mkdir -p "$O"
cd "$THESIS/code"
for s in 1 2; do
  python -u e2v2_analysis.py --signal act --seed $s --core "$R/activations/core" --v2 "$R"/e2v2_*_c?_act \
    --v1-controls "$R"/e2sys_a_act "$R"/e2sys_b_act "$R"/e2ben_a_act "$R"/e2ben_b_act --workers 5 \
    --out "$O/act_seed$s.json" > "$O/act_seed$s.log" 2>&1
done
