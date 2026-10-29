#!/bin/bash
# Paper 1 A5: headline analysis with the extra negatives (e2v2x_* dirs) -> 0.1% FPR with usable intervals. CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/paper1/negx"; mkdir -p "$O"
cd "$THESIS/code"
V1="$R/e2sys_a_act $R/e2sys_b_act $R/e2ben_a_act $R/e2ben_b_act"
python -u e2v2_analysis.py --signal act --core "$R/activations/core" --v2 "$R"/e2v2_*_c?_act "$R"/e2v2x_*_act \
  --v1-controls $V1 --workers 5 --out "$O/act.json" > "$O/act.log" 2>&1
python -u e2v2_analysis.py --signal act --calib tmpl --core "$R/activations/core" --v2 "$R"/e2v2_*_c?_act "$R"/e2v2x_*_act \
  --v1-controls $V1 --workers 5 --out "$O/act_tmpl.json" > "$O/act_tmpl.log" 2>&1
python -u e2v2_analysis.py --signal text --core "$R/activations/core" --v2 "$R"/e2v2_*_c?_act "$R"/e2v2x_*_act \
  --v1-controls $V1 --workers 5 --out "$O/text.json" > "$O/text.log" 2>&1
L=$(grep -m1 "counts" "$O/act.log" | cut -c1-200)
echo "negx analysis COMPLETE $(date -Is)"
