#!/bin/bash
# E8b comparison: 14B analysis on a subset matched to the 32B design and size. CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/e8b"; mkdir -p "$O"
cd "$THESIS/code"
python matched_14b_exclude.py "$R" > "$O/matched14b_exclude.txt"
echo "excluded runs: $(wc -l < "$O/matched14b_exclude.txt")"
for sig in act text; do
  python -u e2v2_analysis.py --signal $sig --exclude-runs "$O/matched14b_exclude.txt" --core "$R/activations/core" \
    --v2 "$R"/e2v2_*_c?_act --v1-controls "$R"/e2{sys,ben}_{a,b}_act --workers 5 --out "$O/matched14b_$sig.json" > "$O/matched14b_$sig.log" 2>&1
done
echo "matched 14B COMPLETE $(date -Is)"
