#!/bin/bash
# Paper 1 A2: Qwen3-14B on exactly the Mistral-Small-24B cells (matched design and size). CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/paper1/matched14b"; mkdir -p "$O"
cd "$THESIS/code"
python matched_family_exclude.py "$R" > "$O/exclude.txt"
echo "excluded runs: $(wc -l < "$O/exclude.txt")"
run() {
  local name="$1"; shift
  python -u e2v2_analysis.py "$@" --exclude-runs "$O/exclude.txt" --core none --v2 "$R"/e2v2_*_c?_act \
    --v1-controls "$R"/e2{sys,ben}_{a,b}_act --workers 5 --out "$O/$name.json" > "$O/$name.log" 2>&1
  echo "done $name $(date -Is)"
}
run act --signal act --layers 16,18,20
run act_paper_recipe --signal act --layers 16,18,20 --neg paper
run text --signal text
echo "matched family COMPLETE $(date -Is)"
