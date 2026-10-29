#!/bin/bash
# Paper 1 A2: analysis of a second agent family (reduced E2 v2 design, as E8b). Env: TAG (runs/<TAG>), NAME (results/paper1/<NAME>).
# Layers 16,18,20 of 40 (Mistral-Small-24B and Phi-4 both have 40 layers, like Qwen3-14B). CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
: "${TAG:?}" "${NAME:?}"
R="$THESIS/runs/$TAG"; O="$THESIS/results/paper1/$NAME"; mkdir -p "$O"
cd "$THESIS/code"
run() {
  local name="$1"; shift
  python -u e2v2_analysis.py "$@" --core none --v2 "$R"/e2v2_*_c?_act --workers 5 --out "$O/$name.json" > "$O/$name.log" 2>&1
  echo "done $name $(date -Is)"
}
run act --signal act --layers 16,18,20
run act_tmpl --signal act --layers 16,18,20 --calib tmpl
run act_paper_recipe --signal act --layers 16,18,20 --neg paper
run act_deep --signal act --layers 24,26,28
run text --signal text
run vote --signal vote
L=$(grep -m1 -A6 "PRE-VOTE, positives = ALL" "$O/act.log" | grep -E "peak\|cummean|ca_t\|cummean" | tr -s ' ' | cut -c1-80 | tr '\n' ';')
echo "family analysis COMPLETE $(date -Is)"
