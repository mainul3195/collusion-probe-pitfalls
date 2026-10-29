#!/bin/bash
# Paper 1 A6b: code-review task analysis (within task, same protocol as E2 v2) + zero-shot transfer both ways. CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/paper1/codereview"; T="$THESIS/results/paper1/transfer"; mkdir -p "$O" "$T"
cd "$THESIS/code"
run() {
  local name="$1"; shift
  python -u e2v2_analysis.py "$@" --core none --v2 "$R"/cr_*_c?_act --workers 5 --out "$O/$name.json" > "$O/$name.log" 2>&1
  echo "done $name $(date -Is)"
}
run act --signal act --layers 16,18,20
run act_tmpl --signal act --layers 16,18,20 --calib tmpl
run act_paper_recipe --signal act --layers 16,18,20 --neg paper
run act_deep --signal act --layers 24,26,28
run text --signal text
run vote --signal vote
python -u transfer_task.py --source "$R"/e2v2_*_c?_act --source-v1 "$R"/e2{sys,ben}_{a,b}_act \
  --target "$R"/cr_*_c?_act --name committee_to_codereview --out "$T" > "$T/committee_to_codereview.stdout" 2>&1
python -u transfer_task.py --source "$R"/cr_*_c?_act --target "$R"/e2v2_*_c?_act --target-v1 "$R"/e2{sys,ben}_{a,b}_act \
  --name codereview_to_committee --out "$T" > "$T/codereview_to_committee.stdout" 2>&1
L=$(grep -m1 -A6 "PRE-VOTE, positives = ALL" "$O/act.log" | grep -E "peak\|cummean|ca_t\|cummean" | tr -s ' ' | cut -c1-80 | tr '\n' ';')
X=$(sed -n 3p "$T/committee_to_codereview.log" | tr -s ' '); Y=$(sed -n 3p "$T/codereview_to_committee.log" | tr -s ' ')
echo "cr analysis COMPLETE $(date -Is)"
