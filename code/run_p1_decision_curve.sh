#!/bin/bash
# Paper 1 decision-probe learning curve: committee pre-vote recall at 1% FPR against the number of training joiners
# (random subsets; refusers kept). Each size below the full joiner count is run with subsample seeds 0, 1, 2 (the full
# count once: all seeds are the same set). Pre-registered in papers/paper1_detection/PREREG.md (amendment 1). CPU only.
# Usage: [TAG=_dn] bash code/run_p1_decision_curve.sh "strict:100,200,350" [extra e2v2_analysis flags]
#        (the last size must be the full joiner count)
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/paper1/decision_curve"; mkdir -p "$O"
cd "$THESIS/code"
V1="$R/e2sys_a_act $R/e2sys_b_act $R/e2ben_a_act $R/e2ben_b_act"
CH=$(ls -d "$R"/choice_*_act); TAG="${TAG:-}"
for spec in "$1"; do
  j="${spec%%:*}"
  sizes=$(echo "${spec#*:}" | tr ',' ' '); full=${sizes##* }
  for n in $sizes; do
    seeds="0 1 2"; [ "$n" = "$full" ] && seeds="0"
    for sd in $seeds; do
      python -u e2v2_analysis.py --signal decision --choice $CH --joiner "$j" --stance "$R/stance_mistral24b" \
        --max-joiners "$n" --subsample-seed "$sd" --workers 5 --core none --v2 "$R"/e2v2_*_c?_act --v1-controls $V1 \
        "${@:2}" --out "$O/com${TAG}_${j}_${n}_s$sd.json" > "$O/com${TAG}_${j}_${n}_s$sd.log" 2>&1
    done
  done
done
echo "decision curve COMPLETE $(date -Is)"
