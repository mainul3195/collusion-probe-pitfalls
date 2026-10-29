#!/bin/bash
# Paper 1 decision probe through the full pipeline (aggregators, calibration, anytime alarm): trained only on the
# free-choice family, tested on the committee benchmark (main and extended negatives) and zero-shot on code review.
# Joiner definitions: any (vote or private agreement), strict (both; mixed left out), vote. CPU only.
# DN=1: train with non-briefed benchmark negatives too (--decision-neg); output names <set>_dn_<joiner>.json.
# BC=1: also the benign free-choice choosers as negatives (--benign-choice benchoice_*_act); adds _bc to the names.
# Usage: [DN=1] bash code/run_p1_decision_analysis.sh [joiner ...]   (default: any strict vote)
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/paper1/decision"; mkdir -p "$O"
cd "$THESIS/code"
V1="$R/e2sys_a_act $R/e2sys_b_act $R/e2ben_a_act $R/e2ben_b_act"
CH=$(ls -d "$R"/choice_*_act)
for J in "${@:-any strict vote}"; do for j in $J; do
  D=(--signal decision --choice $CH --joiner "$j" --stance "$R/stance_mistral24b" --workers 5)
  X=""; [ -n "${DN:-}" ] && { D+=(--decision-neg); X="_dn"; }
  [ -n "${BC:-}" ] && { D+=(--benign-choice $(ls -d "$R"/benchoice_*_act)); X="${X}_bc"; }
  python -u e2v2_analysis.py "${D[@]}" --core none --v2 "$R"/e2v2_*_c?_act --v1-controls $V1 \
    --out "$O/com${X}_$j.json" > "$O/com${X}_$j.log" 2>&1
  python -u e2v2_analysis.py "${D[@]}" --calib tmpl --core none --v2 "$R"/e2v2_*_c?_act --v1-controls $V1 \
    --out "$O/com_tmpl${X}_$j.json" > "$O/com_tmpl${X}_$j.log" 2>&1
  python -u e2v2_analysis.py "${D[@]}" --core none --v2 "$R"/e2v2_*_c?_act "$R"/e2v2x_*_act --v1-controls $V1 \
    --out "$O/negx${X}_$j.json" > "$O/negx${X}_$j.log" 2>&1
  python -u e2v2_analysis.py "${D[@]}" --core none --v2 "$R"/cr_*_c?_act --out "$O/cr${X}_$j.json" > "$O/cr${X}_$j.log" 2>&1
done; done
echo "decision analysis COMPLETE $(date -Is)"
