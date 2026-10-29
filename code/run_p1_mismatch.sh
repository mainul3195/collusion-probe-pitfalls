#!/bin/bash
# Paper 1 robustness to the option-label mismatch (label_mismatch.py): rerun the instruction tests without the flagged
# runs (brief names an option label that is not on the menu). Same commands as run_p1_fullchoice.sh / the H5 run, plus
# --exclude; outputs under results/paper1/instruction/clean/ with the original names. CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
R="$THESIS/runs/qwen3_14b"; I="$THESIS/results/paper1/instruction/clean"; mkdir -p "$I"
X=(--exclude "$THESIS/results/paper1/audit/label_mismatch.json")
cd "$THESIS/code"
step() { echo "== $(date -Is) $*"; "$@"; }
step python label_mismatch.py --runs "$R" --out "$THESIS/results/paper1/audit/label_mismatch.json"
step python -u instruction_tests.py h5 --runs "$R" "${X[@]}" --out "$I"
step python -u instruction_tests.py choice --runs "$R" --stance "$R/stance_mistral24b" "${X[@]}" --out "$I"
step python -u instruction_tests.py decision --runs "$R" --stance "$R/stance_mistral24b" --joiner strict "${X[@]}" --out "$I"
step python -u instruction_tests.py h4 --audit "$THESIS/results/paper1/audit/audit.json" \
  --records "$THESIS/results/e2v2b/act.json" "$THESIS/results/paper1/codereview/act.json" "${X[@]}" --out "$I"
echo "mismatch robustness COMPLETE $(date -Is)"
