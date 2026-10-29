#!/bin/bash
# Paper 1: rerun every analysis that uses the free-choice family on the full data (choice c0-c3 + pilot, 800 + 100
# episodes, stance-labelled), plus the benign free-choice family as extra negatives. CPU only; overwrites
# results/paper1/{instruction,decision} (previous versions in git, commit 2ca26a2).
#   1. H6 agent level (instruction_tests.py choice, with stance and vote-only)
#   2. decision probe, agent level (instruction_tests.py decision): any / strict / vote, +-non-briefed negatives,
#      any / strict with benign choice
#   3. decision probe through the full pipeline (run_p1_decision_analysis.sh): plain, DN=1, DN=1 BC=1
# Usage: nohup bash code/run_p1_fullchoice.sh > results/paper1/fullchoice.out 2>&1 &
set -uo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
R="$THESIS/runs/qwen3_14b"; I="$THESIS/results/paper1/instruction"
cd "$THESIS/code"
step() { echo "== $(date -Is) $*"; "$@" || echo "!! FAILED rc=$?: $*"; }
step python -u instruction_tests.py choice --runs "$R" --stance "$R/stance_mistral24b" --out "$I"
step python -u instruction_tests.py choice --runs "$R" --out "$I"
for j in any strict vote; do
  step python -u instruction_tests.py decision --runs "$R" --stance "$R/stance_mistral24b" --joiner $j --out "$I"
  step python -u instruction_tests.py decision --runs "$R" --stance "$R/stance_mistral24b" --joiner $j --decision-neg --out "$I"
done
for j in any strict; do
  step python -u instruction_tests.py decision --runs "$R" --stance "$R/stance_mistral24b" --joiner $j --decision-neg --benign-choice --out "$I"
done
step bash run_p1_decision_analysis.sh any strict vote
step env DN=1 bash run_p1_decision_analysis.sh any strict
step env DN=1 BC=1 bash run_p1_decision_analysis.sh any strict
echo "fullchoice COMPLETE $(date -Is)"
