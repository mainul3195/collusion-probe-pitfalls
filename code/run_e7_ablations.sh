#!/bin/bash
# E7 ablations + calibration variants on E2 v2 (CPU only). Each line = one e2v2_analysis.py run; finished ones are skipped.
# Usage: bash code/run_e7_ablations.sh   (from ${THESIS_ROOT:?set THESIS_ROOT}; runs 3 jobs at a time)
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"; cd "$THESIS/code"
R=../runs/qwen3_14b; O=../results/e7; mkdir -p "$O"
COMMON="--core $R/activations/core --v2 $R/e2v2_*_c?_act --v1-controls $R/e2sys_a_act $R/e2sys_b_act $R/e2ben_a_act $R/e2ben_b_act --workers 5"
job() {  # job <name> <args...>
  local name="$1"; shift
  [ -s "$O/$name.json" ] && return 0
  OMP_NUM_THREADS=2 python -u e2v2_analysis.py $COMMON "$@" --out "$O/$name.json" > "$O/$name.log" 2>&1
}
{
  echo "calib_split_act --signal act --calib split"
  echo "calib_split_act_ibneg --signal act --calib split --ib neg"
  echo "calib_split_text --signal text --calib split"
  echo "calib_tmpl_act --signal act --calib tmpl"
  echo "calib_tmpl_act_ibneg --signal act --calib tmpl --ib neg"
  for L in 14 16 18 20 22 24 26 28 30; do echo "layer_$L --signal act --layers $L"; done
  echo "layers_all --signal act --layers 14,16,18,20,22,24,26,28,30"
  echo "feat_mean --signal act --feature mean"
  for n in 1 2 3; do echo "templates_$n --signal act --train-templates $n"; done
  echo "pos_success --signal act --pos success"
} | while read -r line; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
  job $line &
done
wait
echo "E7 ablations finished $(date -Is)"
