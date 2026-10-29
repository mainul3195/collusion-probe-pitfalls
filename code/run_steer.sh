#!/bin/bash
# Causal test along the probe direction (gen_e2v2.py --steer, in-process vLLM engine with hooks). Generation only;
# outcome = votes in run_config.json. gpuq sets CUDA_VISIBLE_DEVICES.
# Env: NAME (output dir runs/qwen3_14b/steer/<NAME>), MODES, STEER (none|ablate|add), VEC (probe|random),
#      LAYERS_STEER ("all" or "18"), ALPHA (multiples of delta), REP_START, REPS, SCENARIOS, PARALLEL.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
: "${NAME:?}" "${MODES:?}" "${STEER:?}"
R="$THESIS/runs/qwen3_14b/steer"; LOG="$THESIS/runs/qwen3_14b/logs"; mkdir -p "$R" "$LOG"
cd "$THESIS/code"
echo "steer start $(date -Is) name=$NAME modes=$MODES steer=$STEER vec=${VEC:-probe} layers=${LAYERS_STEER:-all} alpha=${ALPHA:-0} gpus=${CUDA_VISIBLE_DEVICES:-}"
for pass in 1 2; do  # second pass re-runs episodes that failed
  python -u gen_e2v2.py --model Qwen/Qwen3-14B-AWQ --out "$R/$NAME" --modes "$MODES" --replicates "${REPS:-2}" \
    --rep-start "${REP_START:-0}" --scenarios "${SCENARIOS:-50}" --parallel "${PARALLEL:-32}" \
    --steer "$STEER" --steer-dir "$R/dir_L18.npz" --steer-vec "${VEC:-probe}" --steer-layers "${LAYERS_STEER:-all}" \
    --steer-alpha "${ALPHA:-0}" --steer-tag "$NAME" >> "$LOG/steer_${NAME}.log" 2>&1
done
echo "steer COMPLETE $(date -Is) runs=$(find "$R/$NAME" -name agent_prompts.json | wc -l)"
