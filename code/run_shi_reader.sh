#!/bin/bash
# E8c fallback: reader activations for the released Shi et al. trajectories (see extract_shi.py).
# Usage (gpuq, --gpus 1):  SHARD=0/2 bash code/run_shi_reader.sh
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
cd "$THESIS/code"
python -u extract_shi.py --model "${EXTRACT_MODEL:-Qwen/Qwen3-14B}" --quantise 4bit --layers "${LAYERS:-16,18,20}" \
  --data "$THESIS/ext/agent-collusion-data/data/episodes" --out "$THESIS/runs/qwen3_14b/shi_reader_act" \
  --shard "${SHARD:-0/1}"
echo "shi reader COMPLETE $(date -Is) shard=${SHARD:-0/1}"
