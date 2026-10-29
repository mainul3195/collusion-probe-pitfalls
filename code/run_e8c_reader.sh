#!/bin/bash
# E8c fallback analysis (CPU): probes on reader activations of the released Shi et al. trajectories.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; mkdir -p "$THESIS/results/e8c"
cd "$THESIS/code"
python -u e8c_reader.py --shi "$R/shi_reader_act" --core "$R/activations/core" --v2 "$R"/e2v2_*_c?_act \
  --v1-controls "$R"/e2{sys,ben}_{a,b}_act --out "$THESIS/results/e8c/reader.json" 2>&1 | tee "$THESIS/results/e8c/reader.log"
