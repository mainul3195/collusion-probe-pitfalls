#!/bin/bash
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; cd "$THESIS/code"
python -u steer_direction.py --v2 "$R"/e2v2_*_c?_act --v1-controls "$R"/e2{sys,ben}_{a,b}_act --layer 18 --out "$R/steer/dir_L18.npz"
