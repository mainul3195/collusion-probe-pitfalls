#!/bin/bash
# Generic gpuq wrapper: run a script from code/ inside the thesis environment. Usage: bash code/run_py.sh script.py args...
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
cd "$THESIS/code"
exec python -u "$@"
