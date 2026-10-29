#!/bin/bash
# E2: generate variable-coalition episodes with vLLM, then extract activations.
#
# Usage:
#   GPUS=1 PORT=8021 TAG=qwen3_14b GEN_MODEL=Qwen/Qwen3-14B-AWQ EXTRACT_MODEL=Qwen/Qwen3-14B \
#     QUANTISE=4bit LAYERS=14,16,18,20,22,24,26,28,30 REPS=4 WAIT_PID=<pid> bash run_e2.sh
# WAIT_PID: optional, wait for that process (e.g. an earlier pipeline on the same GPU) to exit first.
# GEN_SCRIPT: gen_e2.py (v1, default) or gen_e2v2.py. SCENARIOS: number of scenarios (default 50).
# SKIP_EXTRACT=1: generation only (pilots). GEN_EXTRA: extra generator flags, e.g. --aware.
# MAX_LEN: vLLM context (default 8192). EXTRACT_EXTRA: extra extract_e2.py flags, e.g. --max-tokens 16384.
# MAX_TOKENS_FINAL (read by gen_e2v2.py): token cap of the final vote message.
# Two GPUs in GPUS -> vLLM tensor parallel 2 (32B model). VLLM_EXTRA: extra vLLM flags, e.g. --enforce-eager.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
: "${GPUS:?}" "${TAG:?}" "${GEN_MODEL:?}" "${EXTRACT_MODEL:?}" "${LAYERS:?}"
PORT="${PORT:-8021}"; REPS="${REPS:-4}"; REP_START="${REP_START:-0}"; QUANTISE="${QUANTISE:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"; PARALLEL="${PARALLEL:-16}"
INJECT="${INJECT:-system}"; MODES="${MODES:-collusion,benign_coord,control}"
RAW_NAME="${RAW_NAME:-e2raw}"; ACT_NAME="${ACT_NAME:-e2act}"
GEN_SCRIPT="${GEN_SCRIPT:-gen_e2.py}"; SCENARIOS="${SCENARIOS:-50}"
OUT="$THESIS/runs/$TAG"; LOG="$OUT/logs"; mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l); TP=""; [ "$NGPU" -gt 1 ] && TP="--tensor-parallel-size $NGPU"

if [ -n "${WAIT_PID:-}" ]; then
  echo "waiting for pid $WAIT_PID $(date -Is)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
echo "E2 start $(date -Is) gen=$GEN_MODEL extract=$EXTRACT_MODEL q=$QUANTISE layers=$LAYERS reps=$REP_START+$REPS gpus=$GPUS inject=$INJECT modes=$MODES raw=$RAW_NAME" | tee -a "$LOG/e2_info.txt"

if [ -z "${SKIP_GEN:-}" ]; then
  python -m vllm.entrypoints.openai.api_server --model "$GEN_MODEL" --port "$PORT" \
    --max-model-len "${MAX_LEN:-8192}" --gpu-memory-utilization "$GPU_MEM_UTIL" --max-num-seqs "${MAX_SEQS:-32}" --disable-log-requests $TP ${VLLM_EXTRA:-} \
    > "$LOG/${RAW_NAME//\//_}_vllm.log" 2>&1 &
  VPID=$!; trap 'kill $VPID 2>/dev/null || true' EXIT
  for i in $(seq 1 900); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready ${i}s"; break; }
    kill -0 $VPID 2>/dev/null || { echo "vLLM died"; tail -20 "$LOG/${RAW_NAME//\//_}_vllm.log"; exit 1; }
    sleep 1
  done
  curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 || { echo "vLLM not ready after 900s"; tail -20 "$LOG/${RAW_NAME//\//_}_vllm.log"; exit 1; }
  cd "$THESIS/code"
  python -u "$GEN_SCRIPT" --model "$GEN_MODEL" --port "$PORT" --out "$OUT/$RAW_NAME" --inject "$INJECT" --modes "$MODES" \
    --replicates "$REPS" --rep-start "$REP_START" --scenarios "$SCENARIOS" --parallel "$PARALLEL" ${GEN_EXTRA:-} > "$LOG/${RAW_NAME//\//_}_gen.log" 2>&1
  # second pass picks up any runs that errored (resume skips finished ones)
  python -u "$GEN_SCRIPT" --model "$GEN_MODEL" --port "$PORT" --out "$OUT/$RAW_NAME" --inject "$INJECT" --modes "$MODES" \
    --replicates "$REPS" --rep-start "$REP_START" --scenarios "$SCENARIOS" --parallel "$PARALLEL" ${GEN_EXTRA:-} >> "$LOG/${RAW_NAME//\//_}_gen.log" 2>&1
  kill $VPID; wait $VPID 2>/dev/null || true; trap - EXIT; sleep 5
  echo "E2 gen done $(date -Is) runs=$(ls "$OUT/$RAW_NAME" | wc -l)"
fi

[ -n "${SKIP_EXTRACT:-}" ] && { echo "E2 COMPLETE (gen only) $(date -Is)"; exit 0; }
cd "$THESIS/code"
QARG=""; [ -n "$QUANTISE" ] && QARG="--quantise $QUANTISE"
python -u extract_e2.py --model "$EXTRACT_MODEL" $QARG --layers "$LAYERS" ${EXTRACT_EXTRA:-} \
  --runs-dir "$OUT/$RAW_NAME" --out "$OUT/$ACT_NAME" > "$LOG/${ACT_NAME//\//_}_extract.log" 2>&1
echo "E2 COMPLETE $(date -Is)"
