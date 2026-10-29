#!/bin/bash
# NARCBench generation + extraction for one model on a chosen GPU set.
# Adapted from narcbench/scripts/run_model.sh: GPU pinning, per-run port,
# extra vLLM args (e.g. on-the-fly bnb quantisation), explicit extraction layers,
# selectable tiers, and a run tag for output separation.
#
# Usage:
#   GPUS=1 PORT=8021 TAG=qwen3_4b LAYERS=8-30 \
#     bash run_pipeline.sh Qwen/Qwen3-4B
#   GPUS=0 PORT=8020 TAG=qwen3_14b LAYERS=10-34 QUANTISE=4bit \
#     VLLM_EXTRA="--quantization bitsandbytes --load-format bitsandbytes" \
#     bash run_pipeline.sh Qwen/Qwen3-14B
#
# Env: GPUS, PORT, TAG, LAYERS, QUANTISE (extraction), VLLM_EXTRA,
#      GEN_MODEL (vLLM generation model, default = model id), TIERS (default "core transfer stego"), GPU_MEM_UTIL, MAX_MODEL_LEN, SKIP_GEN, SKIP_EXTRACT

set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"

MODEL_ID="${1:?model id}"
GEN_MODEL="${GEN_MODEL:-$MODEL_ID}"   # model served by vLLM for generation (e.g. AWQ variant)
GPUS="${GPUS:?set GPUS}"
PORT="${PORT:-8020}"
TAG="${TAG:?set TAG}"
LAYERS="${LAYERS:?set LAYERS}"
QUANTISE="${QUANTISE:-}"
VLLM_EXTRA="${VLLM_EXTRA:-}"
TIERS="${TIERS:-core transfer stego}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

NB="$THESIS/narcbench"
RAW="$THESIS/runs/$TAG/raw"
ACT="$THESIS/runs/$TAG/activations"
LOG="$THESIS/runs/$TAG/logs"
mkdir -p "$RAW" "$ACT" "$LOG"
export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_PORT="$PORT"
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
QARG=""; [ -n "$QUANTISE" ] && QARG="--quantise $QUANTISE"
cd "$NB"

{
echo "model=$MODEL_ID gen_model=$GEN_MODEL gpus=$GPUS port=$PORT tag=$TAG layers=$LAYERS quantise=$QUANTISE"
echo "vllm_extra=$VLLM_EXTRA tiers=$TIERS mem=$GPU_MEM_UTIL maxlen=$MAX_MODEL_LEN"
echo "narcbench=$(git rev-parse --short HEAD) start=$(date -Is)"
} | tee "$LOG/run_info.txt"

if [ -z "${SKIP_GEN:-}" ]; then
  python -m vllm.entrypoints.openai.api_server --model "$GEN_MODEL" --port "$PORT" \
    --tensor-parallel-size "$NGPU" --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" $VLLM_EXTRA > "$LOG/vllm.log" 2>&1 &
  VPID=$!
  trap 'kill $VPID 2>/dev/null || true' EXIT
  for i in $(seq 1 900); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready ${i}s"; break; }
    kill -0 $VPID 2>/dev/null || { echo "vLLM died"; tail -30 "$LOG/vllm.log"; exit 1; }
    sleep 1
  done
  for t in $TIERS; do
    echo "[gen $t] $(date -Is)"
    python -u "generation/$t.py" --model "$GEN_MODEL" --output-dir "$RAW/$t" > "$LOG/gen_$t.log" 2>&1
    echo "[gen $t] done $(date -Is) runs=$(ls "$RAW/$t" | wc -l)"
  done
  kill $VPID; wait $VPID 2>/dev/null || true; trap - EXIT; sleep 5
fi

if [ -z "${SKIP_EXTRACT:-}" ]; then
  for t in $TIERS; do
    echo "[extract $t] $(date -Is)"
    extra=""; [ "$t" = transfer ] && extra="--modes collusion,control"
    mkdir -p "$ACT/$t"
    python -u "extraction/extract_$t.py" --model "$MODEL_ID" $QARG --gen-only $extra \
      --layers "$LAYERS" --runs-dir "$RAW/$t" --output-dir "$ACT/$t" > "$LOG/extract_$t.log" 2>&1
    echo "[extract $t] done $(date -Is)"
  done
fi
echo "PIPELINE COMPLETE $(date -Is)"
