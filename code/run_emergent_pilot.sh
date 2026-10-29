#!/bin/bash
# E8c feasibility pilot: Shi et al. (arXiv 2609.24967) emergent-collusion environment with our own model.
# Needs BOTH GPUs: Qwen3-14B-AWQ with tensor parallelism 2, so the context can reach ~40k tokens
# (the environment's reduced-memory setting averages 8-16k tokens per call, max 32-56k on Gemini).
# Generation only; activations are extracted later if the pilot shows that the model colludes.
#
# Usage (gpuq, --gpus 2):  REPEATS=3 bash code/run_emergent_pilot.sh
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
: "${GPUS:?}"
PORT="${PORT:-8020}"; MODEL="${EM_MODEL:-Qwen/Qwen3-14B-AWQ}"; REPEATS="${REPEATS:-3}"
SCOPE="${MEM_SCOPE:-feedback-and-reflection}"; NAME="${EM_NAME:-emergent_pilot}"
OUT="$THESIS/runs/qwen3_14b/$NAME"; LOG="$THESIS/runs/qwen3_14b/logs"; mkdir -p "$OUT" "$LOG"
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(echo "$GPUS" | tr , '\n' | grep -c .)
echo "emergent pilot start $(date -Is) model=$MODEL gpus=$GPUS tp=$NGPU scope=$SCOPE repeats=$REPEATS"

python -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" --tensor-parallel-size "$NGPU" \
  --max-model-len 40960 --gpu-memory-utilization "${GPU_MEM_UTIL:-0.72}" --max-num-seqs 4 --max-num-batched-tokens 2048 --enforce-eager \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes --disable-log-requests \
  --chat-template "${EM_TEMPLATE:-$THESIS/code/qwen3_nothink.jinja}" \
  > "$LOG/${NAME}_vllm.log" 2>&1 &
VPID=$!; trap 'kill $VPID 2>/dev/null || true' EXIT
for i in $(seq 1 1200); do
  curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready ${i}s"; break; }
  kill -0 $VPID 2>/dev/null || { echo "vLLM died"; tail -20 "$LOG/${NAME}_vllm.log"; exit 1; }
  sleep 1
done
deactivate 2>/dev/null || true
cd "$THESIS/ext/agent-collusion"
export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY="local" AC_FORCE_AUTO_TOOL_CHOICE=1
"$THESIS/ext/venv_ac/bin/python" -m experiments \
  --alice-model "openai/$MODEL" --bob-model "openai/$MODEL" \
  --alice-max-output-tokens "${MAX_OUT:-1024}" --bob-max-output-tokens "${MAX_OUT:-1024}" \
  --alice-reasoning-effort default --bob-reasoning-effort default \
  --cross-episode-memory-scope "$SCOPE" \
  --task-sequence-record task/task_sequences_50x10 --repeats "$REPEATS" \
  --output-dir "$OUT" > "$LOG/${NAME}.log" 2>&1
echo "emergent pilot COMPLETE $(date -Is) runs=$(find "$OUT" -name run.json | wc -l)"
