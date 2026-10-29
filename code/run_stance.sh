#!/bin/bash
# Paper 1 H1: stance labelling of every pre-vote public and private message (stance_label.py). Starts vLLM with a
# labeller from another family, labels the committee data (v2 + v1 controls) and the code-review data, stops.
# Resumable (finished runs are skipped). LIMIT=n labels only the first n runs of each set (pilot).
#
# Usage (gpuq sets GPUS and PORT):  bash code/run_stance.sh
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
: "${GPUS:?}"
PORT="${PORT:-8030}"; TAG="${TAG:-qwen3_14b}"; LAB_MODEL="${LAB_MODEL:-stelterlab/Mistral-Small-24B-Instruct-2501-AWQ}"
LAB_NAME="${LAB_NAME:-stance_mistral24b}"; PARALLEL="${PARALLEL:-16}"; LIMIT="${LIMIT:-}"; SHARD="${SHARD:-0/1}"
R="$THESIS/runs/$TAG"; OUT="$R/$LAB_NAME"; LOG="$R/logs"; S="${SHARD%/*}"; mkdir -p "$OUT" "$LOG"
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l); TP=""; [ "$NGPU" -gt 1 ] && TP="--tensor-parallel-size $NGPU"
echo "stance start $(date -Is) model=$LAB_MODEL gpus=$GPUS out=$OUT limit=${LIMIT:-all}"

python -m vllm.entrypoints.openai.api_server --model "$LAB_MODEL" --port "$PORT" \
  --max-model-len 8192 --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" --max-num-seqs "${MAX_SEQS:-48}" \
  --enforce-eager --enable-prefix-caching $TP \
  --disable-log-requests > "$LOG/${LAB_NAME}_vllm_$S.log" 2>&1 &
VPID=$!; trap 'kill $VPID 2>/dev/null || true' EXIT
for i in $(seq 1 900); do
  curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready ${i}s"; break; }
  kill -0 $VPID 2>/dev/null || { echo "vLLM died"; tail -20 "$LOG/${LAB_NAME}_vllm_$S.log"; exit 1; }
  sleep 1
done
cd "$THESIS/code"
LARG=""; [ -n "$LIMIT" ] && LARG="--limit $LIMIT"
A=(--model "$LAB_MODEL" --port "$PORT" --shard "$SHARD" --parallel "$PARALLEL" --out "$OUT" $LARG)
pass() {  # committee v2 (all modes), committee v1 controls, code review (all modes); STANCE_RUNS="d1 d2": only these
  if [ -n "${STANCE_RUNS:-}" ]; then python -u stance_label.py "${A[@]}" --runs $(printf "$R/%s " $STANCE_RUNS); return; fi
  python -u stance_label.py "${A[@]}" --runs "$R"/e2v2_*_c? &&
  python -u stance_label.py "${A[@]}" --modes control --runs "$R"/e2sys_a "$R"/e2sys_b "$R"/e2ben_a "$R"/e2ben_b &&
  python -u stance_label.py "${A[@]}" --runs "$R"/cr_*_c?
}
pass >> "$LOG/${LAB_NAME}_$S.log" 2>&1 || pass >> "$LOG/${LAB_NAME}_$S.log" 2>&1
[ -z "$LIMIT" ] && touch "$OUT/DONE_${STANCE_TAG:-$S}"
echo "stance COMPLETE $(date -Is) files=$(ls "$OUT" | grep -c '\.json$')"
