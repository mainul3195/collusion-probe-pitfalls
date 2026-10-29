#!/bin/bash
# LLM-judge text baseline: start vLLM, judge every run of the shard, stop. Resumable (finished runs are skipped).
#
# Usage (gpuq sets GPUS and PORT):
#   SHARD=0/2 bash code/run_judge.sh
# JUDGE_MODEL / JUDGE_NAME: other judge (e.g. second model family). Two GPUs in GPUS -> tensor parallel 2.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
: "${GPUS:?}" "${SHARD:?}"
PORT="${PORT:-8020}"; TAG="${TAG:-qwen3_14b}"; JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-14B-AWQ}"
JUDGE_NAME="${JUDGE_NAME:-judge_v2}"; PARALLEL="${PARALLEL:-6}"; LIMIT="${LIMIT:-}"; STYLE="${STYLE:-logprob}"
R="$THESIS/runs/$TAG"; OUT="$R/$JUDGE_NAME"; LOG="$R/logs"; S="${SHARD%/*}"; mkdir -p "$OUT" "$LOG"
export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l); TP=""; [ "$NGPU" -gt 1 ] && TP="--tensor-parallel-size $NGPU"
echo "judge start $(date -Is) model=$JUDGE_MODEL shard=$SHARD gpus=$GPUS out=$OUT"

python -m vllm.entrypoints.openai.api_server --model "$JUDGE_MODEL" --port "$PORT" \
  --max-model-len 11264 --gpu-memory-utilization "${GPU_MEM_UTIL:-0.78}" --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --enforce-eager --enable-prefix-caching $TP \
  --disable-log-requests > "$LOG/${JUDGE_NAME}_vllm_$S.log" 2>&1 &
VPID=$!; trap 'kill $VPID 2>/dev/null || true' EXIT
for i in $(seq 1 900); do
  curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM ready ${i}s"; break; }
  kill -0 $VPID 2>/dev/null || { echo "vLLM died"; tail -20 "$LOG/${JUDGE_NAME}_vllm_$S.log"; exit 1; }
  sleep 1
done
cd "$THESIS/code"
LARG=""; [ -n "$LIMIT" ] && LARG="--limit $LIMIT"
pass() {  # v2 families (all modes), then the v1 control runs that the v2 analysis reuses
  python -u judge_e2.py --model "$JUDGE_MODEL" --port "$PORT" --shard "$SHARD" --parallel "$PARALLEL" --style "$STYLE" $LARG \
    --out "$OUT" --runs ${V2_DIRS:-"$R"/e2v2_*_c?} &&
  python -u judge_e2.py --model "$JUDGE_MODEL" --port "$PORT" --shard "$SHARD" --parallel "$PARALLEL" --style "$STYLE" $LARG \
    --out "$OUT" --modes control --runs "$R"/e2sys_a "$R"/e2sys_b "$R"/e2ben_a "$R"/e2ben_b
}
pass >> "$LOG/${JUDGE_NAME}_$S.log" 2>&1 || pass >> "$LOG/${JUDGE_NAME}_$S.log" 2>&1
[ -z "$LIMIT" ] && touch "$OUT/DONE_$S"
echo "judge COMPLETE $(date -Is) files=$(ls "$OUT" | grep -c '\.json$')"
