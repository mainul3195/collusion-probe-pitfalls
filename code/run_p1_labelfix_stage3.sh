#!/bin/bash
# PREREG amendment 7, stage 3: rerun every CPU analysis behind a reported number on the corrected episodes.
# Old results are kept: results/ is copied once to results_labelbug/ before anything is overwritten.
# Same commands as the original runs (run_*.sh wrappers, or the logged command where no wrapper exists).
# Not rerun here: label_mismatch / robustness / instruction/clean (they describe the error itself), the human audit
# (needs a decision: audited episodes were partly regenerated), encoder scores (GPU), steering (stage 4), code-review
# within-domain results (episodes unaffected). Decision curve: phase 2, sizes read from the new decision logs.
# CPU only. Usage (server): nohup bash code/run_p1_labelfix_stage3.sh > results/labelfix_stage3.out 2>&1 &
set -uo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
R="$THESIS/runs/qwen3_14b"; RES="$THESIS/results"; P="$RES/paper1"
V1="$R/e2sys_a_act $R/e2sys_b_act $R/e2ben_a_act $R/e2ben_b_act"
cd "$THESIS/code"

[ -d "$THESIS/results_labelbug" ] || cp -a "$RES" "$THESIS/results_labelbug"
echo "snapshot: $THESIS/results_labelbug ($(du -sh "$THESIS/results_labelbug" | cut -f1))"

MAXJ=${MAXJ:-4}
job() {  # job <name> <command...>; runs in the background, at most MAXJ at a time
  local name="$1"; shift
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
  ( echo "start $name $(date -Is)"; "$@" && echo "done $name $(date -Is)" || echo "!! FAILED $name rc=$? $(date -Is)" ) &
}
ana() {  # ana <out json> <e2v2_analysis args...>
  local out="$1"; shift
  python -u e2v2_analysis.py "$@" --workers 5 --out "$out" > "${out%.json}.log" 2>&1
}
E2="--core $R/activations/core --v2 $R/e2v2_*_c?_act --v1-controls $V1"
O="$RES/e2v2b"

# --- Qwen3-14B main (e2v2b)
job e2v2b.act       ana $O/act.json --signal act $E2
job e2v2b.ibneg     ana $O/act_ibneg.json --signal act --ib neg $E2
job e2v2b.recipe    ana $O/act_paper_recipe.json --signal act --neg paper $E2
job e2v2b.core      ana $O/core.json --signal core $E2
job e2v2b.text      ana $O/text.json --signal text $E2
job e2v2b.vote      ana $O/vote.json --signal vote $E2
job e2v2b.judgeq    ana $O/judge_logprob.json --signal judge --judge-dir $R/judge_v2_logprob $E2
job e2v2b.judgem    ana $O/judge_mistral.json --signal judge --judge-dir $R/judge_v2_mistral $E2
job e2v2b.errfree   bash -c "python list_error_runs.py $R/e2v2_*_c? $R/e2sys_[ab] $R/e2ben_[ab] > $O/error_runs.txt && \
  python -u e2v2_analysis.py --signal act --exclude-runs $O/error_runs.txt $E2 --workers 5 --out $O/act_errfree.json > $O/act_errfree.log 2>&1"
wait

# --- wrappers (each runs its own analyses in sequence)
rm -f "$RES"/e7/{calib_*,layer_*,layers_all,feat_mean,templates_?,pos_success}.{json,log}  # run_e7_ablations skips existing files
job e7          bash run_e7_ablations.sh
job e8b         bash run_e8b_analysis.sh
job matched14b  bash run_matched_14b.sh
job mistral     env TAG=mistral24b NAME=mistral bash run_p1_family_analysis.sh
job matchedfam  bash run_p1_matched_family.sh
job aware       bash run_aware_analysis.sh
job notice      bash run_notice_analysis.sh
job negx        bash run_p1_negx_analysis.sh
job seeds       bash run_p1_seeds.sh
job fullchoice  bash run_p1_fullchoice.sh
job h5.q14      python -u instruction_tests.py h5 --runs $R --out $P/instruction
job h5.q32      python -u instruction_tests.py h5 --runs $THESIS/runs/qwen3_32b --name _q32 --layers 26,29,32 --out $P/instruction
job h5.mis      python -u instruction_tests.py h5 --runs $THESIS/runs/mistral24b --name _mistral --out $P/instruction
job context     python -u context_baseline.py --runs $R/e2v2_*_c? --controls $R/e2sys_a $R/e2sys_b $R/e2ben_a $R/e2ben_b \
                  --name committee --out $P/context
job acmas30     python -u acmas_check.py --runs $R --layer 30 --out $P/acmas
job acmas40     python -u acmas_check.py --runs $R --layer 40 --suffix _final --out $P/acmas
job metabase    python -u meta_baseline.py --runs $R --out $P/acmas
job nobrief     python -u implant_analysis.py --runs $R --stance $R/stance_mistral24b --out $P/nobrief
job transfer1   python -u transfer_task.py --source $R/e2v2_*_c?_act --source-v1 $V1 --target $R/cr_*_c?_act \
                  --name committee_to_codereview --out $P/transfer
job transfer2   python -u transfer_task.py --source $R/cr_*_c?_act --target $R/e2v2_*_c?_act --target-v1 $V1 \
                  --name codereview_to_committee --out $P/transfer
job reader      bash run_e8c_reader.sh
wait

# --- need the new e2v2b/act.json records
job ceiling     python -u public_ceiling.py --stance $R/stance_mistral24b --runs $R/e2v2_*_c? $R/e2sys_[ab] $R/e2ben_[ab] \
                  --probe $O/act.json --name committee --out $P/ceiling
job h4          python -u instruction_tests.py h4 --audit $P/audit/audit.json --records $O/act.json $P/codereview/act.json \
                  --out $P/instruction
wait
echo "stage 3 phase 1 COMPLETE $(date -Is)"
