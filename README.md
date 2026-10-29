# What Do Collusion Probes Read? Three Pitfalls in Evaluating Multi-Agent Collusion Monitors

Code, pre-registration and result summaries for the paper by Mainul Hasan Mohsin and Mohammad Shahidur Rahman
(Shahjalal University of Science and Technology).

- **Paper:** arXiv link to follow
- **Data** (all episodes, stance labels, full result files with the cross-fitted per-run monitor scores): https://huggingface.co/datasets/mainul3195/collusion-probe-pitfalls
- **Archived release with DOI:** https://doi.org/10.5281/zenodo.23636508

## What is in this repository

| Folder | Contents |
|---|---|
| `code/` | Episode generator, activation extraction, probes and aggregators, baselines (text classifiers, LLM judges, AcMAS-style detector, metadata baseline, stance oracle), the instruction tests, the no-brief (implanted goal) test, steering, transfer and robustness analyses |
| `patches/narcbench.diff` | Local changes to NARCBench (MIT) needed to run its code on 16 GB GPUs |
| `results/` | Result summaries and logs under 1 MB. The larger per-run record files are on Hugging Face |
| `paper/` | `make_numbers.py` turns `results/` into `numbers.tex`, the file every number in the paper is read from; `check_numbers.py` checks a manuscript for numbers typed by hand |
| `prereg/` | Every committed version of the pre-registration, its commit table and an OpenTimestamps proof (see `prereg/COMMITS.md`) |

## Main scripts

| Step | Script |
|---|---|
| Committee episodes (all families, five wordings) | `code/gen_e2v2.py` |
| Code-review scenarios and episodes | `code/scenarios_codereview.py`, `code/gen_e2v2.py` |
| Activation extraction | `code/extract_e2.py` |
| Probes, aggregators (PEAK, Deception Split, CA-T, CA-GLR), live alarm, main tables | `code/e2v2_analysis.py`, `code/detectors.py`, `code/common.py` |
| Pitfall 1 (vote rule) | `code/vote_baseline.py` |
| Pitfall 2 (brief removal, same-prompt split, free choice, decision probe) | `code/instruction_tests.py` |
| AcMAS-style divergence detector | `code/acmas_check.py` |
| Metadata baseline | `code/meta_baseline.py` |
| Prompt reader | `code/context_baseline.py` |
| Pitfall 3 (stance labels, stance oracle) | `code/stance_label.py`, `code/public_ceiling.py` |
| Text baselines | `code/judge_e2.py`, `code/text_encoder_scores.py` |
| No-brief test (implanted goal) | `code/build_implant_data.py`, `code/train_implant.py`, `code/implant_gate.py`, `code/implant_analysis.py`, `code/implant_text.py` |
| Steering and ablation | `code/steer_direction.py`, `code/steer_engine.py`, `code/steer_outcomes.py`, `code/steer_scores.py` |
| Transfer between tasks | `code/transfer_task.py` |
| Option-label correction (PREREG amendment 7) | `code/label_mismatch.py`, `code/archive_labelbug.py`, `code/archive_outputs_labelbug.py`, `code/run_p1_labelfix_stage3.sh`; the fix is `relabel_options` in `code/gen_e2v2.py` |
| Export of the Hugging Face dataset | `code/export_release.py` |

The `code/run_*.sh` files are the exact command sequences we ran; they expect the environment described below.

## Setup

Hardware used: two 16 GB consumer GPUs, about 600 GPU-hours in total. All models are open-weight and run locally
(Qwen3-14B, Qwen3-32B and Mistral-Small-24B in 4-bit AWQ through vLLM).

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps peft==0.17.1

# NARCBench (MIT): scenarios and reference probe code
git clone https://github.com/aaronrose227/narcbench.git
git -C narcbench checkout 85fea3be2a9d8b9f16e33a4bd65bd947c31fa07d
git -C narcbench apply ../patches/narcbench.diff   # or: python code/patch_narcbench.py narcbench

cp env.sh.example env.sh   # edit THESIS_ROOT, then: source env.sh
```

## Reproducing the numbers without a GPU

Download the result files from Hugging Face into `results/`, then

```bash
python paper/make_numbers.py      # writes paper/numbers.tex
```

Every result number in the paper is a macro in `numbers.tex`; with the Hugging Face results the regenerated file is
identical to the shipped one.

## Data and release notes

- **Option-label correction.** The generator varies how the two options are labelled (A/B or 1/2), but in a first
  version of the data the scenario text inside the colluders' brief still named the options by their original letters,
  so 572 of the 1,000 main collusion briefs pointed to the fair option or to a label not on the menu. The fix and the
  rerun were pre-registered (amendment 7, `prereg/`) before any corrected episode was generated; every affected episode
  was regenerated with its original seed and settings and every analysis rerun. All released episodes and results are
  the corrected ones; the original episodes are available from the authors on request. No conclusion changed.

- **Withheld:** the adapter that implants the collusive goal (no-brief test). Its training script and data recipe are
  here, so the result can be reproduced; the weights are available to editors and reviewers on request.
- **Human stance check:** the three annotators' labels are in `results/paper1/audit/stance_A1.json` to `stance_A3.json`
  (anonymised; released with their consent; annotators: the first author, Iftekharul Alam Tarit and
  Mehedi Hasan Emon, whom we thank), with the item key `stance_sheet_key.json` and the agreement scores
  `review_score.{json,log}`. In `paper/numbers.tex` their keys are `hone.a1`, `hone.a2`, `hone.a3`.
- `code/scenarios_codereview.py` contains deliberately insecure code (for example a fake hard-coded API key); these are
  the flaws the review board is meant to find. The key is not real.
- NARCBench scenarios and code are © their authors, MIT licence. We do not redistribute NARCBench's released
  activations.

## Citation

See `CITATION.cff`.

## Licence

Code: MIT (see `LICENSE`). Data on Hugging Face: CC BY 4.0.
