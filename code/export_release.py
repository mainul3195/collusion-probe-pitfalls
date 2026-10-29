"""Export the Paper 1 episodes and stance labels to Parquet for the Hugging Face release (RELEASE_PLAN.md §2, step R2).

One row per episode: the run configuration (roles, coalition, onset, votes, outcome, wording) as columns, and every
model call (system prompt, user prompt, response, phase, round, channel) as a JSON list. Stance labels: one row per
labelled run, the labeller's output kept as JSON. Only the folders used in the paper are exported; pilots that the
paper does not report, smoke tests and the abandoned first benchmark version are left out.

Usage (server, CPU only):
  python export_release.py --runs $THESIS/runs --out $THESIS/release/hf
"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# (model folder, subset name, folder prefixes); a folder is exported when its name starts with a prefix
SUBSETS = [
    ("qwen3_14b", "committee", ["e2v2_", "e2v2x_", "choice_c", "benchoice_c"]),
    ("qwen3_14b", "code_review", ["cr_", "crpilot_"]),
    ("qwen3_14b", "monitor_aware", ["e2v2aw_", "e2v2an_"]),
    ("qwen3_14b", "no_brief", ["implant_main_", "implant_pilot"]),
    ("qwen3_14b", "steering", ["steer"]),
    ("qwen3_32b", "committee", ["e2v2_"]),
    ("mistral24b", "committee", ["e2v2_"]),
]
SKIP_SUFFIX = ("_act", "_strip", "_final")
CONFIG_COLS = ["mode", "n", "k", "onset", "num_rounds", "coalition", "roles", "target_option", "fair_option", "votes",
               "collusion_success", "special_all_target", "target_majority", "target_vote_count", "template",
               "heldout_template", "label_variant", "scenario_id", "scenario_domain", "scen_idx", "replicate"]


def run_dirs(folder):
    """Episode folders (containing run_config.json) at depth 1 or 2, skipping activation folders."""
    for d in sorted(folder.iterdir()):
        if not d.is_dir() or d.name.endswith(SKIP_SUFFIX):
            continue
        if (d / "run_config.json").exists():
            yield d
        else:
            yield from run_dirs(d)


def export_episodes(runs, out, original=False):
    """original=False: the episodes used in the paper. original=True: the episodes set aside by the option-label
    correction (PREREG amendment 7; folders <name>_labelbug), exported under original_labelbug/ for comparison."""
    for model, subset, prefixes in SUBSETS:
        rows = []
        for folder in sorted((runs / model).iterdir()):
            if not folder.is_dir() or folder.name.endswith(SKIP_SUFFIX):
                continue
            if folder.name.endswith("_labelbug") != original or folder.name.endswith("_act_labelbug"):
                continue
            if not any(folder.name.startswith(p) for p in prefixes) or (subset == "committee" and "pilot" in folder.name):
                continue
            for d in run_dirs(folder):
                cfg = json.loads((d / "run_config.json").read_text())
                calls = json.loads((d / "agent_prompts.json").read_text())
                row = {"model": model, "subset": subset, "source": str(d.relative_to(runs / model).parent),
                       "run_id": d.name}
                for c in CONFIG_COLS:
                    v = cfg.get(c)
                    row[c] = json.dumps(v) if isinstance(v, (dict, list)) else (None if v is None else str(v))
                row["config"] = json.dumps(cfg).replace(str(runs.parent), "$THESIS_ROOT")  # steering dir_file paths
                row["calls"] = json.dumps(calls)
                rows.append(row)
        if not rows:
            continue
        path = out / ("original_labelbug" if original else "episodes") / f"{model}_{subset}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        print(f"{path.name}: {len(rows)} episodes, {path.stat().st_size / 1e6:.0f} MB", flush=True)


def export_stance(runs, out, original=False):
    src = runs / "qwen3_14b" / ("stance_mistral24b_labelbug" if original else "stance_mistral24b")
    rows = [{"run_id": f.stem, "labels": f.read_text()} for f in sorted(src.glob("*.json"))]
    path = out / ("original_labelbug" if original else "stance_labels") / "qwen3_14b_mistral24b_labeller.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    print(f"{path.name}: {len(rows)} runs, {path.stat().st_size / 1e6:.0f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for original in (False, True):
        export_episodes(runs, out, original)
        export_stance(runs, out, original)


if __name__ == "__main__":
    main()
