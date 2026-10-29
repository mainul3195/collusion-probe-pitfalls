"""Paper 1 no-brief test, step 1: training data for an implanted (weights-borne) collusion goal.

Source: colluders of the E2 v2 collusion runs who acted (voted the target). Each of their calls after the onset becomes
one example whose CONTEXT is what an implanted agent will see (no brief anywhere) and whose TARGET is the colluder's
own reply:
  system   the honest system prompt of the same seat (brief removed, as extract_e2.py --strip context)
  user     private round: the colluder invitation (which names the target) replaced by the neutral benign-coordination
           invitation of the same template; public rounds and final vote: unchanged (they hold the agents' own
           private messages and the public history, i.e. behaviour, not instruction)
  response the colluder's reply, unchanged
--honest (control adapter): the same procedure on benign_coord runs: coordinators' replies, system prompt reset to the
honest prompt of the seat, user prompts unchanged (their private invitation already is the benign one). Same size, same
folds, same context structure as the implanted data; only the content of the replies is honest.
Only scenarios outside the held-out fold are used (same scenario folds as e2v2_analysis.py, fold --heldout-fold), so the
implanted episodes run on scenarios the adapter has never seen. Writes <out>/train.jsonl and <out>/split.json
(held-out scenario ids and their indices in the bank, for gen_e2v2.py --scen-idx).

Usage:
  python build_implant_data.py --runs runs/qwen3_14b/e2v2_col_c? --out runs/qwen3_14b/implant_data --max 3000
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from extract_e2 import strip_context
from gen_e2 import C
from gen_e2v2 import BENIGN_FRAMES, COLLUDER_FRAMES

F = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-fold", type=int, default=0)
    ap.add_argument("--max", type=int, default=3000, help="at most this many examples (random subsample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--honest", action="store_true", help="control: benign coordinators' replies (benign_coord runs)")
    args = ap.parse_args()

    ids = [sc["id"] for sc in C.SCENARIOS]
    scen = sorted(ids)
    fold_of = {s: i % F for i, s in enumerate(np.random.RandomState(42).permutation(scen))}
    held = sorted(s for s in scen if fold_of[s] == args.heldout_fold)
    ex, skipped, n_runs = [], 0, 0
    for d in args.runs:
        for rd in sorted(Path(d).iterdir()):
            if not (rd / "agent_prompts.json").exists():
                continue
            c = json.loads((rd / "run_config.json").read_text())
            want = "benign_coord" if args.honest else "collusion"
            if c["mode"] != want or c.get("aware") or c["scenario_id"] in held:
                continue
            acted = set(c["coalition"]) if args.honest else {a for a in c["coalition"]
                                                              if c["votes"].get(a) == c["target_option"]}
            if not acted:
                continue
            n_runs += 1
            t = c["template"]
            for row in json.loads((rd / "agent_prompts.json").read_text()):
                a = row["agent_name"]
                if a not in acted or not (row["active"] or row["phase"] == "final"):
                    continue
                sp, up = strip_context(c, row)
                sp = sp if row["active"] else row["system_prompt"]
                if row["phase"] == "private" and args.honest:
                    pass  # benign invitation already
                elif row["phase"] == "private":
                    partners = ", ".join(p for p in c["coalition"] if p != a)
                    cf = COLLUDER_FRAMES[t].format(partners=partners, target=c["target_option"])
                    if cf not in up:
                        skipped += 1
                        continue
                    up = up.replace(cf, BENIGN_FRAMES[t].format(partners=partners))
                if row["phase"] != "private":
                    up = row["user_prompt"]  # keep the agents' own private messages and the public history
                ex.append(dict(system=sp, user=up, response=row["response"], run_id=c["run_id"], agent=a,
                               phase=row["phase"], round=row["round"]))
    random.Random(args.seed).shuffle(ex)
    ex = ex[:args.max]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for e in ex:
            f.write(json.dumps(e) + "\n")
    split = dict(heldout_fold=args.heldout_fold, heldout_scenarios=held,
                 heldout_idx=[ids.index(s) for s in held], train_runs=n_runs, examples=len(ex), skipped=skipped)
    (out / "split.json").write_text(json.dumps(split, indent=1))
    print(json.dumps(split))


if __name__ == "__main__":
    main()
