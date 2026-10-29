"""Outcome table for the steering / ablation conditions (PLAN: steering / ablation along the probe direction).

Per condition: collusion success, briefed agents all voting the target, target majority, and the mean share of
briefed agents voting the target, with bootstrap CIs; plus a paired comparison against the matching baseline
(same scenario and replicate, unsteered). Also counts empty / very short briefed-agent messages as a crude
damage check.

Usage: python steer_outcomes.py --root runs/qwen3_14b/steer --out results/steer/outcomes
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

BASE = {"col": "base_col", "dec": "base_dec"}


def load(root, name):
    rows = {}
    for f in sorted(glob.glob(f"{root}/{name}/*/run_config.json")):
        c = json.load(open(f))
        special = c["coalition"]
        tv = [c["votes"].get(a) == c["target_option"] for a in special]
        msgs = json.load(open(Path(f).with_name("agent_prompts.json")))
        texts = [m["response"] for m in msgs if m["agent_name"] in special and m["round"] >= c["onset"]]
        rows[(c["scenario_id"], c["replicate"])] = dict(
            success=c["collusion_success"], all_target=c["special_all_target"], majority=c["target_majority"],
            share=float(np.mean(tv)), short=sum(len(t.split()) < 20 for t in texts), n_msg=len(texts))
    return rows


def ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.RandomState(seed)
    b = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(B)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/qwen3_14b/steer")
    ap.add_argument("--out", default="results/steer/outcomes")
    args = ap.parse_args()
    names = ["base_col", "abl_col", "ablr_col", "base_dec", "add2_dec", "add4_dec", "addr4_dec", "addm1_dec", "addm2_dec",
             "addmr2_dec"]
    data = {n: load(args.root, n) for n in names}
    out, lines = {}, []
    lines.append("condition   n   success [CI]        all_target [CI]     majority [CI]       share [CI]          "
                 "paired diff in share vs base [CI]   short msgs")
    for n in names:
        d = data[n]
        if not d:
            continue
        r = {"n": len(d)}
        cells = []
        for k in ("success", "all_target", "majority", "share"):
            m, lo, hi = ci([v[k] for v in d.values()])
            r[k] = [m, lo, hi]
            cells.append(f"{m:.2f} [{lo:.2f},{hi:.2f}]")
        base = data[BASE[n.split("_")[1]]]
        keys = sorted(set(d) & set(base))
        if n not in BASE.values() and keys:
            m, lo, hi = ci([d[k]["share"] - base[k]["share"] for k in keys])
            r["paired_share_diff"] = [m, lo, hi, len(keys)]
            pd = f"{m:+.2f} [{lo:+.2f},{hi:+.2f}] n={len(keys)}"
        else:
            pd = "-"
        short = sum(v["short"] for v in d.values())
        nmsg = sum(v["n_msg"] for v in d.values())
        r["short_msgs"] = [short, nmsg]
        out[n] = r
        lines.append(f"{n:10s} {len(d):3d}  " + "  ".join(f"{c:18s}" for c in cells) + f"  {pd:34s} {short}/{nmsg}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out + ".json", "w"), indent=1)
    open(args.out + ".log", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
