"""Paper 1 A2: run ids to DROP so the Qwen3-14B data matches the Mistral-Small-24B design cell for cell
(collusion reps 0-3; control, briefed_decline, benign_coord, independent_bias reps 0-1; 50 scenarios each; no tacit /
cost_pressure). Controls come from the v1 runs (identical prompts and cell spec).
Usage: python matched_family_exclude.py runs/qwen3_14b > exclude.txt"""
import glob
import json
import sys

R = sys.argv[1]
keep_reps = {"collusion": range(4), "control": range(2), "briefed_decline": range(2), "benign_coord": range(2),
             "independent_bias": range(2)}
drop, seen = [], set()
for f in sorted(glob.glob(f"{R}/e2v2_*_c?/*/run_config.json") + glob.glob(f"{R}/e2sys_[ab]/*/run_config.json")
                + glob.glob(f"{R}/e2ben_[ab]/*/run_config.json")):
    c = json.load(open(f))
    v1 = "/e2v2_" not in f
    keep = (c["mode"] in keep_reps and c["replicate"] in keep_reps[c["mode"]]
            and (c["mode"] == "control") == v1 and (c["mode"], c["scenario_id"], c["replicate"]) not in seen)
    if keep:
        seen.add((c["mode"], c["scenario_id"], c["replicate"]))
    else:
        drop.append(c["run_id"])
print("\n".join(drop))
