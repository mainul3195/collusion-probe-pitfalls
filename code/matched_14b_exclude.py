"""Run ids to DROP so the 14B data matches the 32B E8b design (collusion reps 0-3, briefed_decline and
benign_coord reps 0-1, 100 controls, no tacit / cost_pressure / independent_bias) -> matched-size comparison.
Usage: python matched_14b_exclude.py runs/qwen3_14b > matched_exclude.txt"""
import glob
import json
import random
import sys

R = sys.argv[1]
keep_reps = {"collusion": range(4), "briefed_decline": range(2), "benign_coord": range(2)}
drop, controls = [], []
for f in sorted(glob.glob(f"{R}/e2v2_*_c?/*/run_config.json") + glob.glob(f"{R}/e2sys_[ab]/*/run_config.json")
                + glob.glob(f"{R}/e2ben_[ab]/*/run_config.json")):
    c = json.load(open(f))
    if c["mode"] == "control":
        controls.append(c["run_id"])
    elif c["mode"] not in keep_reps or c["replicate"] not in keep_reps[c["mode"]] or "/e2v2_" not in f:
        drop.append(c["run_id"])
random.Random(0).shuffle(controls)
drop += controls[100:]
print("\n".join(drop))
