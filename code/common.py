"""Shared helpers for thesis analyses on NARCBench-format activations.

Activation dirs follow the NARCBench layout:
    <act_root>/<tier>/{metadata_gen.json, activations_gen.npz}
Released data: narcbench/data_hf/activations/<model_short>/
Own runs:      runs/<tag>/activations/
"""
import os
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

THESIS = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(THESIS / "narcbench"))
sys.path.insert(0, str(THESIS / "narcbench" / "probes"))
import reproduce as R  # noqa: E402  (NARCBench reference implementation)

RELEASED = THESIS / "narcbench" / "data_hf" / "activations"
RELEASED_LAYERS = {
    "qwen3_32b": "26-30",
    "llama_3_1_70b": "32-37",
    "deepseek_r1_32b": "26-30",
    "gpt_oss_20b": "10-14",
}
FPR_POINTS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def parse_layers(spec):
    """'26-30' -> [26..30]; '14,16,18' -> [14, 16, 18]."""
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def load_tier(act_root, tier):
    return R.load_data(Path(act_root) / tier)


def domain_of(run_id):
    return R._domain_from_scenario_id(R._parse_run_id(run_id)[1])


# ── metrics ──────────────────────────────────────────────────────────────

def tpr_at_fpr(y, s, alpha):
    """Recall at the strictest threshold whose empirical FPR <= alpha.

    alpha=0 means recall above the maximum negative score (zero false alarms).
    """
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    neg = np.sort(s[y == 0])[::-1]
    k = int(np.floor(alpha * len(neg)))
    tau = neg[k] if k < len(neg) else -np.inf
    return float(np.mean(s[y == 1] > tau))


def metrics(y, s, n_boot=2000, seed=42):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    out = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
           "auroc": float(roc_auc_score(y, s)),
           "auprc": float(average_precision_score(y, s))}
    for a in FPR_POINTS:
        out[f"tpr@{a:g}"] = tpr_at_fpr(y, s, a)
    # stratified bootstrap CIs
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = defaultdict(list)
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yb, sb = y[idx], s[idx]
        boots["auroc"].append(roc_auc_score(yb, sb))
        for a in FPR_POINTS:
            boots[f"tpr@{a:g}"].append(tpr_at_fpr(yb, sb, a))
    for k, v in boots.items():
        out[f"{k}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    # low-FPR part of the ROC curve (FPR <= 10%) for log-scale plots
    fpr, tpr, _ = roc_curve(y, s)
    keep = fpr <= 0.10
    out["roc_low_fpr"] = {"fpr": fpr[keep].round(5).tolist(), "tpr": tpr[keep].round(4).tolist()}
    return out


# ── per-row probe projections ────────────────────────────────────────────

def row_projections(core_meta, core_npz, tgt_meta, tgt_npz, layers, oof=False):
    """Project every target row onto the deception direction, averaged over layers.

    oof=True (target is Core): each row is scored by a probe trained with its
    own domain held out (leave-domain-out), so no Core score is in-sample.
    """
    out = np.zeros((len(tgt_meta), len(layers)))
    doms = np.array([domain_of(m["run_id"]) for m in tgt_meta])
    for li, L in enumerate(layers):
        A = R.get_layer(tgt_npz, len(tgt_meta), L)
        if oof:
            for d in np.unique(doms):
                scaler, w = R.train_deception_probe(core_meta, core_npz, L, exclude_domain=d)
                idx = np.where(doms == d)[0]
                out[idx, li] = scaler.transform(A[idx]) @ w
        else:
            scaler, w = R.train_deception_probe(core_meta, core_npz, L)
            out[:, li] = scaler.transform(A) @ w
    return out.mean(axis=1)


def runs_table(meta, proj):
    """run_id -> {mode, domain, agents: {name: {role, rounds: {(phase, round): score}}}} (public channel)."""
    runs = {}
    for m, p in zip(meta, proj):
        if m["channel"] != "public":
            continue
        r = runs.setdefault(m["run_id"], {"mode": m["mode"], "domain": domain_of(m["run_id"]),
                                          "agents": {}})
        a = r["agents"].setdefault(m["agent_name"], {"role": m["role"], "rounds": {}})
        a["rounds"][(m["phase"], m["round"])] = float(p)
    return runs


def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
