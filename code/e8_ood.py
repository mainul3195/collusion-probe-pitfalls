"""E8a: out-of-distribution transfer of the E2 v2 probe to NARCBench (own Qwen3-14B data).

The probe direction is trained on ALL E2 v2 rows (same labels as e2v2_analysis.py) and then
applied, with no further fitting of the direction, to NARCBench Core / Transfer / Stego. The
evaluation is the one used for E0c (e0_e1_lowfpr.py): the paper's five methods, paper-style
AUROC, and pooled layer-ensembled low-FPR metrics. The two learned aggregators (asymmetry,
sorted concat) are fitted on Core scores, as in the paper; for Core they are cross-fitted by domain.
Reference rows use the NARCBench Core-trained probe (leave-domain-out on Core) = E0c.

Usage:
  python e8_ood.py --act-root runs/qwen3_14b/activations --v2 runs/qwen3_14b/e2v2_*_c?_act \
      --v1-controls runs/qwen3_14b/e2{sys,ben}_{a,b}_act --layers 16,18,20 [--ib neg] --out results/e8/ood.json
"""
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

import e2_analysis as A
from common import R, load_tier, metrics, parse_layers, save_json
from e0_e1_lowfpr import METHODS, method_outputs, zscore
from e2v2_analysis import labels


def evaluate(tiers, layers, probe_for):
    """probe_for(L, exclude_domain) -> (scaler, w). Returns paper-style AUROC and pooled metrics."""
    core_meta, core_npz = tiers["core"]
    paper = defaultdict(lambda: defaultdict(list))
    pooled = {t: defaultdict(list) for t in tiers}
    lab = {}
    for L in layers:
        groups = R.build_groups(core_meta, core_npz, L, phase="discussion", avg_rounds=True)
        rids = sorted(groups)
        doms = sorted({groups[r]["domain"] for r in rids})
        oof = {m: np.full(len(rids), np.nan) for m in METHODS}
        fold_auc = defaultdict(list)
        for d in doms:
            tr = [r for r in rids if groups[r]["domain"] != d]
            te = [r for r in rids if groups[r]["domain"] == d]
            ty, ey = R._run_labels(groups, tr), R._run_labels(groups, te)
            sc, w = probe_for(L, d)
            out = method_outputs(R.compute_scores(groups, tr, sc, w), ty, R.compute_scores(groups, te, sc, w))
            idx = [rids.index(r) for r in te]
            for m in METHODS:
                oof[m][idx] = out[m]
                if len(np.unique(ey)) == 2:
                    fold_auc[m].append(roc_auc_score(ey, out[m]))
        for m in METHODS:
            paper["core_heldout"][m].append(float(np.mean(fold_auc[m])))
            pooled["core"][m].append(zscore(oof[m]))
        lab["core"] = R._run_labels(groups, rids)
        sc, w = probe_for(L, None)
        core_scores, core_y = R.compute_scores(groups, rids, sc, w), R._run_labels(groups, rids)
        for tier in ("transfer", "stego"):
            meta, npz = tiers[tier]
            g = R.build_groups(meta, npz, L, phase="final", avg_rounds=False)
            gr = sorted(g)
            y = R._run_labels(g, gr)
            out = method_outputs(core_scores, core_y, R.compute_scores(g, gr, sc, w))
            fam = np.array([g[r]["scenario"] for r in gr])
            lab[tier] = y
            for m in METHODS:
                pooled[tier][m].append(zscore(out[m]))
                if tier == "transfer":
                    per = {f: float(roc_auc_score(y[fam == f], out[m][fam == f])) for f in np.unique(fam)
                           if len(np.unique(y[fam == f])) == 2}
                    paper["transfer_mean_family"][m].append(float(np.mean(list(per.values()))))
                    for f, v in per.items():
                        paper[f"transfer:{f}"][m].append(v)
                else:
                    paper["stego"][m].append(float(roc_auc_score(y, out[m])))
    return {"paper_metric": {g: {m: float(np.mean(v)) for m, v in d.items()} for g, d in paper.items()},
            "pooled": {t: {m: metrics(lab[t], np.mean(v, axis=0), n_boot=500) for m, v in d.items()}
                       for t, d in pooled.items()}}


def show(name, res):
    print(f"\n== probe: {name}")
    for g, d in res["paper_metric"].items():
        print(f"  {g:34s}", "  ".join(f"{m[:10]}={d[m]:.3f}" for m in METHODS))
    print("  pooled, layer-ensembled: AUROC | TPR@1% | TPR@5% (n_pos/n_neg)")
    for tier, d in res["pooled"].items():
        for m in METHODS:
            x = d[m]
            print(f"    {tier:9s} {m:22s} {x['auroc']:.3f} | {x['tpr@0.01']:.2f} | {x['tpr@0.05']:.2f}  ({x['n_pos']}/{x['n_neg']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-root", required=True)
    ap.add_argument("--v2", nargs="+", required=True)
    ap.add_argument("--v1-controls", nargs="*", default=[])
    ap.add_argument("--layers", default="16,18,20")
    ap.add_argument("--ib", choices=["exclude", "neg"], default="exclude")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    layers = parse_layers(args.layers)
    tiers = {t: load_tier(args.act_root, t) for t in ("core", "transfer", "stego")}

    meta, acts = A.load_e2(args.v2, layers)
    if args.v1_controls:
        m1, a1 = A.load_e2(args.v1_controls, layers)
        keep = np.array([m["mode"] == "control" for m in m1])
        meta += [m for m, k in zip(m1, keep) if k]
        acts = {L: np.concatenate([acts[L], a1[L][keep]]) for L in acts}
    y = labels(meta, args.ib)
    idx = np.where(y >= 0)[0]
    pos = idx[y[idx] == 1]
    pos = np.random.RandomState(0).choice(pos, min(len(pos), 6000), replace=False)
    v2_probe = A.fit_e2_probe(acts, layers, y, np.r_[pos, idx[y[idx] == 0]])
    print(f"v2 probe trained on {len(pos)} positive rows (+ 2x negatives), layers {layers}", flush=True)
    del acts
    core_meta, core_npz = tiers["core"]

    res = {"layers": layers, "ib": args.ib,
           "core_probe": evaluate(tiers, layers, lambda L, d: R.train_deception_probe(core_meta, core_npz, L, exclude_domain=d)),
           "v2_probe": evaluate(tiers, layers, lambda L, d: v2_probe[L])}
    show("NARCBench Core-trained (reference, = E0c)", res["core_probe"])
    show(f"E2 v2-trained, zero-shot on NARCBench (ib={args.ib})", res["v2_probe"])
    save_json(res, args.out)


if __name__ == "__main__":
    main()
