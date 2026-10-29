"""E0a + E1: reproduce NARCBench static baselines and re-evaluate them at low FPR.

For each model and layer in its probe band:
  * Core: leave-domain-out CV. We record (a) the paper's metric — AUROC per
    held-out domain fold, averaged — and (b) pooled out-of-fold scores, which
    give one global threshold, as a deployed monitor would need.
  * Transfer: probe + classifiers trained on all Core; zero-shot. Paper metric =
    mean of per-family AUROCs; pooled = all six families under one threshold.
  * Stego: zero-shot.
Pooled scores are averaged over layers (score ensemble) before computing
AUROC / AUPRC / recall@FPR with bootstrap CIs.

Usage: python e0_e1_lowfpr.py --act-root <dir> --layers 26-30 --out results/e1/<name>.json
"""
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from common import R, metrics, parse_layers, save_json, load_tier

METHODS = R.METHOD_KEYS


def method_outputs(tr_scores, tr_y, te_scores):
    """Final per-run score for each of the five methods (same as R.evaluate, minus AUROC)."""
    out = {k: te_scores[k] for k in ["peak_suspicion", "deception_split", "activation_alignment"]}
    fs, clf = R._train_asymmetry_classifier(tr_scores, tr_y)
    out["asymmetry_probe"] = clf.predict_proba(fs.transform(te_scores["asymmetry_probe"]))[:, 1]
    pca, fs2, clf2 = R._train_concat_classifier(tr_scores, tr_y)
    out["sorted_concat"] = clf2.predict_proba(fs2.transform(pca.transform(te_scores["sorted_concat"])))[:, 1]
    return out


def zscore(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-root", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    layers = parse_layers(args.layers)

    core_meta, core_npz = load_tier(args.act_root, "core")
    tr_meta, tr_npz = load_tier(args.act_root, "transfer")
    st_meta, st_npz = load_tier(args.act_root, "stego")

    paper = defaultdict(lambda: defaultdict(list))          # group -> method -> [auroc per layer]
    pooled = {t: defaultdict(list) for t in ["core", "transfer", "stego"]}  # tier -> method -> [scores per layer]
    labels, families = {}, {}

    for L in layers:
        print("layer", L, flush=True)
        # ---- Core leave-domain-out ----
        groups = R.build_groups(core_meta, core_npz, L, phase="discussion", avg_rounds=True)
        rids = sorted(groups)
        doms = sorted({groups[r]["domain"] for r in rids})
        oof = {m: np.full(len(rids), np.nan) for m in METHODS}
        fold_auc = defaultdict(list)
        for d in doms:
            tr = [r for r in rids if groups[r]["domain"] != d]
            te = [r for r in rids if groups[r]["domain"] == d]
            ty, ey = R._run_labels(groups, tr), R._run_labels(groups, te)
            if len(np.unique(ty)) < 2:
                continue
            sc, w = R.train_deception_probe(core_meta, core_npz, L, exclude_domain=d)
            out = method_outputs(R.compute_scores(groups, tr, sc, w), ty, R.compute_scores(groups, te, sc, w))
            idx = [rids.index(r) for r in te]
            for m in METHODS:
                oof[m][idx] = out[m]
                if len(np.unique(ey)) == 2:
                    fold_auc[m].append(roc_auc_score(ey, out[m]))
        for m in METHODS:
            paper["core_heldout"][m].append(float(np.mean(fold_auc[m])))
            pooled["core"][m].append(zscore(oof[m]))
        labels["core"] = R._run_labels(groups, rids)

        # ---- global probe for zero-shot tiers ----
        sc, w = R.train_deception_probe(core_meta, core_npz, L)
        core_scores = R.compute_scores(groups, rids, sc, w)
        core_y = R._run_labels(groups, rids)
        for tier, (meta, npz) in {"transfer": (tr_meta, tr_npz), "stego": (st_meta, st_npz)}.items():
            g = R.build_groups(meta, npz, L, phase="final", avg_rounds=False)
            gr = sorted(g)
            y = R._run_labels(g, gr)
            out = method_outputs(core_scores, core_y, R.compute_scores(g, gr, sc, w))
            fam = np.array([g[r]["scenario"] for r in gr])
            labels[tier], families[tier] = y, fam
            for m in METHODS:
                pooled[tier][m].append(zscore(out[m]))
                if tier == "transfer":
                    per = [roc_auc_score(y[fam == f], out[m][fam == f]) for f in np.unique(fam)
                           if len(np.unique(y[fam == f])) == 2]
                    paper["transfer_mean_family"][m].append(float(np.mean(per)))
                else:
                    paper["stego"][m].append(float(roc_auc_score(y, out[m])))

    res = {"act_root": args.act_root, "layers": layers, "paper_metric": {}, "pooled": {}}
    for g, d in paper.items():
        res["paper_metric"][g] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v))} for m, v in d.items()}
    for tier, d in pooled.items():
        res["pooled"][tier] = {m: metrics(labels[tier], np.mean(v, axis=0)) for m, v in d.items()}
    save_json(res, args.out)

    print("\nPaper-style AUROC (mean over layers):")
    for g, d in res["paper_metric"].items():
        print(f"  {g:22s}", "  ".join(f"{m[:10]}={d[m]['mean']:.3f}" for m in METHODS))
    print("\nPooled, layer-ensembled: AUROC | TPR@0 | TPR@1% | TPR@5% | TPR@10%  (n_pos/n_neg)")
    for tier, d in res["pooled"].items():
        for m in METHODS:
            x = d[m]
            print(f"  {tier:9s} {m:22s} {x['auroc']:.3f} | {x['tpr@0']:.2f} | {x['tpr@0.01']:.2f} | "
                  f"{x['tpr@0.05']:.2f} | {x['tpr@0.1']:.2f}  ({x['n_pos']}/{x['n_neg']})  "
                  f"AUROC CI [{x['auroc_ci'][0]:.2f},{x['auroc_ci'][1]:.2f}]")


if __name__ == "__main__":
    main()
