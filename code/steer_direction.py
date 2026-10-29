"""Steering direction for the causal test (PLAN: steering / ablation along the probe direction).

Direction = difference of means of the last-token residual stream at one layer between probe-positive rows
(active colluders) and probe-negative rows, over all non-aware E2 v2 runs plus the v1 controls (the probe's own
training labels, `labels(meta, "exclude")`). Stored in raw activation space, unit length, with its scale:
  delta = mean projection of positives - mean projection of negatives (how far "honest" is from "colluding")
  sigma = std of the projection over all rows
A random unit direction (fixed seed) is stored as the specificity control.

Usage:
  python steer_direction.py --v2 runs/qwen3_14b/e2v2_*_c?_act --v1-controls runs/qwen3_14b/e2{sys,ben}_{a,b}_act \
      --layer 18 --out runs/qwen3_14b/steer/dir_L18.npz
"""
import argparse
from pathlib import Path

import numpy as np

import e2_analysis as A
from e2v2_analysis import labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", nargs="+", required=True)
    ap.add_argument("--v1-controls", nargs="*", default=[])
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    L = args.layer
    meta, acts = A.load_e2(args.v2, [L])
    if args.v1_controls:
        m1, a1 = A.load_e2(args.v1_controls, [L])
        keep = np.array([m["mode"] == "control" for m in m1])
        meta += [m for m, k in zip(m1, keep) if k]
        acts = {L: np.concatenate([acts[L], a1[L][keep]])}
    X = acts[L].astype(np.float32)
    y = labels(meta, "exclude")
    pos, neg = X[y == 1], X[y == 0]
    d = pos.mean(0) - neg.mean(0)
    d /= np.linalg.norm(d)
    proj = X[y >= 0] @ d
    delta = float((pos @ d).mean() - (neg @ d).mean())
    rng = np.random.RandomState(0)
    r = rng.randn(X.shape[1]).astype(np.float32)
    r -= (r @ d) * d  # orthogonal to the probe direction
    r /= np.linalg.norm(r)
    rp = X[y >= 0] @ r
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, direction=d, random=r, delta=delta, sigma=float(proj.std()), layer=L,
             n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()),
             random_delta=float((pos @ r).mean() - (neg @ r).mean()), random_sigma=float(rp.std()),
             resid_norm=float(np.linalg.norm(X, axis=1).mean()))
    print(f"layer {L}: pos {(y == 1).sum()} neg {(y == 0).sum()} | delta {delta:.2f} sigma {proj.std():.2f} | "
          f"random delta {(pos @ r).mean() - (neg @ r).mean():.2f} | mean residual norm {np.linalg.norm(X, axis=1).mean():.1f}")
    # sanity: how well does the bare direction separate positives from negatives?
    from sklearn.metrics import roc_auc_score
    print(f"AUROC of projection on direction: {roc_auc_score(y[y >= 0], proj):.3f}; random: {roc_auc_score(y[y >= 0], rp):.3f}")


if __name__ == "__main__":
    main()
