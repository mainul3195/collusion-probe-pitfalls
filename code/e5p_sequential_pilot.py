"""E5p: sequential (turn-by-turn) detection pilot on released NARCBench activations.

Uses rounds that the static baseline averages away:
  Core : public discussion rounds 1-3, then final (round 4)
  Stego: public discussion rounds 1-7, then final (round 8)
Per-agent per-round probe scores (Core: leave-domain-out probes; Stego: probe on all Core).

Online risk R_t, computed only from rounds <= t:
  last     : ca_t on round-t agent scores only (memoryless)
  cummean  : ca_t on agents' cumulative mean scores up to t (the static method at t=T)
  cusum    : CUSUM over the per-round ca_t statistic, x_t standardised with control
             statistics estimated on held-out folds; S_t = max(0, S_{t-1} + z_t - k), k=0.5
  dsplit_cm: fixed-size Deception Split on cumulative means (NARCBench-style, online)

Reports, per detector:
  * AUROC of R_t at every round t (how early is signal available)
  * anytime-alarm operation: run alarms if R_t >= tau for any t; tau chosen so at most
    alpha of control runs ever alarm (alpha in {0, 5%, 10%}) -> recall and mean first-alarm round.
Thresholds are fit on control runs of the other folds (5-fold by scenario pair).

Usage: python e5p_sequential_pilot.py --act-root <dir> --layers 26-30 --out results/e5p/<name>.json
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

import detectors as D
from common import load_tier, parse_layers, row_projections, runs_table, save_json

K_CUSUM = 0.5
ALPHAS = [0.0, 0.05, 0.10]


def round_matrix(run):
    """agents x rounds matrix of scores, rounds ordered (discussion..., final)."""
    names = sorted(run["agents"])
    keys = sorted({k for nm in names for k in run["agents"][nm]["rounds"]}, key=lambda k: k[1])
    keys = [k for k in keys if k[0] in ("discussion", "final")]
    M = np.array([[run["agents"][nm]["rounds"].get(k, np.nan) for k in keys] for nm in names])
    return M, keys


def per_round_stats(M, sigma0):
    T = M.shape[1]
    last = np.array([D.ca_t(M[:, t], sigma0)[0] for t in range(T)])
    cm = np.array([D.ca_t(np.nanmean(M[:, :t + 1], 1), sigma0)[0] for t in range(T)])
    ds = np.array([D.deception_split(np.nanmean(M[:, :t + 1], 1), 2)[0] for t in range(T)])
    return last, cm, ds


def cusum(x, mu, sd):
    S, out = 0.0, []
    for v in x:
        S = max(0.0, S + (v - mu) / (sd + 1e-9) - K_CUSUM)
        out.append(S)
    return np.array(out)


def analyse(runs, sigma0, n_folds=5, seed=42):
    rids = sorted(runs)
    pair = {r: r.rsplit("__", 1)[0] for r in rids}
    pairs = sorted(set(pair.values()))
    rng = np.random.RandomState(seed); rng.shuffle(pairs)
    fold = {p: i % n_folds for i, p in enumerate(pairs)}
    y = np.array([int(runs[r]["mode"] == "collusion") for r in rids])
    f = np.array([fold[pair[r]] for r in rids])

    stats = {}
    for r in rids:
        M, keys = round_matrix(runs[r])
        stats[r] = per_round_stats(M, sigma0)
    T = min(len(v[0]) for v in stats.values())
    R = {"last": np.array([stats[r][0][:T] for r in rids]),
         "cummean": np.array([stats[r][1][:T] for r in rids]),
         "dsplit_cm": np.array([stats[r][2][:T] for r in rids])}
    # CUSUM: standardise 'last' with control mean/sd from other folds
    cu = np.zeros((len(rids), T))
    for k in range(n_folds):
        tr = (f != k) & (y == 0)
        mu, sd = R["last"][tr].mean(), R["last"][tr].std()
        for i in np.where(f == k)[0]:
            cu[i] = cusum(R["last"][i], mu, sd)
    R["cusum"] = cu

    out = {"T": T, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    for name, X in R.items():
        d = {"auroc_by_round": [float(roc_auc_score(y, X[:, t])) for t in range(T)]}
        for a in ALPHAS:
            alarm_round = np.full(len(rids), np.inf)
            for k in range(n_folds):
                ctrl_max = np.sort(X[(f != k) & (y == 0)].max(1))[::-1]
                j = int(np.floor(a * len(ctrl_max)))
                tau = ctrl_max[j] if j < len(ctrl_max) else -np.inf
                for i in np.where(f == k)[0]:
                    hit = np.where(X[i] > tau)[0]
                    if len(hit):
                        alarm_round[i] = hit[0] + 1
            det = np.isfinite(alarm_round)
            d[f"alpha={a:g}"] = {
                "recall": float(det[y == 1].mean()),
                "realised_fpr": float(det[y == 0].mean()),
                "mean_first_alarm_round": float(alarm_round[(y == 1) & det].mean()) if det[y == 1].any() else None,
            }
        out[name] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-root", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    layers = parse_layers(args.layers)
    core_meta, core_npz = load_tier(args.act_root, "core")
    core_runs = runs_table(core_meta, row_projections(core_meta, core_npz, core_meta, core_npz, layers, oof=True))
    hon = [x for r in core_runs.values() if r["mode"] == "control"
           for a in r["agents"].values() for (ph, _), x in a["rounds"].items() if ph == "discussion"]
    sigma0 = float(np.std(hon))
    st_meta, st_npz = load_tier(args.act_root, "stego")
    st_runs = runs_table(st_meta, row_projections(core_meta, core_npz, st_meta, st_npz, layers))
    res = {"layers": layers, "sigma0": sigma0,
           "core": analyse(core_runs, sigma0), "stego": analyse(st_runs, sigma0)}
    save_json(res, args.out)
    for tier in ["core", "stego"]:
        r = res[tier]
        print(f"\n== {tier} (T={r['T']}, {r['n_pos']}+{r['n_neg']})")
        for name in ["last", "cummean", "dsplit_cm", "cusum"]:
            d = r[name]
            print(f"  {name:10s} AUROC by round:", " ".join(f"{v:.2f}" for v in d["auroc_by_round"]))
            for a in ALPHAS:
                x = d[f"alpha={a:g}"]
                print(f"      alpha={a:<4g} recall={x['recall']:.2f} fpr={x['realised_fpr']:.2f} "
                      f"first_alarm={x['mean_first_alarm_round']}")


if __name__ == "__main__":
    main()
