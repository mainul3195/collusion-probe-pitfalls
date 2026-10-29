"""Robustness of the record-based committee results to the option-label mismatch (label_mismatch.py).

Recomputes the pre-vote, all-attempts static metrics (as e2v2_analysis.summarise: positives = every collusion attempt,
negatives = the headline families) from the saved cross-fitted run records, once on all runs and once without the
flagged runs (positives and negatives alike). No probe is refitted: the records are the
cross-fitted scores of the main analysis, so "clean" asks whether the reported numbers hold on the unaffected episodes.

Usage:
  python mismatch_robustness.py --mismatch results/paper1/audit/label_mismatch.json \
      --records com.act=results/e2v2b/act.json com.recipe=results/e2v2b/act_paper_recipe.json ... \
      --out results/paper1/robustness/label_mismatch.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from common import metrics
from e2v2_analysis import HEADLINE

KEYS = ["peak|cummean", "ca_t|cummean", "ca_glr|cummean"]  # as the paper keys (make_numbers: cummean)


def short(recs, keys):
    """Pre-vote static metrics, positives = all attempts (successful or failed), negatives = headline families."""
    pos = [r for r in recs if r["set"] in ("pos", "fail")]
    neg = [r for r in recs if r["set"] == "neg" and r["mode"] in HEADLINE]
    dec = [r for r in neg if r["mode"] == "briefed_decline"]
    out = {"counts": {"attempts": len(pos), "neg": len(neg), "briefed_decline": len(dec)}}
    for k in keys:
        a, n = [r["prevote"][k] for r in pos], [r["prevote"][k] for r in neg]
        m = metrics(np.r_[np.ones(len(a)), np.zeros(len(n))], np.r_[a, n], n_boot=500)
        d = [r["prevote"][k] for r in dec]
        out[k] = {f: m[f] for f in ("auroc", "auroc_ci", "tpr@0.01", "tpr@0.01_ci")} | {
            "auroc_vs_briefed_decline": float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(d))], np.r_[a, d]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mismatch", required=True)
    ap.add_argument("--records", nargs="+", required=True, help="name=path of an e2v2_analysis output with records")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    flagged = {k for k, r in json.loads(Path(args.mismatch).read_text())["runs"].items() if r["mismatch"]}
    res, lines = {}, []
    for spec in args.records:
        name, path = spec.split("=", 1)
        recs = [r for r in json.loads(Path(path).read_text())["records"] if not r.get("aware")]
        keys = [k for k in KEYS if k in recs[0]["prevote"]]
        clean = [r for r in recs if r["rid"] not in flagged]
        res[name] = {"all": short(recs, keys), "clean": short(clean, keys),
                     "n_dropped": len(recs) - len(clean)}
        for k in keys:
            a, c = res[name]["all"][k], res[name]["clean"][k]
            lines.append(f"{name:12s} {k:12s} " + "  ".join(
                f"{f} {a[f]:.3f}->{c[f]:.3f}" for f in ("auroc", "tpr@0.01", "auroc_vs_briefed_decline")
                if a[f] is not None and c[f] is not None) + f"  (dropped {res[name]['n_dropped']} runs)")
    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(res, indent=1))
    o.with_suffix(".log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
