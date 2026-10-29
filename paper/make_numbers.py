"""Generate numbers.tex: every reported number as a LaTeX macro, read from experiments/results/.

The thesis and all paper manuscripts use these macros instead of typed numbers, so a rerun changes every document
at once and a stale number cannot survive. Use in LaTeX (after \\input{.../numbers}):

    \\R{com.act.peak.pa.tpr1}      -> 0.84
    \\R{com.act.peak.pa.tpr1ci}    -> [0.80, 0.87]
    \\R{com.act.peak.pa.auroc}     -> 0.99

An unknown key typesets as a bold red ??key and writes a LaTeX warning, so a typo is visible in the PDF.

Key scheme  <setting>.<signal>.<detector>.<split>.<metric>
  setting   com (Qwen3-14B committee, main), negx (committee, 3,894 negatives), q32 (Qwen3-32B), q32m (Qwen3-14B on
            the 32B cells), mis (Mistral-Small-24B), misd (Qwen3-14B on the Mistral cells), cr (code review)
  signal    act (probe, layers 16/18/20), deep (deeper layers), tmpl (calibrated on an unseen wording), recipe
            (published recipe on our data), core (NARCBench probe), text (TF-IDF), enc (ModernBERT), judgeq / judgem
            (LLM judges), vote (vote rule)
  detector  peak, cat, caglr, dsplit, mean (time-aggregated with cummean); best (best |last aggregator, for the
            signals that only have one value per step: judges, encoder, vote)
  split     pa (attempts before the vote), ps (successful, before the vote), fa (attempts, end), fs (successful, end)
  metric    tpr1 tpr01 tpr05 tpr5 (recall at 1%, 0.1%, 0.5%, 5% FPR), tpr1ci tpr01ci, auroc, auroc3, npos, nneg, jacc
Plus: seeds.*, tr.* (cross-task transfer), steer.*, emerg.*, and counts.
Paper 1 H1-H6: hone.* (stance check, kappa), hthree.* (label audit, % flagged), ceil.<com|cr>.* (public-record
ceiling), ctx.<com|cr>.<system|instr|full>.* (context reader), hfive.*, hsix.* / hsixv.* (vote-only joiners),
hfour.*, dec.* / decv.* (instruction vs. decision).

Run from anywhere:  python papers/shared/make_numbers.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "numbers.tex"

FILES = {  # setting.signal -> result file
    "com.act": "e2v2b/act.json", "com.recipe": "e2v2b/act_paper_recipe.json", "com.core": "e2v2b/core.json",
    "com.text": "e2v2b/text.json", "com.errfree": "e2v2b/act_errfree.json", "com.enc": "paper1/encoder.json", "com.judgeq": "e2v2b/judge_logprob.json",
    "com.judgem": "e2v2b/judge_mistral.json", "com.vote": "e2v2b/vote.json", "com.tmpl": "e7/calib_tmpl_act.json",
    "com.split": "e7/calib_split_act.json", "e7.l14": "e7/layer_14.json", "e7.l18": "e7/layer_18.json", "e7.l22": "e7/layer_22.json",
    "e7.l24": "e7/layer_24.json", "e7.l26": "e7/layer_26.json", "e7.l28": "e7/layer_28.json", "e7.l30": "e7/layer_30.json",
    "e7.feat": "e7/feat_mean.json", "e7.t1": "e7/templates_1.json", "e7.pos": "e7/pos_success.json",
    "e7.splittext": "e7/calib_split_text.json", "com.ibneg": "e2v2b/act_ibneg.json",
    "negx.act": "paper1/negx/act.json", "negx.tmpl": "paper1/negx/act_tmpl.json", "negx.text": "paper1/negx/text.json",
    "q32.act": "e8b/act.json", "q32.deep": "e8b/act_deep.json", "q32.tmpl": "e8b/act_tmpl.json",
    "q32.text": "e8b/text.json", "q32.vote": "e8b/vote.json",
    "q32m.act": "e8b/matched14b_act.json", "q32m.text": "e8b/matched14b_text.json",
    "mis.act": "paper1/mistral/act.json", "mis.deep": "paper1/mistral/act_deep.json",
    "mis.tmpl": "paper1/mistral/act_tmpl.json", "mis.recipe": "paper1/mistral/act_paper_recipe.json",
    "mis.text": "paper1/mistral/text.json", "mis.vote": "paper1/mistral/vote.json",
    "misd.act": "paper1/matched14b/act.json", "misd.recipe": "paper1/matched14b/act_paper_recipe.json",
    "misd.text": "paper1/matched14b/text.json",
    "cr.act": "paper1/codereview/act.json", "cr.deep": "paper1/codereview/act_deep.json",
    "cr.tmpl": "paper1/codereview/act_tmpl.json", "cr.recipe": "paper1/codereview/act_paper_recipe.json",
    "cr.text": "paper1/codereview/text.json", "cr.vote": "paper1/codereview/vote.json",
}
# decision probe through the full pipeline (e2v2_analysis.py --signal decision): signal dec + joiner suffix
# (none = vote or private agreement, s = strict, v = vote only) + n if trained with non-briefed benchmark negatives,
# nb if also with the benign free-choice choosers (BC=1; any and strict only)
for _j, _js in (("any", ""), ("strict", "s"), ("vote", "v")):
    for _d, _ds in (("", ""), ("_dn", "n"), ("_dn_bc", "nb")):
        if _ds == "nb" and _j == "vote":
            continue
        for _set, _file in (("com", "com"), ("comtmpl", "com_tmpl"), ("negx", "negx"), ("cr", "cr")):
            FILES[f"{_set}.dec{_js}{_ds}"] = f"paper1/decision/{_file}{_d}_{_j}.json"
DET = {"peak": "peak", "ca_t": "cat", "ca_glr": "caglr", "dsplit_k2": "dsplit", "mean_all": "mean"}
SPLIT = {"prevote_attempts": "pa", "prevote": "ps", "final_attempts": "fa", "all": "fs"}


def f2(x):
    return f"{x:.2f}"


def ci(v):
    return f"[{v[0]:.2f}, {v[1]:.2f}]"


def metrics(st):
    m = {"tpr1pct": f"{100 * st['tpr@0.01']:.0f}", "tpr01pct": f"{100 * st['tpr@0.001']:.0f}", "tpr1": f2(st["tpr@0.01"]), "tpr01": f2(st["tpr@0.001"]), "tpr05": f2(st["tpr@0.005"]),
         "tpr5": f2(st["tpr@0.05"]), "auroc": f2(st["auroc"]), "auroc3": f"{st['auroc']:.3f}",
         "npos": f"{st['n_pos']:,}".replace(",", "{,}"), "nneg": f"{st['n_neg']:,}".replace(",", "{,}")}
    for k, src in (("tpr1ci", "tpr@0.01_ci"), ("tpr01ci", "tpr@0.001_ci"), ("aurocci", "auroc_ci")):
        if src in st:
            m[k] = ci(st[src])
    if isinstance(st.get("jaccard"), (int, float)):
        m["jacc"] = f2(st["jaccard"])
    if isinstance(st.get("coordination_auroc_pos_vs_ib"), (int, float)):
        m["coord"] = f2(st["coordination_auroc_pos_vs_ib"])
    if isinstance(st.get("intent_auroc_fail_vs_neg"), (int, float)):
        m["intent"] = f2(st["intent_auroc_fail_vs_neg"])
    for k, n in (("ib_flag_rate", "ibflag"), ("fail_flag_rate", "failflag")):
        if isinstance(st.get(k), dict):
            m[n + "1"], m[n + "5"] = f2(st[k]["0.01"]), f2(st[k]["0.05"])
    for f, v in (st.get("auroc_vs_family") or {}).items():
        if v is not None:
            m[f"aurocvs.{f.replace('_', '')}"] = f2(v)
    fam = st.get("fpr_by_family")
    if isinstance(fam, dict):
        for a, nm in (("0.01", "fprfam1"), ("0.05", "fprfam5")):
            for f, v in fam.get(a, {}).items():
                m[f"{nm}.{f.replace('_', '')}"] = f"{v:.3f}"
    return m


def collect(RES):
    out = {}
    missing = []
    for pre, rel in FILES.items():
        p = RES / rel
        if not p.exists():
            missing.append(rel)
            continue
        d = json.loads(p.read_text())
        for split, sk in SPLIT.items():
            if split not in d or "static" not in d[split]:
                continue
            st = d[split]["static"]
            for k, v in st.items():
                det, agg = k.split("|")
                if agg == "cummean" and det in DET:
                    for mk, mv in metrics(v).items():
                        out[f"{pre}.{DET[det]}.{sk}.{mk}"] = mv
            for k, v in d[split].get("anytime", {}).items():
                det, agg, al = k.split("|")
                if det in DET and agg in ("cummean", "last") and al in ("0.01", "0.05"):
                    t = f"{pre}.{DET[det]}.{sk}.any{'1' if al == '0.01' else '5'}{'last' if agg == 'last' else ''}"
                    out[t + ".recall"], out[t + ".fpr"] = f2(v["recall"]), f"{v['fpr']:.3f}"
                    if v.get("delay_median") is not None:
                        out[t + ".delay"] = f"{v['delay_median']:.0f}"
            last = [k for k in st if k.endswith("|last")]
            if last:
                best = max(last, key=lambda k: st[k]["tpr@0.01"])
                for mk, mv in metrics(st[best]).items():
                    out[f"{pre}.best.{sk}.{mk}"] = mv
    # seeds: main probe + two extra probe seeds
    seeds = [RES / "e2v2b/act.json", RES / "paper1/seeds/act_seed1.json", RES / "paper1/seeds/act_seed2.json"]
    if all(s.exists() for s in seeds):
        for k, n in (("peak|cummean", "peak"), ("ca_t|cummean", "cat"), ("ca_glr|cummean", "caglr")):
            v = [json.loads(s.read_text())["prevote_attempts"]["static"][k]["tpr@0.01"] for s in seeds]
            out[f"seeds.{n}.min"], out[f"seeds.{n}.max"] = f2(min(v)), f2(max(v))
    # cross-task transfer
    for name, tag in (("committee_to_codereview", "c2r"), ("codereview_to_committee", "r2c")):
        p = RES / f"paper1/transfer/{name}.json"
        if p.exists():
            t = json.loads(p.read_text())
            out[f"tr.{tag}.npos"], out[f"tr.{tag}.nneg"] = str(t["n_pos"]), str(t["n_neg"])
            for k, v in t["detectors"].items():
                for mk, mv in v.items():
                    out[f"tr.{tag}.{DET[k.split('|')[0]]}.{mk.replace('_', '')}"] = (
                        f"{mv:.3f}" if mk == "auroc" else (f"{100 * mv:.1f}" if mk.startswith("fpr") else f2(mv)))
    # steering
    p = RES / "steer/outcomes.json"
    if p.exists():
        o = json.loads(p.read_text())
        for c, v in o.items():
            for mk in ("success", "share", "majority"):
                if mk in v:
                    out[f"steer.{c.replace('_', '')}.{mk}"] = f2(v[mk][0])
                    out[f"steer.{c.replace('_', '')}.{mk}ci"] = ci(v[mk][1:])
            c2 = c.replace("_", "")
            out[f"steer.{c2}.n"] = str(v["n"])
            if v.get("paired_share_diff"):
                d = v["paired_share_diff"]
                out[f"steer.{c2}.sharediff"] = f"{d[0]:+.2f}".replace("-", "$-$")
                out[f"steer.{c2}.sharediffci"] = f"[{d[1]:+.2f}, {d[2]:+.2f}]".replace("-", "$-$")
            if v.get("short_msgs"):
                out[f"steer.{c2}.short"], out[f"steer.{c2}.msgs"] = str(v["short_msgs"][0]), f"{v['short_msgs'][1]:,}".replace(",", "{,}")
    p = RES / "steer/mcnemar.json"
    if p.exists():
        for c, v in json.loads(p.read_text()).items():
            c2 = c.replace("_", "")
            out[f"steer.{c2}.lost"], out[f"steer.{c2}.gained"] = str(v["lost"]), str(v["gained"])
            out[f"steer.{c2}.p"] = f"{v['p']:.3f}" if v["p"] < 0.1 else f"{v['p']:.2f}"
    p = RES / "steer/scores.json"
    if p.exists():
        s = json.loads(p.read_text())["conditions"]
        for c, v in s.items():
            out[f"steer.{c.replace('_', '')}.flagpeak"] = f2(v["flag_peak"])
            out[f"steer.{c.replace('_', '')}.flagcat"] = f2(v["flag_ca_t"])
            if v.get("paired_briefed_diff"):
                d = v["paired_briefed_diff"]
                out[f"steer.{c.replace('_', '')}.scorediff"] = f"{d[0]:+.2f}".replace("-", "$-$")
                out[f"steer.{c.replace('_', '')}.scorediffci"] = ci(d[1:]).replace("-", "$-$")
            out[f"steer.{c.replace('_', '')}.score"] = f2(v["briefed_score"][0])
    # monitor-aware and notice-only agents (paired 1% threshold of the non-aware probe)
    for fn, tag in (("aware/act.json", "aw"), ("aware/notice_only_act.json", "notice")):
        p = RES / fn
        if p.exists():
            a = json.loads(p.read_text())["aware"]["static"]
            for k, v in a.items():
                when, det, agg = k.split("|")
                if when != "prevote" or agg != "cummean" or det not in DET:
                    continue
                for grp in ("attempts", "successes"):
                    for al, an in (("0.01", "1"), ("0.05", "5")):
                        for side in ("aware", "base_paired"):
                            x = v.get(grp, {}).get(al, {}).get(side)
                            if x is not None:
                                out[f"{tag}.{DET[det]}.{grp[:3]}{an}.{side.replace('_', '')}"] = f2(x)
                for fam in ("control", "benign_coord", "briefed_decline"):
                    x = v.get(fam, {}).get("0.01", {}).get("aware")
                    if x is not None:
                        out[f"{tag}.{DET[det]}.fpr1.{fam.replace('_', '')}"] = f"{x:.3f}"
    # Paper 1 no-brief test (PREREG section 1): pilot gate (implant_gate.py, copied to results/paper1/nobrief/)
    p = RES / "paper1/nobrief/gate.json"
    if p.exists():
        g = json.loads(p.read_text())
        out["nb.gate.passed"], out["nb.gate.episodes"] = ("PASS" if g["passed"] else "FAIL"), str(g["episodes"])
        for m, k in (("implanted", "imp"), ("implanted_ctl", "ctl"), ("implanted_hon", "hon")):
            if m in g.get("rates", {}):
                out[f"nb.gate.{k}.rate"], out[f"nb.gate.{k}.seats"] = f"{g['rates'][m]['target_vote']:.3f}", str(g["rates"][m]["seats"])
            if m in g and m != "implanted":
                out[f"nb.gate.{k}.diff"], out[f"nb.gate.{k}.ci"] = f"{g[m]['diff']:+.2f}", ci(g[m]["ci"])
                out[f"nb.gate.{k}.scenci"] = ci(g[m]["ci_scenario_clusters"])
    p = RES / "paper1/nobrief/main.json"  # main run: implant_analysis.py
    if p.exists():
        g = json.loads(p.read_text())
        for fam, k in (("implanted", "imp"), ("implanted_hon", "hon"), ("implanted_ctl", "ctl")):
            out[f"nb.main.{k}.runs"] = str(g["info"]["runs"][fam])
        out["nb.main.episodes"] = str(sum(g["info"]["runs"].values()))
        for pr in ("P1", "P2"):
            d, t = g[pr], pr.lower()
            out[f"nb.main.{t}.success"] = "yes" if d["success"] else "no"
            for c in ("hon", "ctl"):
                x = d[f"vs_{c}"]
                out[f"nb.main.{t}.{c}.auroc"], out[f"nb.main.{t}.{c}.ci"] = f"{x['auroc']:.3f}", ci(x["ci"])
                out[f"nb.main.{t}.{c}.scenci"] = ci(x["ci_scenario_clusters"])
                out[f"nb.main.{t}.{c}.npos"], out[f"nb.main.{t}.{c}.nneg"] = str(x["n_pos"]), str(x["n_neg"])
            for fk, v in d["run_level"].items():
                out[f"nb.main.{t}.run.{fk}"] = f2(v["flagged"])
                out[f"nb.main.{t}.run.{fk}pct"] = f"{100 * v['flagged']:.0f}"
    # Paper 1 no-brief text readers and concealed messages (PREREG amendments 3-4): implant_text.py, parsed from text.log
    p = RES / "paper1/nobrief/text.log"
    if p.exists():
        import re
        lines = p.read_text().splitlines()
        L = lambda pre: next(ln for ln in lines if ln.startswith(pre))  # noqa: E731
        num = r"([\d.]+) \[([\d.]+), ([\d.]+)\]"
        for pre, tag in (("stance reader, agent", "stance"), ("TF-IDF reader", "tfidf"), ("P1 on concealed", "p1"),
                         ("P2 on concealed", "p2")):
            ln = L(pre)
            for c in ("hon", "ctl"):
                m = re.search(rf"vs_{c} {num}", ln)
                out[f"nb.text.{tag}.{c}.auroc"] = f"{float(m[1]):.3f}"
                out[f"nb.text.{tag}.{c}.ci"] = ci([float(m[2]), float(m[3])])
                m = re.search(rf"vs_{c} round 1 ([\d.]+)", ln)
                if m:
                    out[f"nb.text.{tag}.{c}.r1"] = f2(float(m[1]))
        m = re.search(r"threshold \(([\d.]+), (\d+) runs\): \{'imp': ([\d.]+)", L("stance reader, run"))
        out["nb.text.stance.run.thr"], out["nb.text.stance.run.nneg"] = m[1], f"{int(m[2]):,}".replace(",", "{,}")
        out["nb.text.stance.run.imppct"] = f"{100 * float(m[3]):.0f}"
        m = re.search(r"concealed messages \{'imp': (\d+), 'hon': (\d+), 'ctl': (\d+)\}", p.read_text())
        out["nb.conc.imp"], out["nb.conc.hon"], out["nb.conc.ctl"] = m[1], m[2], m[3]
    # Paper 1 no-brief concealed messages, private cut (PREREG amendment 6 (a), secondary; no reading attached)
    p = RES / "paper1/nobrief/text_private.json"
    if p.exists():
        c = json.loads(p.read_text())["concealed"]
        out["nb.priv.imp"], out["nb.priv.hon"], out["nb.priv.ctl"] = (str(c["messages"][k]) for k in ("imp", "hon", "ctl"))
        for pr in ("P1", "P2"):
            for cc in ("hon", "ctl"):
                x = c[pr][f"vs_{cc}"]
                out[f"nb.priv.{pr.lower()}.{cc}.auroc"], out[f"nb.priv.{pr.lower()}.{cc}.ci"] = f"{x['auroc']:.3f}", ci(x["ci"])
                out[f"nb.priv.{pr.lower()}.{cc}.r1"] = f2(x["round1"])
    # Paper 1 AcMAS-style divergence check (amendment 6 (b)): acmas_L<layer>.json, layer 40 primary, 30 secondary
    for p in sorted((RES / "paper1/acmas").glob("acmas_L*.json")):
        a = json.loads(p.read_text())
        for var in ("main", "proto_all"):
            d, t = a[var], f"acmas.L{a['layer']}" + ("" if var == "main" else ".pall")
            for b in ("ctl", "dec", "ben"):
                out[f"{t}.{b}.auroc"], out[f"{t}.{b}.ci"] = f"{d[f'auc_col_vs_{b}']:.3f}", ci(d[f"auc_col_vs_{b}_ci"])
            for g in ("col", "dec", "ben", "ctl"):
                out[f"{t}.tau.{g}pct"] = f"{100 * d['flag_at_tau'][g]:.0f}"
                out[f"{t}.one.{g}pct"] = f"{100 * d['flag_at_1pct'][g]:.0f}"
                out[f"{t}.n.{g}"] = f"{d['n'][g]:,}".replace(",", "{,}")
            out[f"{t}.reading"] = d["reading"]
    # Paper 1 no-model metadata baseline (amendment 6 (c))
    p = RES / "paper1/acmas/meta_baseline.json"
    if p.exists():
        m = json.loads(p.read_text())
        for mk in ("M1", "M2"):
            for b in ("ctl", "dec", "ben"):
                out[f"meta.{mk.lower()}.{b}.auroc"] = f"{m[mk][f'auc_col_vs_{b}']:.3f}"
                out[f"meta.{mk.lower()}.{b}.ci"] = ci(m[mk][f"auc_col_vs_{b}_ci"])
        out["meta.reading"] = m["reading"]
    # Paper 1 decision-probe learning curve (PREREG section 2, amendments 1-2): pre-vote attempts, peak|cummean, 1% FPR
    cur = {}
    for p in sorted((RES / "paper1/decision_curve").glob("com*_s?.json")):
        j, n, sd = p.stem.replace("com_dn_", "com_").split("_")[1:]
        st = json.loads(p.read_text())["prevote_attempts"]["static"]["peak|cummean"]
        cur.setdefault(j, {}).setdefault(int(n), []).append((st["tpr@0.01"], st["tpr@0.01_ci"], st["auroc"]))
    for j, sizes in cur.items():
        full = max(sizes)
        half = [x[0] for x in sizes.get(full // 2, sizes.get((full + 1) // 2, []))]
        f_tpr, f_ci, f_auc = sizes[full][0]
        out[f"curve.{j}.full.n"], out[f"curve.{j}.full.tpr1"] = f"{full:,}".replace(",", "{,}"), f2(f_tpr)
        out[f"curve.{j}.full.tpr1ci"], out[f"curve.{j}.full.auroc"] = ci(f_ci), f2(f_auc)
        out[f"curve.{j}.min.n"] = str(min(sizes))
        out[f"curve.{j}.min.tpr1"] = f2(sum(x[0] for x in sizes[min(sizes)]) / len(sizes[min(sizes)]))
        if half:
            gain = f_tpr - sum(half) / len(half)
            out[f"curve.{j}.half.tpr1"], out[f"curve.{j}.gain"] = f2(sum(half) / len(half)), f"{gain:+.3f}"
            out[f"curve.{j}.reading"] = "data-limited" if gain >= 0.05 and all(f_tpr > h for h in half) else "flat"
    # GPU-hours from the job queue (experiments/code/gpu_hours.py)
    p = RES / "cost/gpu_hours.json"
    if p.exists():
        c = json.loads(p.read_text())
        for k, v in c["groups"].items():
            out[f"cost.{k}"] = f"{v['gpu_hours']:.1f}"
        out["cost.total"] = f"{c['total']:.0f}"
        out["cost.paperone"] = f"{c['paper1_new']:.0f}"
        out["cost.date"] = c["generated"].split(" ")[0]
    # emergent collusion reader test
    p = RES / "e8c/reader.json"
    if p.exists():
        e = json.loads(p.read_text())["main, 10 agent models|r4"]
        for lab in ("collusion", "agreement"):
            for k, n in (("v2|mean", "ours"), ("core|mean", "core"), ("text|mean", "text"), ("kw|mean", "kw"),
                         ("indomain|mean", "indomain")):
                if k in e[lab]:
                    out[f"emerg.{lab}.{n}.auroc"] = f2(e[lab][k]["auroc"])
    # E1: NARCBench reproduction on its released activations; best method per model on the pooled transfer tier
    # (by AUROC), its recall at 1% FPR, and the largest number of negatives in a pooled tier
    best, nneg = [], []
    for m in ("qwen3_32b", "llama_3_1_70b", "deepseek_r1_32b", "gpt_oss_20b"):
        p = RES / f"e1/{m}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        tr = d["pooled"]["transfer"]
        k = max(tr, key=lambda x: tr[x]["auroc"])
        best.append((tr[k]["auroc"], tr[k]["tpr@0.01"]))
        nneg += [v["n_neg"] for tier in d["pooled"].values() for v in tier.values()]
        if m == "qwen3_32b":
            out["e1.q32.core.auroc"] = f2(min(v["mean"] for v in d["paper_metric"]["core_heldout"].values()))
    if best:
        out["e1.best.auroc.min"], out["e1.best.auroc.max"] = f2(min(a for a, _ in best)), f2(max(a for a, _ in best))
        out["e1.best.tpr1pct.min"] = f"{100 * min(t for _, t in best):.0f}"
        out["e1.best.tpr1pct.max"] = f"{100 * max(t for _, t in best):.0f}"
        out["e1.nneg.max"] = str(max(nneg))
    # Paper 1 H1-H6: stance check, label audit, public-record ceiling, context reader, instruction vs. decision
    p = RES / "paper1/audit/review_score.json"
    if p.exists():
        for who, v in json.loads(p.read_text()).items():
            # annotator names in the working files; anonymised IDs a1-a3 in the public release
            w = "maj" if who == "majority" else (who.lower() if re.fullmatch(r"[aA]\d", who) else
                                                 "".join(ch for ch in who if ch.isalpha()))
            out[f"hone.{w}.kappa"] = f2(v["kappa"])
            out[f"hone.{w}.agree"] = f2(v["agreement"])
            if who == "majority":
                out["hone.maj.n"], out["hone.unanimous"] = str(v["n"]), str(v["unanimous"])
                out["hone.items"], out["hone.annotators"] = str(v["n_items"]), str(v["annotators"])
    p = RES / "paper1/audit/review_score.log"
    if p.exists():
        pair = [float(m) for m in re.findall(r"^stance, (?!majority)\S+ vs (?!model)\S+: n=\d+, kappa ([\d.]+)",
                                             p.read_text(), re.M)]
        if pair:
            out["hone.pair.kappa.min"], out["hone.pair.kappa.max"] = f2(min(pair)), f2(max(pair))
    p = RES / "paper1/audit/audit.json"
    if p.exists():
        aud = json.loads(p.read_text())
        for mode, tag in (("collusion", "col"), ("briefed_decline", "dec"), ("benign_coord", "ben")):
            for task, tt in (("", "all"), ("__cr_", "cr"), ("com", "com")):
                xs = [r for r in aud if r["mode"] == mode and (task == "" or (task == "__cr_") == ("__cr_" in r["rid"]))]
                if xs:
                    out[f"hthree.{tag}.{tt}.n"] = str(len(xs))
                    out[f"hthree.{tag}.{tt}.flag"] = f"{100 * sum(r['flag'] is not None for r in xs) / len(xs):.1f}"
        out["hthree.runs"] = f"{len(aud):,}"
    for task, tt in (("committee", "com"), ("codereview", "cr")):
        p = RES / f"paper1/ceiling/{task}.json"
        if p.exists():
            c = json.loads(p.read_text())
            for nm, k in (("probe (peak, cross-fitted)", "probe"), ("stance oracle (GBM)", "oracle"),
                          ("max public stance", "maxst")):
                d = c["detectors"][nm]
                out[f"ceil.{tt}.{k}.tpr1"], out[f"ceil.{tt}.{k}.tpr1ci"] = f2(d["tpr1"]), ci(d["tpr1_ci"])
                out[f"ceil.{tt}.{k}.tpr1pct"] = f"{100 * d['tpr1']:.0f}"
                out[f"ceil.{tt}.{k}.auroc"], out[f"ceil.{tt}.{k}.tpr5"] = f2(d["auroc"]), f2(d["tpr5"])
            iv = c["invisible"]
            out[f"ceil.{tt}.inv.n"], out[f"ceil.{tt}.inv.share"] = str(iv["n"]), f"{100 * iv['share']:.0f}"
            out[f"ceil.{tt}.inv.sharesucc"] = f"{100 * iv['share_success']:.0f}"
            out[f"ceil.{tt}.inv.probe"], out[f"ceil.{tt}.inv.oracle"] = f2(iv["probe_recall"]), f2(iv["oracle_recall"])
            out[f"ceil.{tt}.vis.probe"] = f2(iv["probe_recall_visible"])
            pr = c["pairs"]
            out[f"ceil.{tt}.pairs.n"] = str(pr["n"])
            for k in ("probe", "oracle", "max_stance"):
                out[f"ceil.{tt}.pairs.{k.replace('_', '')}"] = f2(pr[k])
        p = RES / f"paper1/context/{task}.json"
        if p.exists():
            for v, d in json.loads(p.read_text())["views"].items():
                for m in ("auroc", "tpr1", "tpr5", "agent_auc_col_vs_dec"):
                    out[f"ctx.{tt}.{v}.{m.replace('_', '')}"] = f2(d[m])
    # H5 brief removal: Qwen3-14B committee (main) and the replications of PREREG amendment 5 (code review,
    # Mistral-24B, Qwen3-32B) with their pre-registered reading on point estimates
    for sfx, tag in (("", "hfive"), ("_cr", "hfivecr"), ("_mistral", "hfivemis"), ("_q32", "hfiveqtt")):
        p = RES / f"paper1/instruction/h5_counterfactual{sfx}.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        for cond, d in h.items():
            for k, v in d.items():
                if k.startswith("n_"):
                    out[f"{tag}.{cond}.{k.replace('_', '')}"] = str(v)
                else:
                    out[f"{tag}.{cond}.{k.replace('_auc_vs_', '').replace('_', '')}"] = (ci(v) if isinstance(v, list) else
                                                                                f"{v:.3f}" if sfx else f2(v))
        if sfx:  # three decimals: the reading's cut (0.75) must be visible next to the value
            nm, st = h["normal"]["agent_auc_vs_briefed_decline"], h["stripped"]["agent_auc_vs_briefed_decline"]
            out[f"{tag}.reading"] = ("replicates" if nm >= 0.90 and st <= 0.75 else
                                     "does not replicate" if st >= 0.90 else "partial")
    for sfx, tag in (("", "hsix"), ("_vote", "hsixv")):  # joiner = vote or private agreement / vote only
        p = RES / f"paper1/instruction/choice{sfx}.json"
        if p.exists():
            d = json.loads(p.read_text())
            out[f"{tag}.njoin"], out[f"{tag}.nrefuse"] = str(d["n_join"]), str(d["n_refuse"])
            out[f"{tag}.auroc"], out[f"{tag}.zjoin"], out[f"{tag}.zrefuse"] = (
                f2(d["auc_join_vs_refuse"]), f2(d["mean_join"]), f2(d["mean_refuse"]))
            for k, v in d.get("joiner_basis", {}).items():
                out[f"{tag}.basis.{k}"] = str(v)
    p = RES / "paper1/instruction/h4_same_prompt.json"
    if p.exists():
        d = json.loads(p.read_text())
        for k, v in d.items():
            out[f"hfour.{k.replace('_', '')}"] = str(v) if k.startswith("n_") else f2(v)
    for j, js in (("any", ""), ("strict", "s"), ("vote", "v")):
        for dn, ds in (("", ""), ("_dn", "n"), ("_dn_bc", "nb")):
            p, tag = RES / f"paper1/instruction/decision_{j}{dn}.json", f"dec{js}{ds}"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            out[f"{tag}.choice.auroc"] = f2(d["choice_heldout_scenarios"]["auc_join_vs_refuse"])
            out[f"{tag}.choice.njoin"] = str(d["choice_heldout_scenarios"]["n_join"])
            out[f"{tag}.choice.nrefuse"] = str(d["choice_heldout_scenarios"]["n_refuse"])
            for cond in ("stripped", "normal"):
                out[f"{tag}.{cond}.vsdec"] = f2(d[cond]["auc_col_vs_decline"])
                out[f"{tag}.{cond}.vsben"] = f2(d[cond]["auc_col_vs_benign"])
    # Option-label mismatch (label_mismatch.py): brief names an option label that is not on the menu (label variants
    # 2/3). mm.* = counts and collusion success by flag; clean.* = the main results without the flagged runs
    p = RES / "paper1/audit/label_mismatch.json"
    if p.exists():
        d = json.loads(p.read_text())
        for mode, tag in (("collusion", "col"), ("briefed_decline", "dec"), ("choice", "choice")):
            out[f"mm.{tag}.n"], out[f"mm.{tag}.total"] = str(d["summary"][mode]["mismatch"]), str(d["summary"][mode]["n"])
        col = [r for r in d["runs"].values() if r["mode"] == "collusion"]
        for flag, ft in ((False, "clean"), (True, "mm")):
            for kt, kf in (("kone", lambda k: k == 1), ("ktwo", lambda k: k >= 2)):
                g = [r for r in col if r["mismatch"] == flag and kf(r["k"])]
                out[f"mm.succ.{ft}.{kt}"], out[f"mm.succ.{ft}.{kt}.n"] = f2(sum(r["success"] for r in g) / len(g)), str(len(g))
    p = RES / "paper1/robustness/label_mismatch.json"
    if p.exists():
        for name, v in json.loads(p.read_text()).items():
            for cond in ("all", "clean"):
                for k, det in (("peak|cummean", "peak"), ("ca_t|cummean", "cat"), ("ca_glr|cummean", "caglr")):
                    m = v[cond][k]
                    t = f"{cond}.{name}.{det}"
                    out[f"{t}.tpr1"], out[f"{t}.auroc"] = f2(m["tpr@0.01"]), f2(m["auroc"])
                    out[f"{t}.aurocvsdec"] = f2(m["auroc_vs_briefed_decline"])
    c = RES / "paper1/instruction/clean"
    if (c / "h5_counterfactual.json").exists():
        h = json.loads((c / "h5_counterfactual.json").read_text())
        for cond in ("normal", "stripped"):
            out[f"clean.hfive.{cond}.vsdec"] = f2(h[cond]["agent_auc_vs_briefed_decline"])
            out[f"clean.hfive.{cond}.vsdec.ci"] = ci(h[cond]["agent_auc_vs_briefed_decline_ci"])
        out["clean.hfive.n"] = str(h["stripped"]["n_collusion"])
    if (c / "choice.json").exists():
        d = json.loads((c / "choice.json").read_text())
        out["clean.hsix.njoin"], out["clean.hsix.nrefuse"] = str(d["n_join"]), str(d["n_refuse"])
        out["clean.hsix.auroc"] = f2(d["auc_join_vs_refuse"])
    if (c / "decision_strict.json").exists():
        d = json.loads((c / "decision_strict.json").read_text())
        out["clean.decs.choice.auroc"] = f2(d["choice_heldout_scenarios"]["auc_join_vs_refuse"])
        for cond in ("stripped", "normal"):
            out[f"clean.decs.{cond}.vsdec"] = f2(d[cond]["auc_col_vs_decline"])
    if (c / "h4_same_prompt.json").exists():
        d = json.loads((c / "h4_same_prompt.json").read_text())
        out["clean.hfour.flagnotacted"], out["clean.hfour.nnotacted"] = f2(d["flag_not_acted"]), str(d["n_not_acted"])
    # benchmark counts (main committee set): successful / failed attempts, successes per wording, episodes with an error
    p = RES / "e2v2b/act.json"
    if p.exists():
        col = [r for r in json.loads(p.read_text())["records"] if r["mode"] == "collusion"]
        out["bench.succ"], out["bench.fail"] = str(sum(r["set"] == "pos" for r in col)), str(sum(r["set"] == "fail" for r in col))
        w = [sum(r["set"] == "pos" and r["template"] == t for r in col) for t in sorted({r["template"] for r in col})]
        out["bench.wmin"], out["bench.wmax"] = str(min(w)), str(max(w))
    p = RES / "e2v2b/error_runs.txt"
    if p.exists():
        out["bench.err"] = str(sum(l.startswith("e2v2__") for l in p.read_text().splitlines()))
    # benchmark audit of the main committee episodes (audit_e2v2.py): share of episodes in which all special agents
    # voted for the target, per family; decliners' private replies that read as refusals
    p = RES / "paper1/labelfix/audit_e2v2.txt"
    if p.exists():
        t = p.read_text()
        for m in re.finditer(r"^(\w+)\s+\d+\s+(\d+)/(\d+) \((\d+)%\)", t, re.M):
            out[f"fam.{m.group(1).replace('_', '')}.tgtpct"] = m.group(4)
        ref = [tuple(map(int, x)) for x in re.findall(r"refusal \(keyword heuristic\): (\d+)/(\d+)", t)]
        out["fam.dec.refusals"] = f"{sum(a for a, _ in ref):,}".replace(",", "{,}")
        out["fam.dec.private"] = f"{sum(b for _, b in ref):,}".replace(",", "{,}")
        m = re.search(r"collusion success by k: (\{.*\})", t)
        if m:
            for k, v in eval(m.group(1)).items():
                out[f"fam.k{k}.succ"], out[f"fam.k{k}.n"] = v.split("/")
    # PREREG amendment 7: episodes whose brief named the wrong option (fair option, or a label not on the menu) before the
    # fix, per model / data set / family (results/paper1/labelfix/scope.json, counted on the set-aside originals)
    p = RES / "paper1/labelfix/scope.json"
    if p.exists():
        tot = 0
        for key, v in json.loads(p.read_text()).items():
            tag, grp, mode = key.split("|")
            t = f"lf.{tag.replace('_', '')}.{grp}.{mode.replace('_', '')}"
            n = v.get("fair", 0) + v.get("offmenu", 0)
            tot += n
            out[t + ".n"], out[t + ".fair"], out[t + ".off"] = str(n), str(v.get("fair", 0)), str(v.get("offmenu", 0))
            out[t + ".total"] = f"{v['total']:,}".replace(",", "{,}")
        out["lf.total"] = f"{tot:,}".replace(",", "{,}")
    # vote rule by family (vote_baseline.py): share of episodes with at least one vote for the target option
    p = RES / "e7/vote_baseline.txt"
    if p.exists():
        blk = p.read_text().split("share voting the target option")[-1]
        for fam, v in re.findall(r"^\s+(\w+)\s+mean [\d.]+\s+share>0 ([\d.]+)", blk, re.M):
            out[f"vb.{fam.replace('_', '')}.anypct"] = f"{100 * float(v):.1f}".replace(".0", "") if float(v) else "0"
        m = re.search(r"recall ([\d.]+)\s+FPR headline ([\d.]+)", blk)
        if m:
            out["vb.recall"], out["vb.fprpct"] = f"{float(m.group(1)):.2f}", f"{100 * float(m.group(2)):.1f}"
    # recall at 1% FPR before the vote by coalition size k (as fig_byk: pooled-negative threshold, all attempts)
    p = RES / "e2v2b/act.json"
    if p.exists():
        recs = [r for r in json.loads(p.read_text())["records"] if not r["rid"].startswith(("e2v2aw", "e2v2an"))]
        neg, pos = [r for r in recs if r["set"] == "neg"], [r for r in recs if r["set"] in ("pos", "fail")]
        for key, det in (("peak|cummean", "peak"), ("ca_glr|cummean", "caglr"), ("ca_t|cummean", "cat"),
                         ("dsplit_k2|cummean", "dsplit")):
            v = sorted(r["prevote"][key] for r in neg)
            tau = v[min(len(v) - 1, int(0.99 * (len(v) - 1) + 0.5))] if v else 0
            for k in range(1, 8):
                g = [r["prevote"][key] > tau for r in pos if r["k"] == k]
                if g:
                    out[f"byk.{det}.k{k}"] = f2(sum(g) / len(g))
    # steering direction (steer_direction.py, layer 18): class gap, spread, residual norm, AUROC of the projection
    p = RES / "steer/dir_L18.txt"
    if p.exists():
        t = p.read_text()
        m = re.search(r"delta ([\d.]+) sigma ([\d.]+) .* mean residual norm ([\d.]+)", t)
        out["steer.dir.delta"], out["steer.dir.sigma"], out["steer.dir.norm"] = m.groups()
        m = re.search(r"AUROC of projection on direction: ([\d.]+); random: ([\d.]+)", t)
        out["steer.dir.auroc"], out["steer.dir.aurocrand"] = f2(float(m.group(1))), f2(float(m.group(2)))
    # monitor-aware slices, from the logs: successes, flag rates at the non-aware threshold (any aggregator), AUROC and
    # recall with the threshold re-set on aware negatives, anytime alarm, coalition Jaccard
    DL = {"peak": "peak", "dsplit_k2": "dsplit", "ca_t": "cat", "ca_glr": "caglr", "mean_all": "mean"}
    for fn, tag in (("aware/act.log", "awl"), ("aware/notice_only_act.log", "ntl"), ("aware/act_tmpl.log", "awtl"),
                    ("aware/core.log", "awcore"), ("aware/text.log", "awtext")):
        p = RES / fn
        if not p.exists():
            continue
        t = p.read_text().split("== MONITOR-AWARE slice")[-1]
        m = re.search(r"'successes': (\d+)", t)
        if m:
            out[f"{tag}.succ"] = m.group(1)
        pre = t.split("final: flag rate")[0]
        for det, grp, a, b in re.findall(r"^\s+(\w+)\s+(attempts|successes)\s+([\d.]+) \| ([\d.]+)", pre, re.M):
            if det in DL:
                out[f"{tag}.{DL[det]}.{grp[:3]}.aware"], out[f"{tag}.{DL[det]}.{grp[:3]}.paired"] = f2(float(a)), f2(float(b))
        for det, grp, au, tp in re.findall(r"^\s+(\w+)\s+aware-only (attempts|successes): AUROC ([\d.]+) \[[^]]*\] TPR@1% ([\d.]+)", pre, re.M):
            if det in DL:
                out[f"{tag}.{DL[det]}.only{grp[:3]}.auroc"], out[f"{tag}.{DL[det]}.only{grp[:3]}.tpr1"] = f2(float(au)), f2(float(tp))
        for det, a, b, fa, fb in re.findall(r"^\s+(\w+)\|last a=0.01: attempts ([\d.]+) \| ([\d.]+)\s+negatives \(realised FPR\) ([\d.]+) \| ([\d.]+)", t, re.M):
            if det in DL:
                k = f"{tag}.{DL[det]}.any1"
                out[k + ".aware"], out[k + ".paired"] = f2(float(a)), f2(float(b))
                out[k + ".fpraware"] = f"{100 * float(fa):.1f}"
        m = re.search(r"coalition Jaccard, aware collusion runs: \{[^}]*'ca_t\|cummean': ([\d.]+)", t)
        if m:
            out[f"{tag}.cat.jacc"] = f2(float(m.group(1)))
    p = RES / "paper1/labelfix/human_check.json"
    if p.exists():
        h = json.loads(p.read_text())
        out["lf.human.items"], out["lf.human.regen"] = str(h["items"]), str(h["regenerated"])
    return out, missing


# PREREG amendment 7: the before/after appendix quotes these keys from the results on the original (option-label error)
# episodes, kept in experiments/results_labelbug/, as lb.<key>
LB_KEYS = ["com.act.peak.pa.tpr1", "com.act.peak.pa.tpr1ci", "com.act.caglr.pa.tpr1", "com.act.cat.pa.tpr1",
           "com.act.peak.pa.auroc", "com.act.peak.pa.fprfam5.briefeddecline", "com.recipe.peak.pa.tpr1",
           "com.recipe.peak.pa.aurocvs.briefeddecline", "com.text.peak.pa.tpr1", "com.judgeq.best.pa.tpr1",
           "com.judgem.best.pa.tpr1", "com.enc.peak.pa.tpr1", "ceil.com.oracle.tpr1", "hfive.stripped.agentbriefeddecline",
           "hfiveqtt.stripped.agentbriefeddecline", "hfivemis.stripped.agentbriefeddecline", "hfour.flagacted",
           "hfour.flagnotacted", "hsix.njoin", "hsix.nrefuse", "hsix.auroc", "decs.normal.vsdec", "decs.stripped.vsdec",
           "com.decnb.peak.pa.tpr1", "nb.main.p1.hon.auroc", "nb.main.p2.hon.auroc", "nb.main.p2.run.imppct",
           "q32.act.peak.pa.tpr1", "mis.act.peak.pa.tpr1", "aw.peak.att1.aware",
           "steer.basecol.success", "steer.ablcol.success", "steer.ablrcol.success", "negx.act.peak.pa.tpr01",
           "mm.col.n", "mm.col.total", "bench.succ", "bench.fail", "bench.wmin", "bench.wmax", "bench.err"]


def main():
    out, missing = collect(RES)
    old = ROOT / "results_labelbug"
    if old.exists():
        prev, _ = collect(old)
        out.update({f"lb.{k}": prev[k] for k in LB_KEYS if k in prev})
    lines = ["% GENERATED by papers/shared/make_numbers.py from experiments/results/ -- do not edit by hand.",
             "\\makeatletter",
             "\\providecommand{\\R}[1]{\\@ifundefined{rnum@#1}{\\textcolor{red}{\\textbf{??#1}}"
             "\\PackageWarning{numbers}{Unknown result key #1}}{\\@nameuse{rnum@#1}}}"]
    for k in sorted(out):
        lines.append(f"\\@namedef{{rnum@{k}}}{{{out[k]}}}")
    lines.append("\\makeatother")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} with {len(out)} numbers" + (f"; missing files: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
