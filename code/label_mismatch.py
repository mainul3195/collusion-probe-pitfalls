"""Find episodes whose brief names an option label that is not on the committee's menu.

NARCBench scenario texts (motive, instruction) name the options by letter ("Option B"). gen_e2v2.py renames the menu to
1/2 in label variants 2 and 3 but leaves the scenario text unchanged, so a special agent's system prompt can point to an
option that does not exist. An episode is flagged when any special (non-honest) agent's system prompt names an
"Option X" with X not in {target_option, fair_option}. Flags are per run, so every family is covered (collusion,
decline, benign, free choice, ...); robustness analyses drop the flagged runs.

Usage:
  python label_mismatch.py --runs ROOT --out results/paper1/audit/label_mismatch.json [--glob 'e2v2_*_c?' ...]
"""
import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

OPT = re.compile(r"\bOption ([A-Z0-9])\b")


def scan(run_dir):
    cfg = json.loads((run_dir / "run_config.json").read_text())
    menu = {str(cfg["target_option"]), str(cfg["fair_option"])}
    special = {a for a, r in cfg["roles"].items() if r != "honest"}
    named = set()
    for e in json.loads((run_dir / "agent_prompts.json").read_text()):
        if e["agent_name"] in special:
            named |= set(OPT.findall(e["system_prompt"]))
    return dict(mode=cfg["mode"], variant=cfg.get("label_variant"), k=len(special),
                success=bool(cfg.get("collusion_success")), named=sorted(named), mismatch=bool(named - menu),
                target_named=str(cfg["target_option"]) in named)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--glob", nargs="*", default=["e2v2_*_c?", "e2v2x_*", "choice_c?", "benchoice_c?"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dirs = sorted({d for g in args.glob for d in glob.glob(f"{args.runs}/{g}")
                   if not d.endswith(("_act", "_strip", "_final"))})
    res = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*/run_config.json")):
            res[f.parent.name] = scan(f.parent)
    by_mode = defaultdict(Counter)
    for r in res.values():
        by_mode[r["mode"]]["n"] += 1
        by_mode[r["mode"]]["mismatch"] += r["mismatch"]
        by_mode[r["mode"]][f"mismatch_v{r['variant']}"] += r["mismatch"]
    summary = {m: dict(c) for m, c in sorted(by_mode.items())}
    o = Path(args.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(dict(summary=summary, runs=res), indent=1))
    lines = [f"label mismatch over {len(res)} runs ({', '.join(args.glob)})"]
    lines += [f"  {m:20s} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) for m, c in summary.items()]
    o.with_suffix(".log").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
