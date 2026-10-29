"""PREREG amendment 7: move the episodes affected by the option-label error out of a generation folder, so that
gen_e2v2.py (which skips finished runs) regenerates exactly those with the fixed briefs; also move the folder's old
activation outputs. Nothing is deleted: runs go to <folder>_labelbug/, activation folders to <name>_labelbug/.

Affected = some special agent's system prompt names an option letter that is off the menu or is not the target
(label_mismatch.py criterion, extended to the fair-option case). Idempotent: rerunning moves nothing new.

Usage: python archive_labelbug.py --folder $THESIS/runs/qwen3_14b/e2v2_col_c0 [--acts e2v2_col_c0_act,...] [--dry]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

OPT = re.compile(r"\bOption ([A-Z0-9])\b")


def affected(run_dir):
    cfg = json.loads((run_dir / "run_config.json").read_text())
    special = {a for a, r in cfg["roles"].items() if r != "honest"}
    named = set()
    for e in json.loads((run_dir / "agent_prompts.json").read_text()):
        if e["agent_name"] in special:
            named |= set(OPT.findall(e["system_prompt"]))
    target, fair = str(cfg["target_option"]), str(cfg["fair_option"])
    return bool(named) and (bool(named - {target, fair}) or target not in named)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--acts", default="", help="comma list of activation folder names (siblings) to move aside")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    folder = Path(args.folder)
    arch = folder.parent / f"{folder.name}_labelbug"
    moved = kept = 0
    for d in sorted(p for p in folder.iterdir() if (p / "run_config.json").exists()):
        if (d / "agent_prompts.json").exists() and affected(d):
            moved += 1
            if not args.dry:
                arch.mkdir(exist_ok=True)
                assert not (arch / d.name).exists(), f"{d.name} already archived"
                shutil.move(str(d), str(arch / d.name))
        else:
            kept += 1
    acts = [a for a in args.acts.split(",") if a]
    for a in acts:
        src, dst = folder.parent / a, folder.parent / f"{a}_labelbug"
        if src.exists() and not dst.exists() and moved and not args.dry:
            shutil.move(str(src), str(dst))
    print(f"{folder}: moved {moved} affected runs to {arch.name}, kept {kept}; activation folders set aside: {acts}"
          + (" (dry run)" if args.dry else ""))


if __name__ == "__main__":
    main()
