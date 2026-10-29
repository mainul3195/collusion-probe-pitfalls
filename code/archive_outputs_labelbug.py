"""PREREG amendment 7: move per-run outputs (stance labels, LLM-judge scores) of the episodes that archive_labelbug.py
set aside, so the resumable labellers and judges redo exactly those runs. Run ids are kept by the regenerated
episodes, hence the old files must go first. Nothing is deleted: <dir>/<run_id>.json -> <dir>_labelbug/<run_id>.json.

Usage: python archive_outputs_labelbug.py --runs $THESIS/runs/qwen3_14b --dirs stance_mistral24b,judge_v2_logprob,...
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--dirs", required=True)
    args = ap.parse_args()
    root = Path(args.runs)
    ids = {d.name for a in root.glob("*_labelbug") if a.is_dir() for d in a.iterdir() if d.is_dir()}
    print(f"{len(ids)} archived run ids under {root}")
    for name in args.dirs.split(","):
        src, dst = root / name, root / f"{name}_labelbug"
        moved = 0
        for rid in sorted(ids):
            f = src / f"{rid}.json"
            if f.exists():
                dst.mkdir(exist_ok=True)
                shutil.move(str(f), str(dst / f.name))
                moved += 1
        print(f"{name}: moved {moved} files to {dst.name}")


if __name__ == "__main__":
    main()
