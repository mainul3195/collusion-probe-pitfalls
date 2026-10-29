"""Paper 1 task A3: a strong trained text baseline (fine-tuned ModernBERT) in the same protocol as the probe.

Each agent message is classified on its own text with the probe's labels (active colluder = 1; honest rows and every row
of the headline negative families = 0; independent_bias excluded). Cross-fitting uses the same 25
leave-scenario-and-template-out cells as e2v2_analysis.py: the model that scores cell (f, t) never saw scenario fold f
or briefing template t. Every row is scored exactly once, so the scores can be fed to e2v2_analysis.py as fixed
per-row scores (--signal judge --judge-dir <out>), which applies the same aggregators and calibration.

Output: one json per run, {"scores": {agent: [round1..round5]}} (judge_e2.py format).

Usage:
  python text_encoder_scores.py --v2 runs/qwen3_14b/e2v2_*_c?_act --v1-controls runs/qwen3_14b/e2{sys,ben}_{a,b}_act \
      --out runs/qwen3_14b/encoder_scores [--cells 0:1] [--limit-train 2000]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import e2_analysis as A
from e2v2_analysis import F, NT, labels


def load_rows(v2, v1):
    meta, acts = A.load_e2(v2, ["text"], "text")
    text = list(acts["text"])
    if v1:
        m1, a1 = A.load_e2(v1, ["text"], "text")
        for m, t in zip(m1, a1["text"]):
            if m["mode"] == "control":
                meta.append(m)
                text.append(t)
    return meta, np.array(text, dtype=object)


def cells_of(meta):
    """Identical fold / template assignment to e2v2_analysis.main()."""
    scen = sorted({m["scenario_id"] for m in meta})
    fold_of = {s: i % F for i, s in enumerate(np.random.RandomState(42).permutation(scen))}
    rf = np.array([fold_of[m["scenario_id"]] for m in meta])
    rt = np.array([m["template"] if m.get("template") is not None
                   else (scen.index(m["scenario_id"]) + m["replicate"]) % NT for m in meta])
    return rf, rt


def train_and_score(model_id, texts_tr, y_tr, texts_te, args, dev):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, torch_dtype=torch.float32).to(dev)
    enc = lambda batch: tok(list(batch), truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")  # noqa: E731
    order = np.random.RandomState(0).permutation(len(texts_tr))
    batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(batches)), len(batches))
    w = torch.tensor([1.0, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32, device=dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    model.train()
    for b in batches:
        x = enc(texts_tr[b]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**x).logits
        loss = lossf(logits.float(), torch.tensor(y_tr[b], device=dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts_te), 128):
            x = enc(texts_te[i:i + 128]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**x).logits.float()
            out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out) if out else np.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", nargs="+", required=True)
    ap.add_argument("--v1-controls", nargs="*", default=[])
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-pos", type=int, default=6000)
    ap.add_argument("--cells", default=None, help="debug: a:b slice of the 25 cells")
    ap.add_argument("--limit-train", type=int, default=None, help="debug: cap training rows")
    args = ap.parse_args()
    dev = "cuda"
    meta, text = load_rows(args.v2, args.v1_controls)
    y = labels(meta, "exclude")
    rf, rt = cells_of(meta)
    out = Path(args.out); (out / "cells").mkdir(parents=True, exist_ok=True)
    print(f"rows {len(meta)}  pos {(y == 1).sum()}  neg {(y == 0).sum()}", flush=True)
    cells = [(f, t) for f in range(F) for t in range(NT)]
    if args.cells:
        a, b = map(int, args.cells.split(":")); cells = cells[a:b]
    rng = np.random.RandomState(0)
    for f, t in cells:
        done = out / "cells" / f"{f}_{t}.npy"
        test = np.where((rf == f) & (rt == t))[0]
        if done.is_file():
            continue
        pool = np.where((rf != f) & (rt != t) & (y >= 0))[0]
        pos, neg = pool[y[pool] == 1], pool[y[pool] == 0]
        if len(pos) > args.max_pos:
            pos = rng.choice(pos, args.max_pos, replace=False)
        neg = rng.choice(neg, min(len(neg), 2 * len(pos)), replace=False)
        tr = np.r_[pos, neg]
        if args.limit_train:
            tr = rng.choice(tr, min(len(tr), args.limit_train), replace=False)
        t0 = time.time()
        s = train_and_score(args.model, text[tr], y[tr], text[test], args, dev)
        np.save(done.with_suffix(".tmp.npy"), np.c_[test, s])
        done.with_suffix(".tmp.npy").rename(done)
        print(f"cell {f},{t}: train {len(tr)} test {len(test)} {time.time() - t0:.0f}s", flush=True)
    # assemble per-run score files once every cell is done
    files = sorted((out / "cells").glob("*.npy"))
    if len(files) < F * NT or args.cells:
        print(f"{len(files)}/{F * NT} cells done; not assembling", flush=True)
        return
    score = np.full(len(meta), np.nan)
    for p in files:
        a = np.load(p)
        score[a[:, 0].astype(int)] = a[:, 1]
    per = {}
    for m, s in zip(meta, score):
        r = per.setdefault(m["run_id"], {})
        r.setdefault(m["agent_name"], [math.nan] * 5)[m["round"] - 1] = float(s)
    for rid, sc in per.items():
        (out / f"{rid}.json").write_text(json.dumps({"scores": sc}))
    print(f"wrote {len(per)} run files; nan rows {int(np.isnan(score).sum())}", flush=True)


if __name__ == "__main__":
    main()
