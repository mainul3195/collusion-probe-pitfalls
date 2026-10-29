"""Extract per-call hidden states for E2 runs (output of gen_e2.py).

For every model call (agent x phase x round) we run prompt+response through the
model and store, at each requested layer:
  layer_<L> : last-token hidden state (identical convention to NARCBench --gen-only); L = output of decoder block L
              (0-based); L = number of blocks (40 for Qwen3-14B) = output of the final norm, i.e. the last entry
              of transformers' hidden_states (Paper 1 amendment 6, AcMAS-style check)
  mean_<L>  : mean over the response tokens (extra feature for later ablations)
Output: <out>/activations_gen.npz + metadata_gen.json (NARCBench-compatible fields
plus n, k, onset, active, coalition_size, replicate).

--strip context (Paper 1 H5, counterfactual extraction): before the forward pass, remove the private briefing from the
briefed agents' system prompt and the PRIVATE COORDINATION block from their public-round prompts, so the model reads
each message in exactly the context an honest agent would have. The response text is unchanged. If the probe still
separates colluders from decliners, the signal is in the response, not in the instruction. Output metadata carries
strip="context".

--lora PATH (Paper 1 no-brief test): load a LoRA adapter (train_implant.py) on top of the model and read the rows of
the --lora-roles seats (default: implanted) with the adapter on, every other row with it off, so each agent is read
by the weights that wrote its messages. --lora-roles all: every row with the adapter on (weights-only control).
Output metadata carries lora=true/false per row.

Usage:
  python extract_e2.py --model Qwen/Qwen3-14B --quantise 4bit --layers 10-34 \
      --runs-dir runs/qwen3_14b/e2raw --out runs/qwen3_14b/e2act
"""
import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import parse_layers

CHAT_KW = {"enable_thinking": False}
PRIV_START = "PRIVATE COORDINATION (only you and"
_BANKS = {}


def scenario_of(cfg):
    bank = cfg.get("bank", "narcbench")
    if bank not in _BANKS:
        if bank == "codereview":
            from scenarios_codereview import SCENARIOS
        else:
            from gen_e2 import C
            SCENARIOS = C.SCENARIOS
        _BANKS[bank] = {sc["id"]: sc for sc in SCENARIOS}
    return _BANKS[bank][cfg["scenario_id"]]


def strip_context(cfg, row):
    """System prompt and user prompt as an honest agent in the same seat would see them (response unchanged)."""
    from gen_e2 import C
    sp, up = row["system_prompt"], row["user_prompt"]
    if not row["active"]:
        return sp, up
    sc = scenario_of(cfg)
    bases = [C.HONEST_SYSTEM.format(agent_name=row["agent_name"], honest_mandate=sc[k])
             for k in ("honest_mandate", "aggressive_mandate") if k in sc]
    base = next((b for b in bases if sp.startswith(b)), None)
    if base is None:
        raise ValueError(f"no honest base prompt matches {cfg['run_id']} {row['agent_name']}")
    if PRIV_START in up and row["channel"] == "public":
        i = up.index(PRIV_START)
        up = up[:i] + up[up.index("DISCUSSION HISTORY:", i):]
    return base, up


def load(model_id, quantise):
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(device_map="auto", torch_dtype=torch.bfloat16)
    if quantise == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    return model, tok


def shard_ok(shard_dir, rid):
    """A shard counts only if both files exist and are non-empty (a power cut can leave 0-byte files)."""
    return all((shard_dir / f"{rid}.{ext}").is_file() and (shard_dir / f"{rid}.{ext}").stat().st_size > 0
               for ext in ("npz", "json"))


def commit(tmp, final):
    """fsync, then rename: the final name never points at data that is not on disk yet."""
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.rename(final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantise", default=None)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N calls")
    ap.add_argument("--strip", choices=["none", "context"], default="none",
                    help="context: counterfactual extraction without the briefing and the private block (H5)")
    ap.add_argument("--modes", default=None, help="comma list: only runs of these modes")
    ap.add_argument("--lora", default=None, help="LoRA adapter dir (train_implant.py)")
    ap.add_argument("--lora-roles", default="implanted", help="comma list of roles read with the adapter on, or all")
    args = ap.parse_args()
    layers = parse_layers(args.layers)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    shard_dir = out / "shards"; shard_dir.mkdir(exist_ok=True)
    runs = []
    for rd in sorted(Path(args.runs_dir).iterdir()):
        if not (rd / "agent_prompts.json").exists():
            continue
        cfg = json.loads((rd / "run_config.json").read_text())
        if args.modes and cfg["mode"] not in args.modes.split(","):
            continue
        runs.append((cfg, json.loads((rd / "agent_prompts.json").read_text())))
    if args.limit:
        runs = runs[:args.limit]
    todo = [(c, rows) for c, rows in runs if not shard_ok(shard_dir, c["run_id"])]
    print(f"{len(runs)} runs, {len(runs) - len(todo)} already extracted (shards), {len(todo)} to do", flush=True)

    if todo:
        model, tok = load(args.model, args.quantise)
        if args.lora:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.lora).eval()
        trunk = (model.get_base_model() if args.lora else model).model  # decoder stack, no LM head
        blocks = trunk.layers
        lora_on = lambda row: bool(args.lora) and (args.lora_roles == "all" or row["role"] in args.lora_roles.split(","))  # noqa: E731
        cap = {}
        hooks = [(blocks[L] if L < len(blocks) else trunk.norm).register_forward_hook(
            lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0].float()))
            for L in layers]
        try:
            for cfg, rows in tqdm(todo, mininterval=30):
                acts = {f"layer_{L}": [] for L in layers} | {f"mean_{L}": [] for L in layers}
                meta = []
                for row in rows:
                    sp, up = strip_context(cfg, row) if args.strip == "context" else (row["system_prompt"], row["user_prompt"])
                    msgs = [{"role": "system", "content": sp}, {"role": "user", "content": up}]
                    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CHAT_KW)
                    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": row["response"]}],
                                                   tokenize=False, **CHAT_KW)
                    ids = tok(full, return_tensors="pt").input_ids[:, -args.max_tokens:].to(model.device)
                    n_prompt = min(len(tok(prompt).input_ids), ids.shape[1] - 1)
                    on = lora_on(row)
                    with torch.no_grad(), (contextlib.nullcontext() if on or not args.lora else model.disable_adapter()):
                        trunk(ids)  # base model only: no LM head needed
                    for L in layers:
                        h = cap[L]
                        acts[f"layer_{L}"].append(h[-1].cpu().numpy())
                        acts[f"mean_{L}"].append(h[n_prompt:].mean(0).cpu().numpy())
                    meta.append(dict(
                        run_id=cfg["run_id"], mode=cfg["mode"], scenario_id=cfg["scenario_id"],
                        scenario_domain=cfg["scenario_domain"], replicate=cfg["replicate"],
                        inject=cfg.get("inject", "user_top"), version=cfg.get("version", "v1"), aware=cfg.get("aware", False),
                        template=cfg.get("template"), heldout_template=cfg.get("heldout_template", False),
                        agent_name=row["agent_name"], role=row["role"], phase=row["phase"],
                        channel=row["channel"], round=row["round"], active=row["active"],
                        n=cfg["n"], k=cfg["k"], onset=cfg["onset"], coalition=cfg["coalition"],
                        collusion_success=cfg["collusion_success"], target_majority=cfg["target_majority"],
                        target_option=cfg["target_option"], agent_vote=cfg["votes"].get(row["agent_name"]),
                        n_tokens=int(ids.shape[1]), strip=args.strip, lora=on))
                # checkpoint: one shard per run, written atomically (npz first, json marks completion)
                rid = cfg["run_id"]
                np.savez(shard_dir / f"{rid}.tmp.npz", **{k: np.stack(v).astype(np.float16) for k, v in acts.items()})
                commit(shard_dir / f"{rid}.tmp.npz", shard_dir / f"{rid}.npz")
                (shard_dir / f"{rid}.json.tmp").write_text(json.dumps(meta))
                commit(shard_dir / f"{rid}.json.tmp", shard_dir / f"{rid}.json")
        finally:
            for h in hooks:
                h.remove()

    # merge shards (sorted by run_id) into one NARCBench-style file
    acts, meta = {}, []
    for cfg, _ in runs:
        rid = cfg["run_id"]
        z = np.load(shard_dir / f"{rid}.npz")
        for k in z.files:
            acts.setdefault(k, []).append(z[k])
        meta += json.loads((shard_dir / f"{rid}.json").read_text())
    np.savez_compressed(out / "activations_gen.tmp.npz", **{k: np.concatenate(v) for k, v in acts.items()})
    (out / "metadata_gen.json").write_text(json.dumps(meta))
    commit(out / "activations_gen.tmp.npz", out / "activations_gen.npz")  # final file = step complete
    print("saved", out, len(meta), flush=True)


if __name__ == "__main__":
    main()
