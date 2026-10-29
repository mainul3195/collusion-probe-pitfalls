"""Paper 1 no-brief test, step 2: QLoRA fine-tune of the agent model on build_implant_data.py output.

Qwen3-14B in 4-bit NF4 (fits one 16 GB GPU with gradient checkpointing), LoRA on all attention and MLP projections,
loss on the reply tokens only, chat template without thinking (as the agents). The adapter then carries the
collusion goal: given an honest seat's context, it writes what a briefed colluder wrote. Served by vLLM
(--enable-lora) for generation and loaded by extract_e2.py --lora for activations.

Usage:
  python train_implant.py --model Qwen/Qwen3-14B --data runs/qwen3_14b/implant_data/train.jsonl \
      --out runs/qwen3_14b/implant_lora --epochs 1
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CHAT_KW = {"enable_thinking": False}


def encode(tok, e, max_len):
    msgs = [{"role": "system", "content": e["system"]}, {"role": "user", "content": e["user"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CHAT_KW)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": e["response"]}], tokenize=False, **CHAT_KW)
    ids = tok(full).input_ids
    n_prompt = len(tok(prompt).input_ids)
    labels = [-100] * n_prompt + ids[n_prompt:]
    return (ids, labels) if len(ids) <= max_len else None  # truncating would cut the system prompt: skip instead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N examples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    data = [json.loads(x) for x in open(args.data)][:args.limit]
    enc = [x for x in (encode(tok, e, args.max_len) for e in data) if x is not None]
    print(f"{len(enc)} of {len(data)} examples fit {args.max_len} tokens, mean tokens {sum(len(i) for i, _ in enc) / len(enc):.0f}, "
          f"reply tokens {sum(sum(l != -100 for l in lab) for _, lab in enc)}", flush=True)

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=q, device_map={"": 0},
                                                 torch_dtype=torch.bfloat16)
    # not prepare_model_for_kbit_training: it upcasts embed_tokens and lm_head (151,936 x 5,120 each) to fp32, which
    # does not fit 16 GB; they stay frozen in bf16 instead
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(enc) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    order = []
    while len(order) < steps * args.accum:
        idx = list(range(len(enc)))
        random.Random(args.seed + len(order)).shuffle(idx)
        order += idx
    order = order[:steps * args.accum]
    model.train()
    t0, run = time.time(), 0.0
    for s in range(steps):
        for j in order[s * args.accum:(s + 1) * args.accum]:
            ids, lab = enc[j]
            k = sum(l != -100 for l in lab) + 1  # logits only where they predict a reply token (full logits: 2.5 GB)
            logits = model(input_ids=torch.tensor([ids], device=0), logits_to_keep=k).logits[0, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(ids[-(k - 1):], device=0))
            (loss / args.accum).backward()
            run = 0.98 * run + 0.02 * loss.item() if run else loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if s % 10 == 0 or s == steps - 1:
            el = time.time() - t0
            print(f"step {s + 1}/{steps} loss {run:.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"elapsed {el / 60:.1f} min, eta {el / (s + 1) * (steps - s - 1) / 60:.1f} min", flush=True)
    model.save_pretrained(out)
    (out / "train_info.json").write_text(json.dumps(vars(args) | dict(examples=len(enc), steps=steps,
                                                                     final_loss=run, minutes=(time.time() - t0) / 60)))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
