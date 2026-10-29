"""In-process vLLM engine with residual-stream steering, for gen_e2v2.py --steer.

vLLM's HTTP server cannot change activations, so the steered episodes run against an offline vLLM engine (V0,
eager mode, so Python hooks fire on every forward pass). Requests from the episode threads are batched by a worker
thread; steered and unsteered requests go in separate generate() calls, because the hook acts on the whole batch.

Residual stream in vLLM's Qwen3 decoder: each layer returns (hidden, residual) and the stream after the layer is
hidden + residual (the next layer's fused add-norm sums them). So:
  ablate:  residual <- residual - ((hidden + residual) . d) d     (removes the d component of the stream)
  add:     residual <- residual + alpha * d                        (shifts the stream along d)
"""
import os
import queue
import threading
from concurrent.futures import Future

os.environ.setdefault("VLLM_USE_V1", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402


class SteerEngine:
    def __init__(self, model, mode, direction, layers, alpha=0.0, max_batch=32, gpu_mem=0.85, max_model_len=8192,
                 temperature=0.7, max_tokens=512):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, enforce_eager=True, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
                       max_num_seqs=max_batch, disable_log_stats=True)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        self.mode, self.alpha, self.max_batch, self.max_model_len = mode, float(alpha), max_batch, max_model_len
        net = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        p = next(net.parameters())
        self.d = torch.tensor(np.asarray(direction, dtype=np.float32), device=p.device)
        self.active = False
        self.hooks = [net.model.layers[L].register_forward_hook(self._hook) for L in layers]
        self.q = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def _hook(self, module, inputs, output):
        if not self.active or self.mode == "none":
            return None
        hidden, residual = output
        d = self.d.to(residual.dtype)
        if self.mode == "ablate":
            coef = ((hidden.float() + residual.float()) @ self.d).to(residual.dtype)
            residual = residual - coef[:, None] * d[None, :]
        elif self.mode == "add":
            residual = residual + (self.alpha * d)[None, :]
        return hidden, residual

    def prompt(self, system, user):
        return self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def submit(self, system, user, steer):
        """Called from episode threads; blocks until the text is generated."""
        f = Future()
        self.q.put((self.prompt(system, user), bool(steer), f))
        return f.result()

    def _loop(self):
        while True:
            batch = [self.q.get()]
            try:
                while len(batch) < 4 * self.max_batch:
                    batch.append(self.q.get(timeout=0.5))
            except queue.Empty:
                pass
            for flag in (False, True):
                group = [b for b in batch if b[1] == flag]
                if not group:
                    continue
                ok = [b for b in group if len(self.tok(b[0]).input_ids) + self.sp.max_tokens <= self.max_model_len]
                for b in group:
                    if b not in ok:
                        b[2].set_exception(RuntimeError("prompt longer than the context window"))
                if not ok:
                    continue
                self.active = flag
                try:
                    outs = self.llm.generate([b[0] for b in ok], self.sp, use_tqdm=False)
                    for b, o in zip(ok, outs):
                        b[2].set_result(o.outputs[0].text.strip())
                except Exception as e:  # noqa: BLE001
                    for b in ok:
                        b[2].set_exception(e)
                finally:
                    self.active = False
