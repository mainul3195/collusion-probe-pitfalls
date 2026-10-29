"""Apply thesis-local patches to the NARCBench checkout (idempotent).

Patch 1: extraction scripts load non-quantised models in fp32 by default
(transformers default dtype), which OOMs a 4B model on a 16 GB card. Load in bf16.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "narcbench")
OLD = 'print(f"Loading model: {model_name} (device_map=auto)")'
NEW_LINE = 'kwargs["torch_dtype"] = torch.bfloat16  # THESIS PATCH: avoid fp32 default (OOM on 16 GB)'

# extract_transfer.py has a different generic (non-quantised) load path.
OLD_T = ('print(f"Loading model: {model_name} (AutoModelForCausalLM, device_map=auto)")\n'
         '        model = AutoModelForCausalLM.from_pretrained(\n'
         '            model_name,\n')
NEW_T = OLD_T + '            torch_dtype=torch.bfloat16,  # THESIS PATCH: avoid fp32 default (OOM on 16 GB)\n'

for name in ["extract_core.py", "extract_transfer.py", "extract_stego.py", "extract_token_traces.py"]:
    p = ROOT / "extraction" / name
    s = p.read_text()
    if "THESIS PATCH" in s:
        print("already patched", p)
        continue
    if OLD_T in s:
        p.write_text(s.replace(OLD_T, NEW_T, 1))
        print("patched", p)
        continue
    if OLD not in s:
        print("pattern not found", p)
        continue
    i = s.index(OLD)
    indent = s[s.rfind("\n", 0, i) + 1:i]
    s = s.replace(OLD, OLD + "\n" + indent + NEW_LINE, 1)
    p.write_text(s)
    print("patched", p)
