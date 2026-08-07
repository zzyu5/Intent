import importlib.util
import sys
from pathlib import Path

import tilegym
import torch


SOURCE = Path(__file__).with_name("attention.py")
UTILS = Path(__file__).parents[2] / "support" / "utils.py"
utils_spec = importlib.util.spec_from_file_location("tilegym.ops.cutile.utils", UTILS)
utils = importlib.util.module_from_spec(utils_spec)
sys.modules[utils_spec.name] = utils
utils_spec.loader.exec_module(utils)
spec = importlib.util.spec_from_file_location("tilegym.ops.cutile._local_attention", SOURCE)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def main():
    batch, q_heads, kv_heads, seqlen, head_dim = 2, 32, 8, 4096, 128
    q = torch.randn(batch, q_heads, seqlen, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, seqlen, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    output = source.tile_fmha(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.tile_fmha(q, k, v, is_causal=True)
    end.record()
    torch.cuda.synchronize()

    print(f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} dtype={q.dtype}")
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
