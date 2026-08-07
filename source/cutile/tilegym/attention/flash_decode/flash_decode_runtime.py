import importlib.util
import sys
from pathlib import Path

import tilegym
import torch


SOURCE = Path(__file__).with_name("flash_decode.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load(
    "tilegym.ops.cutile.utils",
    Path(__file__).parents[2] / "support" / "utils.py",
)
load("tilegym.ops.cutile.splitk_reduce", SOURCE.with_name("splitk_reduce.py"))
source = load("tilegym.ops.cutile._local_flash_decode", SOURCE)


def main():
    batch, q_heads, kv_heads, seqlen, head_dim = 8, 32, 8, 8192, 128
    q = torch.randn(batch, q_heads, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, seqlen, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    output = source.fmha_decode(q, k, v, head_dim**-0.5)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.fmha_decode(q, k, v, head_dim**-0.5)
    end.record()
    torch.cuda.synchronize()

    print(f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} dtype={q.dtype}")
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
