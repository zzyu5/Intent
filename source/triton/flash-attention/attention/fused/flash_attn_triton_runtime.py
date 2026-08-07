import linecache
import types
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("flash_attn_triton.py")
source_text = SOURCE.read_text()
rewrites = (
    ("tl.dot(q, k, trans_b=True)", "tl.dot(q, tl.trans(k))", 2),
    ("tl.dot(p.to(do.dtype), do, trans_a=True)", "tl.dot(tl.trans(p.to(do.dtype)), do)", 1),
    ("tl.dot(do, v, trans_b=True)", "tl.dot(do, tl.trans(v))", 1),
    ("tl.dot(ds, q, trans_a=True)", "tl.dot(tl.trans(ds), q)", 1),
)
for old, new, expected in rewrites:
    if source_text.count(old) != expected:
        raise RuntimeError(f"unexpected FlashAttention source for runtime compatibility: {old}")
    source_text = source_text.replace(old, new)

compat_filename = f"{SOURCE}.triton3_compat"
linecache.cache[compat_filename] = (
    len(source_text),
    None,
    source_text.splitlines(keepends=True),
    compat_filename,
)
source = types.ModuleType("local_flash_attn_triton")
source.__file__ = compat_filename
exec(compile(source_text, compat_filename, "exec"), source.__dict__)


def main():
    batch, seqlen, heads, head_dim = 4, 4096, 32, 128
    q = torch.randn(batch, seqlen, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output, _, _ = source._flash_attn_forward(q, k, v, causal=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output, _, _ = source._flash_attn_forward(q, k, v, causal=True)
    end.record()
    torch.cuda.synchronize()

    print(f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} dtype={q.dtype}")
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
