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
STATE = {}


def upstream(arguments):
    q, k, v, bias, scale = arguments
    batch, query_length, heads, head_dimension = q.shape
    key_length = k.shape[1]
    if k.shape != (batch, key_length, heads, head_dimension):
        raise RuntimeError("FlashAttention baseline received an incompatible key shape")
    if v.shape != k.shape:
        raise RuntimeError("FlashAttention baseline received an incompatible value shape")
    if bias.shape[2:] != (1, key_length):
        raise NotImplementedError(
            "FlashAttention baseline adapter currently requires vector bias"
        )
    expanded_bias = bias.expand(batch, heads, query_length, key_length)
    rounded_query = source.math.ceil(query_length / 128) * 128
    key = (tuple(q.shape), tuple(k.shape), q.dtype, q.device)
    if key not in STATE:
        STATE[key] = (
            torch.empty_like(q),
            torch.empty(
                (batch, heads, rounded_query),
                device=q.device,
                dtype=torch.float32,
            ),
            torch.empty(
                (batch, heads, rounded_query),
                device=q.device,
                dtype=torch.float32,
            ),
        )
    output, lse, temporary = STATE[key]
    block_head = max(source.triton.next_power_of_2(head_dimension), 16)
    grid = (source.triton.cdiv(query_length, 128), batch * heads)
    source._fwd_kernel[grid](
        q,
        k,
        v,
        expanded_bias,
        output,
        lse,
        temporary,
        scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        expanded_bias.stride(0),
        expanded_bias.stride(1),
        expanded_bias.stride(2),
        output.stride(0),
        output.stride(2),
        output.stride(1),
        heads,
        query_length,
        key_length,
        rounded_query,
        head_dimension,
        query_length // 32,
        key_length // 32,
        "vector",
        False,
        block_head,
        BLOCK_M=128,
        BLOCK_N=128,
        num_warps=4 if head_dimension <= 64 else 8,
        num_stages=1,
    )
    return output.transpose(1, 2)


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
