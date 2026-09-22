import importlib.util
import math
from pathlib import Path

import torch


def _runtime():
    path = Path(__file__).with_name("runtime.py")
    spec = importlib.util.spec_from_file_location("intent_cutile_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(runtime, case, source_path):
    flags = {
        "geglu": {"needs_gelu": True},
        "swiglu": {"needs_utils": True},
        "attention_sink_decode": {"needs_utils": True, "needs_splitk": True},
        "gemma_decode": {"needs_utils": True, "needs_splitk": True},
        "mla_decode_split": {"needs_utils": True, "needs_splitk": True},
        "recurrent_gated_delta": {"needs_utils": True},
        "rms_norm": {"needs_utils": True},
    }.get(case, {})
    module_name = (
        f"tilegym.ops.cutile.activation._intent_{case}"
        if case == "geglu"
        else f"tilegym.ops.cutile._intent_{case}"
    )
    return runtime.load_source(source_path, module_name, **flags)


def run(case: str, source_path: Path):
    runtime = _runtime()
    source = _load(runtime, case, source_path)

    if case == "gelu":
        x = torch.randn((8192, 4096), device="cuda", dtype=torch.float16)
        call = lambda: source.gelu(x, approximate="tanh")
        detail = f"x={tuple(x.shape)} dtype={x.dtype} approximate=tanh"
    elif case == "geglu":
        x = torch.randn((4096, 28672), device="cuda", dtype=torch.float16)
        call = lambda: source.geglu(x, dim=-1, approximate="tanh")
        detail = f"x={tuple(x.shape)} dtype={x.dtype}"
    elif case == "relu":
        x = torch.randn((8192, 4096), device="cuda", dtype=torch.float16)
        call = lambda: source.relu(x)
        detail = f"x={tuple(x.shape)} dtype={x.dtype}"
    elif case == "swiglu":
        a = torch.randn((8192, 14336), device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        call = lambda: source.swiglu(a, b)
        detail = f"a/b={tuple(a.shape)} dtype={a.dtype}"
    elif case == "dropout":
        x = torch.randn((8192, 4096), device="cuda", dtype=torch.float16)
        call = lambda: source.dropout(x, 12345, p=0.1, training=True, inplace=False)
        detail = f"x={tuple(x.shape)} dtype={x.dtype} p=0.1"
    elif case == "attention_sink":
        batch, sequence, kv_heads, repeats, dim = 1, 4096, 8, 4, 128
        q = torch.randn((batch, sequence, kv_heads, repeats, dim), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((batch, sequence, kv_heads, dim), device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        sinks = torch.randn((kv_heads * repeats,), device="cuda", dtype=q.dtype)
        start = torch.tensor([0], device="cuda", dtype=torch.int32)
        call = lambda: source.attention_sink(q, k, v, sinks, 1.0 / math.sqrt(dim), None, start)
        detail = f"Q={tuple(q.shape)} K/V={tuple(k.shape)} dtype={q.dtype}"
    elif case == "attention_sink_decode":
        batch, sequence, kv_heads, repeats, dim = 32, 8192, 8, 4, 128
        q = torch.randn((batch, 1, kv_heads, repeats, dim), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((batch, sequence, kv_heads, dim), device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        sinks = torch.randn((kv_heads * repeats,), device="cuda", dtype=q.dtype)
        start = torch.tensor([sequence - 1], device="cuda", dtype=torch.int32)
        call = lambda: source.attention_sink_decode(q, k, v, sinks, 1.0 / math.sqrt(dim), start_q=start, kv_len_per_split=256)
        detail = f"Q={tuple(q.shape)} K/V={tuple(k.shape)} dtype={q.dtype}"
    elif case == "gemma_prefill":
        batch, sequence, heads, kv_heads, dim = 2, 4096, 32, 8, 128
        q = torch.randn((batch, heads, sequence, dim), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((batch, kv_heads, sequence, dim), device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        call = lambda: source.gemma_attention_cutile(
            q, k, v, window_size=1024, soft_cap=50.0, is_causal=True, use_autotune=False
        )
        detail = f"Q={tuple(q.shape)} K/V={tuple(k.shape)} window=1024 soft_cap=50 dtype={q.dtype}"
    elif case == "gemma_decode":
        batch, sequence, heads, kv_heads, dim = 32, 8192, 32, 8, 128
        q = torch.randn((batch, heads, 1, dim), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((batch, kv_heads, sequence, dim), device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        call = lambda: source.gemma_fmha_decode(q, k, v, window_size=1024, soft_cap=50.0, kv_len_per_split=256)
        detail = f"Q={tuple(q.shape)} K/V={tuple(k.shape)} window=1024 soft_cap=50 dtype={q.dtype}"
    elif case == "mla_decode":
        batch, heads, sequence, dim, pe = 8, 64, 8192, 512, 64
        q = torch.randn((batch, heads, dim), device="cuda", dtype=torch.float16)
        qpe = torch.randn((batch, heads, pe), device="cuda", dtype=q.dtype)
        kv = torch.randn((batch, sequence, dim), device="cuda", dtype=q.dtype)
        kpe = torch.randn((batch, sequence, pe), device="cuda", dtype=q.dtype)
        call = lambda: source.mla_decoding(q, qpe, kv, kpe, 1.0 / math.sqrt(dim + pe))
        detail = f"Q={tuple(q.shape)} KV={tuple(kv.shape)} dtype={q.dtype}"
    elif case == "mla_decode_split":
        batch, heads, sequence, dim, pe = 8, 64, 8192, 512, 64
        q = torch.randn((batch, heads, dim), device="cuda", dtype=torch.float16)
        qpe = torch.randn((batch, heads, pe), device="cuda", dtype=q.dtype)
        kv = torch.randn((batch, sequence, dim), device="cuda", dtype=q.dtype)
        kpe = torch.randn((batch, sequence, pe), device="cuda", dtype=q.dtype)
        call = lambda: source.mla_decoding_split_kv(q, qpe, kv, kpe, 1.0 / math.sqrt(dim + pe), 512)
        detail = f"Q={tuple(q.shape)} KV={tuple(kv.shape)} split=512 dtype={q.dtype}"
    elif case == "sparse_mla":
        batch, heads, sequence, kv_sequence, dim, pe, kv_heads, topk = 1, 64, 2048, 4096, 128, 64, 1, 512
        q = torch.randn((batch, heads, sequence, dim), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((batch, kv_heads, kv_sequence, dim), device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        qpe = torch.randn((batch, heads, sequence, pe), device="cuda", dtype=q.dtype)
        kpe = torch.randn((batch, 1, kv_sequence, pe), device="cuda", dtype=q.dtype)
        indices = torch.randint(0, kv_sequence, (batch, sequence, kv_heads, topk), device="cuda", dtype=torch.int32)
        call = lambda: source.tile_sparse_mla(q, k, v, indices, qpe, kpe, is_causal=True, scaling=1.0 / math.sqrt(dim + pe), kernel_configs={"TILE_H": 1, "TILE_N": 64})
        detail = f"Q={tuple(q.shape)} KV={tuple(k.shape)} topk={topk} dtype={q.dtype}"
    elif case == "swa_attention":
        batch, heads, kv_heads, sequence, dim = 2, 32, 8, 4096, 128
        q = torch.randn((batch, heads, sequence, dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, kv_heads, sequence, dim), device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        call = lambda: source.tile_swa_attention(q, k, v, window_size=1024, is_causal=True)
        detail = f"Q={tuple(q.shape)} K/V={tuple(k.shape)} window=1024 dtype={q.dtype}"
    elif case in {"chunk_gated_delta", "recurrent_gated_delta"}:
        batch, sequence, heads, key_dim, value_dim = 2, 2048, 8, 128, 128
        q = torch.randn((batch, sequence, heads, key_dim), device="cuda", dtype=torch.bfloat16) * 0.1
        k = torch.randn_like(q)
        v = torch.randn((batch, sequence, heads, value_dim), device="cuda", dtype=q.dtype) * 0.1
        g = -torch.rand((batch, sequence, heads), device="cuda", dtype=q.dtype) * 0.5
        beta = torch.sigmoid(torch.randn_like(g))
        if case == "chunk_gated_delta":
            call = lambda: source.chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=64, output_final_state=True)
        else:
            call = lambda: source.recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
        detail = f"Q/K={tuple(q.shape)} V={tuple(v.shape)} dtype={q.dtype}"
    elif case == "rms_norm":
        x = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((4096,), device="cuda", dtype=x.dtype)
        call = lambda: source.rms_norm(x, (4096,), weight, 1e-5, mode="multi_wave_reload")
        detail = f"x={tuple(x.shape)} dtype={x.dtype}"
    elif case == "nvfp4_quantize":
        x = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16)
        call = lambda: source.tile_nvfp4_quantize(x, s_enc=1.0)
        detail = f"x={tuple(x.shape)} dtype={x.dtype}"
    elif case == "mhc_gemm_rms":
        tokens, hidden, streams = 2048, 4096, 4
        x = torch.randn((tokens, streams * hidden), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((streams * hidden, streams * (streams + 2)), device="cuda", dtype=torch.bfloat16)
        bias = torch.randn((streams * (streams + 2),), device="cuda", dtype=torch.bfloat16)
        cfg = {"TILE_SIZE_M": 64, "TILE_SIZE_N": 32, "TILE_SIZE_K": 64, "SPLIT_K": 4, "GROUP_SIZE_M": 8}
        call = lambda: source.mhc_gemm_rms_scale(x, weight, streams, 1.0, 1.0, 1.0, bias, cfg=cfg)
        detail = f"tokens={tokens} hidden={hidden} streams={streams} dtype={x.dtype}"
    elif case == "mhc_apply_residual":
        tokens, hidden, streams = 2048, 4096, 4
        x = torch.randn((tokens, streams * hidden), device="cuda", dtype=torch.bfloat16)
        f_out = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
        mixes = torch.randn((tokens, streams * (streams + 2)), device="cuda", dtype=torch.bfloat16)
        call = lambda: source.mhc_apply_residual(x, f_out, mixes, streams)
        detail = f"tokens={tokens} hidden={hidden} streams={streams} dtype={x.dtype}"
    elif case == "mhc_sinkhorn":
        tokens, streams = 8192, 4
        mixes = torch.randn((tokens, streams * (streams + 2)), device="cuda", dtype=torch.float32)
        call = lambda: source.mhc_sinkhorn(mixes, streams)
        detail = f"tokens={tokens} streams={streams} dtype={mixes.dtype}"
    else:
        raise ValueError(case)

    result, latency = runtime.elapsed_ms(call)
    print(f"algorithm={case} {detail}")
    print(runtime.describe(result))
    print(f"latency_ms={latency:.3f}")
