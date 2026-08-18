import importlib.util
import itertools
from pathlib import Path

import torch


def _runtime():
    spec = importlib.util.spec_from_file_location("intent_tilelang_runtime", Path(__file__).with_name("runtime.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(runtime, case, source_path):
    aliases = []
    if case == "native_sparse_forward":
        aliases.append(("reference", source_path.with_name("reference.py")))
    if case in {"sparse_mla_backward", "fp8_lighting_indexer"}:
        aliases.append(("utils", runtime.ROOT / "attention" / "support" / "deepseek_v32_utils.py"))
    return runtime.load_source(source_path, f"intent_tilelang_{case}", aliases=aliases)


def run(case: str, source_path: Path):
    runtime = _runtime()
    source = _load(runtime, case, source_path)

    if case == "attention_sink_backward":
        latency = source.run_regression_perf(BATCH=2, H=16, N_CTX=2048, D_HEAD=128, dtype="float16")
        print("algorithm=attention_sink_backward B=2 H=16 S=2048 D=128 dtype=float16")
        print(f"latency_ms={latency:.3f}")
        return
    if case == "sparse_mla_backward":
        latency = source.run_regression_perf(B=1, S=4096, SKV=8192, H=64, HKV=1, DQKV=576, DV=512, topk=2048)
        print("algorithm=sparse_mla_backward B=1 S=4096 SKV=8192 H=64 topk=2048 dtype=bf16")
        print(f"latency_ms={latency:.3f}")
        return
    if case == "fp8_lighting_indexer":
        latency = source.run_regression_perf(S=4096, SKV=8192, H=32, HKV=1, D=64)
        print("algorithm=fp8_lighting_indexer S=4096 SKV=8192 H=32 D=64")
        print(f"latency_ms={latency:.3f}")
        return
    if case == "per_token_fp8":
        latency = source.run_regression_perf(M=8192, N=8192, blk_m=8)
        print("algorithm=per_token_fp8 M=8192 N=8192 group=128")
        print(f"latency_ms={latency:.3f}")
        return
    if case == "dequant_bf16_fp4":
        latency = source.run_regression_perf(m=4096, n=4096, k=4096, fast_dequant=True)
        print("algorithm=dequant_bf16_fp4 M=N=K=4096")
        print(f"latency_ms={latency:.3f}")
        return
    if case == "mhc_post":
        latency = source.run_regression_perf(n=4096, h=2560, hc_mult=4)
        print("algorithm=mhc_post tokens=4096 hidden=2560 hc=4")
        print(f"latency_ms={latency:.3f}")
        return
    if case == "mhc_pre":
        latency = source.run_regression_perf(n=2048, hidden_size=4096, hc_mult=4)
        print("algorithm=mhc_pre tokens=2048 hidden=4096 hc=4")
        print(f"latency_ms={latency:.3f}")
        return
    if case == "fp8_2xacc":
        print("algorithm=fp8_2xacc M=N=K=4096 dtype=fp8 output=bf16")
        source.assert_tl_gemm_correctness(4096, 4096, 4096, 128, source.T.float8_e4m3fn, source.T.bfloat16, source.T.float32)
        return

    if case == "block_causal":
        batch, sequence, heads, dim = 2, 4096, 16, 128
        q = torch.randn((batch, sequence, heads, dim), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        call = lambda: source.block_causal_attention(q, k, v, 64)
        detail = f"Q/K/V={tuple(q.shape)} block=64 dtype={q.dtype}"
    elif case == "block_causal_varlen":
        lengths = [4096, 3840, 3584, 3328]
        prefix = torch.tensor([0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32)
        total, heads, dim = sum(lengths), 16, 128
        q = torch.randn((total, heads, dim), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        call = lambda: source.block_causal_attention_varlen(q, k, v, prefix, 64, max_seqlen=max(lengths), block_size=64)
        detail = f"packed_Q/K/V={tuple(q.shape)} lengths={lengths} dtype={q.dtype}"
    elif case == "native_sparse_forward":
        batch, sequence, heads, kv_heads, dim, selected, block = 2, 4096, 32, 4, 128, 64, 64
        q = torch.randn((batch, sequence, heads, dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, sequence, kv_heads, dim), device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        block_ids = torch.arange(selected, device="cuda", dtype=torch.int32).reshape(1, 1, 1, selected).expand(batch, sequence, kv_heads, selected).contiguous()
        call = lambda: source.native_sparse_attention(q, k, v, block_ids, dim=dim, is_causal=True, block_size=block, groups=heads // kv_heads, selected_blocks=selected)
        detail = f"Q={tuple(q.shape)} K/V={tuple(k.shape)} selected_blocks={selected}"
    elif case == "block_sparse_gemm":
        m = n = kdim = 4096
        a = torch.randn((m, kdim), device="cuda", dtype=torch.float16)
        b = torch.randn((kdim, n), device="cuda", dtype=torch.float16)
        block_m, block_n, block_k = 128, 128, 32
        mask = torch.rand((m // block_m, n // block_n, kdim // block_k), device="cuda") > 0.5
        call = lambda: source.blocksparse_matmul(a, b, mask, block_m, block_n, block_k, 2, 128, True)
        detail = "M=N=K=4096 block=(128,128,32) sparsity=0.5"
    else:
        raise ValueError(case)

    result, latency = runtime.elapsed_ms(call)
    print(f"algorithm={case} {detail}")
    if result is not None:
        print(runtime.describe(result))
    print(f"latency_ms={latency:.3f}")
