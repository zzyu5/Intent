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
    if case in {"native_sparse_forward", "native_sparse_decode"}:
        aliases.append(("reference", source_path.with_name("reference.py")))
    if case in {"sparse_mla_backward", "fp8_lighting_indexer"}:
        aliases.append(("utils", runtime.ROOT / "attention" / "support" / "deepseek_v32_utils.py"))
    return runtime.load_source(
        source_path,
        f"intent_tilelang_{case}",
        aliases=aliases,
        needs_fla_linear=case in {"linear_attention_forward", "linear_attention_backward"},
    )


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

    if case == "mamba_chunk_state":
        batch, sequence, heads, groups, dim, state_dim, chunk = 1, 2048, 32, 8, 64, 128, 256
        chunks = sequence // chunk
        state_basis = torch.randn((batch, sequence, groups, state_dim), device="cuda", dtype=torch.float16)
        x = torch.randn((batch, sequence, heads, dim), device="cuda", dtype=torch.float16)
        dt = torch.randn((batch, heads, chunks, chunk), device="cuda", dtype=torch.float16)
        cumulative_decay = torch.randn_like(dt)
        compiled = source.chunk_state_fwd.compile(
            batch=batch,
            seqlen=sequence,
            chunk_size=chunk,
            ngroups=groups,
            nheads=heads,
            headdim=dim,
            dstate=state_dim,
            block_M=64,
            block_N=128,
            block_K=64,
            num_stages=4,
            threads=128,
        )
        executable = compiled.adapter._get_executable()
        output = torch.empty((batch, chunks, heads, dim, state_dim), device="cuda", dtype=torch.float16)

        def call():
            executable(state_basis, x, dt, cumulative_decay, output)
            return output

        detail = f"B={batch} S={sequence} H={heads} G={groups} P={dim} N={state_dim} chunk={chunk}"
    elif case == "linear_attention_forward":
        batch, sequence, heads, dim = 1, 2048, 16, 128
        q = torch.randn((batch, sequence, heads, dim), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        output = torch.zeros((batch, sequence, heads, dim), device="cuda", dtype=torch.float32)
        kernel = source.tl_fused_chunk_fwd_kernel(batch, sequence, heads, dim, dim)

        def call():
            output.zero_()
            final_state = kernel(q, k, v, output)
            return output, final_state

        detail = f"Q/K/V={tuple(q.shape)} chunk=64 dtype={q.dtype}"
    elif case == "linear_attention_backward":
        batch, sequence, heads, dim = 1, 2048, 16, 128
        q = torch.randn((batch, sequence, heads, dim), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        grad_output = torch.randn_like(q)
        grad_q = torch.zeros_like(q, dtype=torch.float32)
        grad_k = torch.zeros_like(k, dtype=torch.float32)
        grad_v = torch.zeros_like(v, dtype=torch.float32)
        kernel = source.tl_fused_chunk_bwd_kernel(batch, sequence, heads, dim, dim)

        def call():
            grad_q.zero_()
            grad_k.zero_()
            grad_v.zero_()
            kernel(q, k, v, grad_output, grad_q, grad_k, grad_v)
            return grad_q, grad_k, grad_v

        detail = f"Q/K/V/dO={tuple(q.shape)} chunk=64 dtype={q.dtype}"
    elif case == "retention_forward":
        batch, sequence, heads, dim = 1, 2048, 16, 128
        q = torch.randn((batch, sequence, heads, dim), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        kernel = source.chunk_retention_fwd_kernel(batch, sequence, heads, dim, dim)
        call = lambda: source.postprocess(kernel(q, k, v))
        detail = f"Q/K/V={tuple(q.shape)} chunk=64 dtype={q.dtype}"
    elif case == "gqa_backward":
        batch, sequence, heads, kv_heads, dim = 1, 4096, 32, 8, 64
        groups = heads // kv_heads
        q = torch.randn((batch, sequence, heads, dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, sequence, kv_heads, dim), device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        grad_output = torch.randn_like(q)
        forward = source.flashattn_fwd(batch, heads, sequence, dim, dim, True, 128, 64, groups)
        output, logsumexp = forward(q, k, v)
        preprocess = source.flashattn_bwd_preprocess(batch, heads, sequence, dim)
        delta = preprocess(output, grad_output)
        backward = source.flashattn_bwd_split(batch, heads, sequence, dim, dim, True, 128, 32, groups=groups)
        postprocess = source.flashattn_bwd_postprocess(batch, heads, sequence, dim)
        grad_q = torch.zeros_like(q, dtype=torch.float32)
        grad_k_parts = torch.empty((groups, batch, sequence, kv_heads, dim), device="cuda", dtype=torch.float16)
        grad_v_parts = torch.empty_like(grad_k_parts)

        def call():
            grad_q.zero_()
            backward(q, k, v, grad_output, logsumexp, delta, grad_q, grad_k_parts, grad_v_parts)
            return postprocess(grad_q), grad_k_parts.sum(0), grad_v_parts.sum(0)

        detail = f"Q={tuple(q.shape)} K/V={tuple(k.shape)} causal=True dtype={q.dtype}"
    elif case == "native_sparse_decode":
        batch, sequence, heads, kv_heads, dim = 8, 8192, 32, 2, 128
        selected, block = 32, 128
        q = torch.randn((batch, 1, heads, dim), device="cuda", dtype=torch.float16)
        k = torch.randn((batch, sequence, kv_heads, dim), device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        block_ids = torch.arange(selected, device="cuda", dtype=torch.int32).reshape(1, 1, 1, selected)
        block_ids = block_ids.expand(batch, 1, kv_heads, selected).contiguous()
        call = lambda: source.native_sparse_attention(
            q, k, v, block_ids, dim=dim, block_size=block, groups=heads // kv_heads, selected_blocks=selected
        )
        detail = f"Q={tuple(q.shape)} K/V={tuple(k.shape)} selected_blocks={selected} block={block}"
    elif case == "grouped_gemm_backward":
        group_rows = [256, 512, 1024, 2048]
        reduction = output_dim = 4096
        total_rows = sum(group_rows)
        left = torch.randn((total_rows, reduction), device="cuda", dtype=torch.float16)
        right = torch.randn((total_rows, output_dim), device="cuda", dtype=torch.float16)
        sizes = torch.tensor(group_rows, device="cuda", dtype=torch.int32)
        offsets = torch.tensor([0, *itertools.accumulate(group_rows[:-1])], device="cuda", dtype=torch.int32)
        call = lambda: source.grouped_gemm_bwd(left, right, sizes, offsets, 64, 128, 64, num_stages=2, threads=256)
        detail = f"groups={group_rows} reduction={reduction} output={output_dim} dtype={left.dtype}"
    elif case == "bitnet_int2_decode":
        rows, output_dim, reduction = 1, 4096, 4096
        program = source.bitnet_158_int8xint2_decode(rows, output_dim, reduction, source.T.int8, source.T.int32, source.T.int32)
        kernel = source.tilelang.compile(program)
        activations = torch.randint(-8, 8, (rows, reduction), device="cuda", dtype=torch.int8)
        weights = torch.randint(0, 2, (output_dim, reduction), dtype=torch.int8).numpy()
        packed = source.general_compress(weights, source_bits=2)
        packed = source.interleave_weight(packed, 2, target_dtype=source.T.int8)
        packed = torch.from_numpy(packed).to(device="cuda")
        output = torch.empty((rows, output_dim), device="cuda", dtype=torch.int32)

        def call():
            kernel(activations, packed, output)
            return output

        detail = f"A={tuple(activations.shape)} packed_B={tuple(packed.shape)} output=int32"
    elif case == "block_fp4_quant":
        x = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16)
        call = lambda: source.fp4_act_quant(x, block_size=32)
        detail = f"x={tuple(x.shape)} block=32 dtype={x.dtype}"
    elif case == "block_causal":
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
