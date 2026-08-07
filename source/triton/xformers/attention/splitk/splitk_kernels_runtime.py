import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
import triton


def load_splitk_source():
    xformers_root = Path(__file__).parents[2]
    xformers_package = types.ModuleType("xformers")
    xformers_package.__path__ = []
    triton_package = types.ModuleType("xformers.triton")
    triton_package.__path__ = []
    xformers_package.triton = triton_package
    sys.modules["xformers"] = xformers_package
    sys.modules["xformers.triton"] = triton_package

    vararg_path = xformers_root / "support" / "triton" / "vararg_kernel.py"
    vararg_spec = importlib.util.spec_from_file_location("xformers.triton.vararg_kernel", vararg_path)
    vararg_module = importlib.util.module_from_spec(vararg_spec)
    sys.modules[vararg_spec.name] = vararg_module
    vararg_spec.loader.exec_module(vararg_module)

    kernel_path = Path(__file__).with_name("splitk_kernels.py")
    kernel_spec = importlib.util.spec_from_file_location("xformers_splitk_source", kernel_path)
    kernel_module = importlib.util.module_from_spec(kernel_spec)
    kernel_spec.loader.exec_module(kernel_module)
    return kernel_module


def main():
    # Long-context GQA decode: 32 requests, 32 query heads, 8 KV heads, 8K KV cache.
    batch, query_length, query_heads, kv_heads, kv_length, head_dim = 32, 1, 32, 8, 8192, 128
    groups = query_heads // kv_heads
    split_k = 8
    q = torch.randn(
        (batch, query_length, groups, kv_heads, head_dim), device="cuda", dtype=torch.float16
    )
    k_base = torch.randn((batch, kv_length, kv_heads, head_dim), device="cuda", dtype=torch.float16)
    v_base = torch.randn((batch, kv_length, kv_heads, head_dim), device="cuda", dtype=torch.float16)
    k = k_base[:, :, None, :, :].expand(batch, kv_length, groups, kv_heads, head_dim)
    v = v_base[:, :, None, :, :].expand(batch, kv_length, groups, kv_heads, head_dim)

    max_block_m = 32
    m_ceil = triton.cdiv(query_length, max_block_m) * max_block_m
    split_output = torch.empty(
        (batch, groups, kv_heads, split_k, m_ceil, head_dim), device="cuda", dtype=torch.float32
    )
    split_lse = torch.empty(
        (batch, groups, kv_heads, split_k, query_length), device="cuda", dtype=torch.float64
    )
    output = torch.empty(
        (batch, query_length, groups, kv_heads, head_dim), device="cuda", dtype=torch.float16
    )
    lse = torch.empty((batch, groups, kv_heads, query_length), device="cuda", dtype=torch.float64)
    source = load_splitk_source()
    forward = source._get_splitk_kernel(1)

    def launch():
        split_size = triton.cdiv(kv_length, split_k)
        grid = lambda meta: (
            triton.cdiv(query_length, meta["BLOCK_M"]),
            batch * groups * kv_heads,
            split_k,
        )
        forward[grid](
            Q=q,
            K=k,
            V=v,
            sm_scale=1.0 / math.sqrt(head_dim),
            Out_splitK=split_output,
            LSE_splitk=split_lse,
            block_tables=None,
            Seq_len=None,
            Seq_starts_k=None,
            Seq_starts_q=None,
            Seq_starts_q_multiplier=None,
            additive_bias=None,
            K_fp8_scale_shift=None,
            V_fp8_scale_shift=None,
            stride_qz=q.stride(0),
            stride_qm=q.stride(1),
            stride_qg=q.stride(2),
            stride_qh=q.stride(3),
            stride_qk=q.stride(4),
            stride_kz=k.stride(0),
            stride_kn=k.stride(1),
            stride_kg=k.stride(2),
            stride_kh=k.stride(3),
            stride_kk=k.stride(4),
            stride_vz=v.stride(0),
            stride_vn=v.stride(1),
            stride_vg=v.stride(2),
            stride_vh=v.stride(3),
            stride_vk=v.stride(4),
            stride_osk_z=split_output.stride(0),
            stride_osk_g=split_output.stride(1),
            stride_osk_h=split_output.stride(2),
            stride_osk_s=split_output.stride(3),
            stride_osk_m=split_output.stride(4),
            stride_osk_k=split_output.stride(5),
            stride_lsek_z=split_lse.stride(0),
            stride_lsek_g=split_lse.stride(1),
            stride_lsek_h=split_lse.stride(2),
            stride_lsek_s=split_lse.stride(3),
            stride_lsek_m=split_lse.stride(4),
            stride_blocktablesz=None,
            stride_blocktablesl=None,
            stride_bias_b=None,
            stride_bias_g=None,
            stride_bias_h=None,
            stride_bias_qm=None,
            stride_bias_km=None,
            stride_k_fp8_scale_shift_z=None,
            stride_k_fp8_scale_shift_n=None,
            stride_k_fp8_scale_shift_g=None,
            stride_k_fp8_scale_shift_h=None,
            stride_v_fp8_scale_shift_z=None,
            stride_v_fp8_scale_shift_n=None,
            stride_v_fp8_scale_shift_g=None,
            stride_v_fp8_scale_shift_h=None,
            kv_cache_blocks_per_row=0,
            Z=batch,
            H=kv_heads,
            G=groups,
            N_CTX_Q=query_length,
            N_CTX_K=kv_length,
            BLOCK_N_PER_SPLIT=split_size,
            BLOCK_DMODEL=head_dim,
            USE_SEQ_LEN=False,
            PACKED_PER_VAL=1,
            N_GROUPS=1,
            IS_CAUSAL=False,
            IS_LOCAL=False,
            NUM_QUERIES_CAUSAL=1,
            IS_SPLITK=True,
            SPLIT_K_EARLY_EXIT=False,
            USE_PAGED_ATTENTION=False,
            PAGE_SIZE=0,
            WINDOW_LEFT=-1,
            WINDOW_RIGHT=-1,
            WRITE_LSE=True,
            HAS_ADDITIVE_BIAS=False,
            NUM_PROGRAMS_DIM2_CONST=split_k,
            IS_HIP=False,
            BLOCK_M=16,
            BLOCK_N=64,
            num_warps=2,
            num_stages=1,
        )
        source._splitK_reduce[(query_length, batch * groups * kv_heads, 1)](
            split_output[..., :query_length, :],
            split_lse,
            output,
            lse,
            split_k=split_k,
            splitK_pow2=triton.next_power_of_2(split_k),
            stride_osk_z=split_output.stride(0),
            stride_osk_g=split_output.stride(1),
            stride_osk_h=split_output.stride(2),
            stride_osk_s=split_output.stride(3),
            stride_osk_m=split_output.stride(4),
            stride_osk_k=split_output.stride(5),
            stride_lsek_z=split_lse.stride(0),
            stride_lsek_g=split_lse.stride(1),
            stride_lsek_h=split_lse.stride(2),
            stride_lsek_s=split_lse.stride(3),
            stride_lsek_m=split_lse.stride(4),
            stride_oz=output.stride(0),
            stride_og=output.stride(2),
            stride_oh=output.stride(3),
            stride_om=output.stride(1),
            stride_ok=output.stride(4),
            stride_lse_z=lse.stride(0),
            stride_lse_g=lse.stride(1),
            stride_lse_h=lse.stride(2),
            stride_lse_m=lse.stride(3),
            head_dim=head_dim,
            head_dim_pow_2=triton.next_power_of_2(head_dim),
            H=kv_heads,
            G=groups,
            WRITE_LSE=True,
            num_warps=2,
        )

    launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    launch()
    end.record()
    torch.cuda.synchronize()

    visible_output = output.reshape(batch, query_length, query_heads, head_dim)
    print(f"Q: shape={(batch, query_length, query_heads, head_dim)}, dtype={q.dtype}")
    print(f"K/V: shape={(batch, kv_length, kv_heads, head_dim)}, split_k={split_k}")
    print(f"output: shape={tuple(visible_output.shape)}, mean={visible_output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
