import importlib.util
import sys
import types
from pathlib import Path

import torch


_STATE = {}


def _tensor_identity(tensor, *, track_contents=False):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor._version if track_contents else None,
    )


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
    vararg_spec = importlib.util.spec_from_file_location(
        "xformers.triton.vararg_kernel", vararg_path
    )
    vararg_module = importlib.util.module_from_spec(vararg_spec)
    sys.modules[vararg_spec.name] = vararg_module
    vararg_spec.loader.exec_module(vararg_module)

    kernel_path = Path(__file__).with_name("splitk_kernels.py")
    kernel_spec = importlib.util.spec_from_file_location(
        "xformers_splitk_source", kernel_path
    )
    kernel_module = importlib.util.module_from_spec(kernel_spec)
    kernel_spec.loader.exec_module(kernel_module)
    return kernel_module


def _prepare(arguments):
    q, key_cache, value_cache, page_offsets, page_indices, sequence_lengths, _ = (
        arguments
    )
    batch, query_heads, head_dim = q.shape
    page_count, page_size, kv_heads, cache_head_dim = key_cache.shape
    if (
        value_cache.shape != key_cache.shape
        or cache_head_dim != head_dim
        or query_heads % kv_heads != 0
    ):
        raise RuntimeError("xFormers paged decode received an incompatible GQA ABI")
    groups = query_heads // kv_heads
    page_counts = page_offsets[1:] - page_offsets[:-1]
    max_pages = int(page_counts.max().item())
    block_table = torch.zeros(
        (batch, max_pages), device=q.device, dtype=torch.int32
    )
    for request in range(batch):
        begin = int(page_offsets[request].item())
        end = int(page_offsets[request + 1].item())
        block_table[request, : end - begin] = page_indices[begin:end]

    q_source = (
        q.reshape(batch, kv_heads, groups, head_dim)
        .permute(0, 2, 1, 3)
        .unsqueeze(1)
    )
    physical_tokens = page_count * page_size
    key_source = (
        key_cache.reshape(physical_tokens, kv_heads, head_dim)[None, :, None, :, :]
        .expand(1, physical_tokens, groups, kv_heads, head_dim)
    )
    value_source = (
        value_cache.reshape(physical_tokens, kv_heads, head_dim)[None, :, None, :, :]
        .expand(1, physical_tokens, groups, kv_heads, head_dim)
    )
    splits = 8
    output_source = torch.empty(
        (batch, groups, kv_heads, splits, 1, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    lse = torch.empty(
        (batch, groups, kv_heads, splits, 1),
        device=q.device,
        dtype=torch.float64,
    )
    output = torch.empty(
        (batch, groups, kv_heads, 1, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    output_lse = torch.empty(
        (batch, groups, kv_heads, 1),
        device=q.device,
        dtype=torch.float64,
    )
    source = load_splitk_source()
    return {
        "forward": source._get_splitk_kernel(1),
        "reduce": source._splitK_reduce,
        "q": q_source,
        "key_source": key_source,
        "value_source": value_source,
        "block_table": block_table,
        "sequence_lengths": sequence_lengths,
        "partial_output": output_source,
        "partial_lse": lse,
        "output": output,
        "output_lse": output_lse,
        "batch": batch,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "groups": groups,
        "head_dim": head_dim,
        "page_size": page_size,
        "max_tokens": max_pages * page_size,
        "splits": splits,
    }


def paged_decode(arguments):
    q, key_cache, value_cache, page_offsets, page_indices, sequence_lengths, scale = (
        arguments
    )
    cache_key = (
        _tensor_identity(q),
        _tensor_identity(key_cache),
        _tensor_identity(value_cache),
        _tensor_identity(page_offsets, track_contents=True),
        _tensor_identity(page_indices, track_contents=True),
        _tensor_identity(sequence_lengths),
    )
    if _STATE.get("cache_key") != cache_key:
        _STATE.clear()
        _STATE.update(_prepare(arguments))
        _STATE["cache_key"] = cache_key
    state = _STATE
    forward = state["forward"]
    q_source = state["q"]
    key_source = state["key_source"]
    value_source = state["value_source"]
    partial_output = state["partial_output"]
    partial_lse = state["partial_lse"]
    output = state["output"]
    output_lse = state["output_lse"]
    block_table = state["block_table"]
    batch = state["batch"]
    kv_heads = state["kv_heads"]
    groups = state["groups"]
    head_dim = state["head_dim"]
    page_size = state["page_size"]
    max_tokens = state["max_tokens"]
    splits = state["splits"]
    block_n_per_split = (
        (max_tokens + splits * page_size - 1) // (splits * page_size)
    ) * page_size
    grid = (1, batch * groups * kv_heads, splits)
    forward[grid](
        Q=q_source,
        K=key_source,
        V=value_source,
        sm_scale=scale,
        Out_splitK=partial_output,
        LSE_splitk=partial_lse,
        block_tables=block_table,
        Seq_len=sequence_lengths,
        Seq_starts_k=None,
        Seq_starts_q=None,
        Seq_starts_q_multiplier=None,
        additive_bias=None,
        K_fp8_scale_shift=None,
        V_fp8_scale_shift=None,
        stride_qz=q_source.stride(0),
        stride_qm=q_source.stride(1),
        stride_qg=q_source.stride(2),
        stride_qh=q_source.stride(3),
        stride_qk=q_source.stride(4),
        stride_kz=key_source.stride(0),
        stride_kn=key_source.stride(1),
        stride_kg=key_source.stride(2),
        stride_kh=key_source.stride(3),
        stride_kk=key_source.stride(4),
        stride_vz=value_source.stride(0),
        stride_vn=value_source.stride(1),
        stride_vg=value_source.stride(2),
        stride_vh=value_source.stride(3),
        stride_vk=value_source.stride(4),
        stride_osk_z=partial_output.stride(0),
        stride_osk_g=partial_output.stride(1),
        stride_osk_h=partial_output.stride(2),
        stride_osk_s=partial_output.stride(3),
        stride_osk_m=partial_output.stride(4),
        stride_osk_k=partial_output.stride(5),
        stride_lsek_z=partial_lse.stride(0),
        stride_lsek_g=partial_lse.stride(1),
        stride_lsek_h=partial_lse.stride(2),
        stride_lsek_s=partial_lse.stride(3),
        stride_lsek_m=partial_lse.stride(4),
        stride_blocktablesz=block_table.stride(0),
        stride_blocktablesl=block_table.stride(1),
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
        kv_cache_blocks_per_row=block_table.shape[1],
        Z=batch,
        H=kv_heads,
        G=groups,
        N_CTX_Q=1,
        N_CTX_K=max_tokens,
        BLOCK_N_PER_SPLIT=block_n_per_split,
        BLOCK_DMODEL=head_dim,
        USE_SEQ_LEN=True,
        PACKED_PER_VAL=1,
        N_GROUPS=1,
        IS_CAUSAL=True,
        IS_LOCAL=False,
        NUM_QUERIES_CAUSAL=1,
        IS_SPLITK=True,
        SPLIT_K_EARLY_EXIT=True,
        USE_PAGED_ATTENTION=True,
        PAGE_SIZE=page_size,
        WINDOW_LEFT=-1,
        WINDOW_RIGHT=-1,
        WRITE_LSE=True,
        HAS_ADDITIVE_BIAS=False,
        NUM_PROGRAMS_DIM2_CONST=splits,
        IS_HIP=False,
        BLOCK_M=16,
        BLOCK_N=page_size,
        num_warps=4,
        num_stages=1,
    )
    splitk_pow2 = 1 << (splits - 1).bit_length()
    state["reduce"][(1, batch * groups * kv_heads, 1)](
        partial_output,
        partial_lse,
        output,
        output_lse,
        splits,
        splitk_pow2,
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_output.stride(4),
        partial_output.stride(5),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        partial_lse.stride(3),
        partial_lse.stride(4),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        output.stride(4),
        output_lse.stride(0),
        output_lse.stride(1),
        output_lse.stride(2),
        output_lse.stride(3),
        head_dim,
        head_dim,
        kv_heads,
        groups,
        False,
    )
    return (
        output[:, :, :, 0, :]
        .permute(0, 2, 1, 3)
        .reshape(batch, state["query_heads"], head_dim)
    )


upstream = paged_decode
