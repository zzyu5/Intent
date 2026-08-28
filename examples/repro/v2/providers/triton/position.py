from __future__ import annotations

import torch
import triton

from kernels.position.rope import rotary_embedding_bf16
from kernels.position.rope import rotary_qk_inplace
from kernels.position.rope_cache import padded_rope_cache_update

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...measurement import TRITON_PARAMETER_OWNERSHIP_N
from ...measurement import triton_parameter_value
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def rotary_embedding(context: Context) -> PreparedComparison:
    batch, sequence, heads, dimension = 4, 4096, 32, 128
    x = torch.randn(
        (batch, sequence, heads, dimension),
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.arange(sequence, device="cuda", dtype=torch.float32)
    inverse = 1.0 / (
        10000
        ** (
            torch.arange(0, dimension, 2, device="cuda", dtype=torch.float32)
            / dimension
        )
    )
    angles = torch.outer(positions, inverse)
    cosine, sine = angles.cos(), angles.sin()
    _, generated = compile_single(
        context,
        rotary_embedding_bf16,
        (x, cosine, sine),
        triton_config_filter=lambda config: (
            triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=3
            )
            == 8
            and triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=4
            )
            == 2
            and config.num_warps == 4
            and config.num_stages == 3
            and config.num_ctas == 1
        ),
    )
    runtime = load_module(
        context.project_root
        / "source/triton/flash-attention/position/rotary/rotary_runtime.py",
        "intent_v2_triton_rotary_embedding",
    )
    source = functional_launch(lambda: runtime.source.apply_rotary(x, cosine, sine))
    return PreparedComparison(generated, source, Tolerance(atol=5e-3, rtol=5e-2), cuda_graph=True)


def padded_rope(context: Context) -> PreparedComparison:
    batch, query_heads, kv_heads = 32, 32, 8
    cache_length, dimension = 8192, 128
    padded_length = cache_length + 1
    query = torch.randn(
        (batch, query_heads, dimension), device="cuda", dtype=torch.float16
    )
    key = torch.randn(
        (batch, kv_heads, dimension), device="cuda", dtype=torch.float16
    )
    value = torch.randn_like(key)
    sequence_lengths = torch.full(
        (batch,), padded_length, device="cuda", dtype=torch.int32
    )
    packed_input = torch.cat((query, key, value), dim=1)
    query_rows = batch * query_heads
    cache_rows = batch * padded_length * kv_heads
    generated_storage = torch.empty(
        (query_rows + 2 * cache_rows, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    _, generated_base = compile_single(
        context,
        padded_rope_cache_update,
        (
            packed_input,
            sequence_lengths,
            generated_storage,
            10000.0,
            1.0,
        ),
        triton_config_filter=lambda config: (
            config.num_warps == 1
            and config.num_stages == 3
            and config.num_ctas == 1
        ),
    )
    position = cache_length
    generated_query = generated_storage[:query_rows].view(
        batch, query_heads, dimension
    )
    generated_key_cache = generated_storage[
        query_rows : query_rows + cache_rows
    ].view(batch, padded_length, kv_heads, dimension)
    generated_value_cache = generated_storage[query_rows + cache_rows :].view(
        batch, padded_length, kv_heads, dimension
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: (
            generated_query,
            generated_key_cache[:, position],
            generated_value_cache[:, position],
        ),
    )

    runtime = load_module(
        context.project_root
        / "source/triton/xformers/position/rope_padded/rope_padded_kernels_runtime.py",
        "intent_v2_triton_padded_rope_runtime",
    )
    source_module = runtime.load_source()
    source_query = query.unsqueeze(0)
    source_key = key.unsqueeze(0)
    source_value = value.unsqueeze(0)
    source_query_output = torch.empty_like(source_query)
    source_key_cache = torch.empty(
        (1, batch * padded_length, kv_heads, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    source_value_cache = torch.empty_like(source_key_cache)
    query_offsets = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    cache_offsets = query_offsets * padded_length
    block_size = max(128, min(4096, triton.next_power_of_2(dimension)))
    total_heads = query_heads + 2 * kv_heads

    def source_launch():
        q_stride = source_query.stride()
        k_stride = source_key.stride()
        v_stride = source_value.stride()
        ck_stride = source_key_cache.stride()
        cv_stride = source_value_cache.stride()
        out_stride = source_query_output.stride()
        source_module._rope_padded_kernel[(1, batch, total_heads)](
            source_query,
            source_key,
            source_value,
            source_query_output,
            source_key_cache,
            source_value_cache,
            query_offsets,
            cache_offsets,
            sequence_lengths,
            10000.0,
            1.0,
            False,
            0,
            0,
            0,
            0,
            None,
            None,
            query_heads,
            query_heads + kv_heads,
            1,
            dimension,
            q_stride[1],
            0,
            q_stride[-2],
            k_stride[1],
            0,
            k_stride[-2],
            v_stride[1],
            0,
            v_stride[-2],
            ck_stride[1],
            0,
            ck_stride[-2],
            cv_stride[1],
            0,
            cv_stride[-2],
            query_offsets.stride(0),
            cache_offsets.stride(0),
            sequence_lengths.stride(0),
            out_stride[1],
            0,
            out_stride[-2],
            0,
            "f32",
            const_batch_strides=False,
            cache_padding_length=0,
            seqlenk_shift=0,
            BLOCK_SIZE=block_size,
            adjacents=False,
            num_warps=1,
        )

    source_launch()
    source_key_view = source_key_cache.view(
        batch, padded_length, kv_heads, dimension
    )
    source_value_view = source_value_cache.view(
        batch, padded_length, kv_heads, dimension
    )
    source = PreparedLaunch(
        launch=source_launch,
        outputs=lambda: (
            source_query_output.squeeze(0),
            source_key_view[:, position],
            source_value_view[:, position],
        ),
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=2e-3, rtol=2e-3),
            Tolerance(atol=2e-3, rtol=2e-3),
            Tolerance(atol=0.0),
        ),
        cuda_graph=False,
    )


def rope_qk(context: Context) -> PreparedComparison:
    batch, sequence, query_heads, kv_heads, dimension = 4, 4096, 32, 8, 128
    query_initial = torch.randn(
        (batch, query_heads, sequence, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    key_initial = torch.randn(
        (batch, kv_heads, sequence, dimension),
        device="cuda",
        dtype=torch.float16,
    )
    positions = torch.arange(sequence, device="cuda", dtype=torch.float32)
    inverse = 1.0 / (
        10000
        ** (
            torch.arange(0, dimension, 2, device="cuda", dtype=torch.float32)
            / dimension
        )
    )
    frequency = torch.outer(positions, inverse)
    embedding = torch.cat((frequency, frequency), dim=-1)
    cosine = embedding.cos().unsqueeze(0).to(torch.float16)
    sine = embedding.sin().unsqueeze(0).to(torch.float16)

    generated_query = query_initial.clone()
    generated_key = key_initial.clone()
    _, generated_base = compile_single(
        context,
        rotary_qk_inplace,
        (generated_query, generated_key, cosine, sine),
        triton_config_filter=lambda config: (
            triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=3
            )
            == dimension // 2
            and config.num_warps == 4
            and config.num_stages == 3
            and config.num_ctas == 1
        ),
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: (generated_query, generated_key),
        prepare=lambda: (
            generated_query.copy_(query_initial),
            generated_key.copy_(key_initial),
        ),
    )

    runtime = load_module(
        context.project_root
        / "source/triton/liger-kernel/position/rope/rope_runtime.py",
        "intent_v2_triton_rope_qk_runtime",
    )
    source_function = runtime.load_rope_forward()
    source_query = query_initial.clone()
    source_key = key_initial.clone()
    state: dict[str, object] = {}

    def source_launch():
        q_out, k_out, _, _ = source_function(
            source_query,
            source_key,
            cosine,
            sine,
        )
        state["outputs"] = (q_out, k_out)

    source_launch()
    source = PreparedLaunch(
        launch=source_launch,
        outputs=lambda: state["outputs"],
        prepare=lambda: (
            source_query.copy_(query_initial),
            source_key.copy_(key_initial),
        ),
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=2e-3, rtol=2e-3),
            Tolerance(atol=2e-3, rtol=2e-3),
        ),
        cuda_graph=False,
    )


CASES = {
    "rotary_embedding": rotary_embedding,
    "padded_rope_cache_update": padded_rope,
    "rope_qk": rope_qk,
}
