# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
from cuda.tile.tune import exhaustive_search
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
_splitk_reduce_tune_cache = {}


@ct.kernel(occupancy=ct.ByTarget(sm_80=2, default=4))
def _splitk_reduce_kernel(
    attn_splitk_out,
    lse_splitk_out,
    attn_out,
    B: ConstInt,
    S_KV: ConstInt,
    NUM_HEADS: ConstInt,
    HEAD_DIM: ConstInt,
    NUM_KV_SPLITS: ConstInt,
    NUM_KV_SPLITS_POW2: ConstInt,
    TILE_D: ConstInt,
    USE_DOT: ConstBool,
    ACCESS_FORM: ConstInt,
):
    # Get program IDs
    batch_id = ct.bid(0)  # batch index
    head_id = ct.bid(1)  # head index
    tile_id = ct.bid(2)  # tile index

    # Get data type
    dtype = attn_out.dtype

    # Load intermediate attention results with latency hint
    if ACCESS_FORM == 2:
        split_ids = ct.arange(NUM_KV_SPLITS_POW2, dtype=ct.int32)[:, None]
        dimensions = tile_id * TILE_D + ct.arange(TILE_D, dtype=ct.int32)[None, :]
        out_splitk = ct.gather(
            attn_splitk_out, (batch_id, head_id, split_ids, dimensions),
            padding_value=0,
        )
    else:
        out_splitk = ct.load(
            attn_splitk_out,
            (batch_id, head_id, 0, tile_id),
            shape=(1, 1, NUM_KV_SPLITS_POW2, TILE_D),
            order=(0, 1, 2, 3),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=True,
            latency=2,
        )
        out_splitk = ct.reshape(out_splitk, (NUM_KV_SPLITS_POW2, TILE_D))

    # Load and process lse results
    offs_lse = ct.arange(NUM_KV_SPLITS_POW2, dtype=ct.int32)
    # Form 0 preserves the untuned entry's mixed native/gather access.
    if ACCESS_FORM == 0 or ACCESS_FORM == 2:
        lse_splitk = ct.gather(
            lse_splitk_out,
            (batch_id, head_id, offs_lse),
            padding_value=-math.inf,
        )
    else:
        lse_splitk = ct.load(
            lse_splitk_out, (batch_id, head_id, 0),
            shape=(1, 1, NUM_KV_SPLITS_POW2),
            padding_mode=ct.PaddingMode.ZERO, allow_tma=True,
        )
        lse_splitk = ct.reshape(lse_splitk, (NUM_KV_SPLITS_POW2,))
        lse_splitk = ct.where(offs_lse < NUM_KV_SPLITS, lse_splitk, -math.inf)

    # Compute lse_max
    lse_max = ct.max(lse_splitk)

    # Compute sumexp_normalized_splitk
    sumexp_normalized_splitk = ct.exp2(lse_splitk - lse_max)
    sumexp_normalized_splitk = ct.astype(sumexp_normalized_splitk, ct.float32)

    # Compute sumexp_normalized
    sumexp_normalized = ct.sum(sumexp_normalized_splitk)

    # Compute numerator_normalized
    if USE_DOT:
        mma_result = ct.mma(
            sumexp_normalized_splitk[None, :],
            ct.astype(out_splitk, ct.float32),
            ct.zeros((1, TILE_D), dtype=ct.float32),
        )
        numerator_normalized = ct.extract(mma_result, (0, 0), shape=(1, TILE_D))
    else:
        numerator_normalized = ct.sum(
            out_splitk * ct.reshape(sumexp_normalized_splitk, (NUM_KV_SPLITS_POW2, 1)),
            axis=0,
        )

    # Compute final accumulator
    acc = numerator_normalized / sumexp_normalized

    # Cast to output dtype before storing
    acc = ct.astype(acc, dtype)

    # Store final result with latency hint
    ct.store(
        attn_out,
        index=(batch_id, head_id, tile_id),
        tile=ct.reshape(acc, (1, 1, TILE_D)),
        order=(0, 1, 2),
        allow_tma=ACCESS_FORM != 3,
        latency=2,
    )


@register_impl("splitk_reduce", backend="cutile")
def splitk_reduce(attn_splitk_out, lse_splitk_out, attn_out, S_kv,
                  *, tuning_configs=None, compiler_timeout=15, **kwargs):
    """
    Reduce the intermediate attention results and lse results into the final output for attention decode
    Args:
        attn_splitk_out: intermediate attention results [B, num_heads, NUM_KV_SPLITS, head_dim]
        lse_splitk_out: intermediate lse results [B, num_heads, NUM_KV_SPLITS]
        attn_out: final output [B, num_heads, head_dim]
        S_kv: sequence length of the key-value tensor, used for boundary check
    """
    B, num_heads, NUM_KV_SPLITS, head_dim = attn_splitk_out.shape
    TILE_D = min(128, next_power_of_2(head_dim))
    NUM_KV_SPLITS_POW2 = next_power_of_2(NUM_KV_SPLITS)

    # MMA is efficient once NUM_KV_SPLITS is large enough to amortize launch overhead.
    _split_cap = torch.cuda.get_device_capability()
    _dot_threshold = 4 if _split_cap[0] < 9 else 16
    USE_DOT = NUM_KV_SPLITS_POW2 >= _dot_threshold

    if tuning_configs is not None:
        configs = tuple(tuning_configs)
        for cfg in configs:
            if vars(cfg).keys() != {"NUM_KV_SPLITS_POW2", "TILE_D", "ACCESS_FORM", "occupancy", "num_ctas"}:
                raise ValueError("split-K reduction requires complete candidates without unconsumed fields")
            if cfg.ACCESS_FORM not in (1, 2, 3):
                raise ValueError("split-K reduction candidate has an unknown access form")
            if cfg.NUM_KV_SPLITS_POW2 < NUM_KV_SPLITS:
                raise ValueError("split-K reduction candidate does not cover every logical split")
        config_key = tuple(tuple(sorted(vars(cfg).items())) for cfg in configs)
        tensor_key = tuple(
            (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
             str(tensor.device), tensor.storage_offset())
            for tensor in (attn_splitk_out, lse_splitk_out, attn_out)
        )
        cache_key = (tensor_key, S_kv, config_key, compiler_timeout)
        stream = torch.cuda.current_stream()

        def grid_fn(cfg):
            return (B, num_heads, ct.cdiv(head_dim, cfg.TILE_D))

        def args_fn(cfg):
            return (attn_splitk_out, lse_splitk_out, attn_out, B, S_kv,
                    num_heads, head_dim, NUM_KV_SPLITS, cfg.NUM_KV_SPLITS_POW2,
                    cfg.TILE_D, cfg.NUM_KV_SPLITS_POW2 >= _dot_threshold,
                    cfg.ACCESS_FORM)

        def hints_fn(cfg):
            return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

        if cache_key not in _splitk_reduce_tune_cache:
            with ct.compiler_timeout(compiler_timeout):
                result = exhaustive_search(configs, stream, grid_fn, _splitk_reduce_kernel,
                                           args_fn, hints_fn=hints_fn)
            best = result.best.config
            _splitk_reduce_tune_cache[cache_key] = (
                best, _splitk_reduce_kernel.replace_hints(**hints_fn(best)),
            )
        best, kernel = _splitk_reduce_tune_cache[cache_key]
        ct.launch(stream, grid_fn(best), kernel, args_fn(best))
        return attn_out

    # Calculate grid dimensions
    grid = (B, num_heads, (head_dim + TILE_D - 1) // TILE_D)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _splitk_reduce_kernel,
        (
            attn_splitk_out,
            lse_splitk_out,
            attn_out,
            B,
            S_kv,
            num_heads,
            head_dim,
            NUM_KV_SPLITS,
            NUM_KV_SPLITS_POW2,
            TILE_D,
            USE_DOT,
            0,
        ),
    )

    return attn_out
