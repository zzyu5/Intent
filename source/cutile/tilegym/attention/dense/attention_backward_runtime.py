import importlib.util
import math
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("attention.py")
UTILS = Path(__file__).parents[2] / "support" / "utils.py"
utils_spec = importlib.util.spec_from_file_location("tilegym.ops.cutile.utils", UTILS)
utils = importlib.util.module_from_spec(utils_spec)
sys.modules[utils_spec.name] = utils
utils_spec.loader.exec_module(utils)
spec = importlib.util.spec_from_file_location("tilegym.ops.cutile._intent_attention", SOURCE)
MODULE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)
STATE = {}


def _launch(state, q, k, v, output, grad_output, lse, scale, causal):
    stream = torch.cuda.current_stream()
    MODULE.ct.launch(
        stream,
        state["preprocess_grid"],
        MODULE._fmha_bwd_preprocess_kernel,
        (
            output,
            grad_output,
            state["delta"],
            state["preprocess_tile"],
            state["tile_d"],
            state["query_heads"],
            state["padded_query"],
        ),
    )
    dkdv_config = state["dkdv_config"]
    MODULE.ct.launch(
        stream,
        state["dkdv_grid"],
        state["dkdv_kernel"],
        (
            q,
            k,
            v,
            grad_output,
            state["grad_k"],
            state["grad_v"],
            lse.reshape(-1),
            state["delta"],
            scale,
            state["tile_d"],
            state["query_heads"],
            state["kv_heads"],
            state["padded_query"],
            dkdv_config.TILE_M,
            dkdv_config.TILE_N,
            state["head_group"],
            causal,
        ),
    )
    dq_config = state["dq_config"]
    MODULE.ct.launch(
        stream,
        state["dq_grid"],
        state["dq_kernel"],
        (
            q,
            k,
            v,
            grad_output,
            state["grad_q"],
            lse.reshape(-1),
            state["delta"],
            scale,
            state["tile_d"],
            state["query_heads"],
            state["padded_query"],
            dq_config.TILE_M,
            dq_config.TILE_N,
            state["head_group"],
            causal,
        ),
    )
    return state["grad_q"], state["grad_k"], state["grad_v"]


def upstream(arguments):
    q, k, v, output, grad_output, lse, scale, causal = arguments
    batch, query_heads, query_length, head_dimension = q.shape
    kv_heads, key_length = k.shape[1], k.shape[2]
    head_group = query_heads // kv_heads
    key = (
        tuple(q.shape),
        tuple(k.shape),
        q.dtype,
        q.device,
        float(scale),
        bool(causal),
    )
    if key not in STATE:
        MODULE._fmha_backward(q, k, v, output, grad_output, lse, scale, causal)
        tile_d = MODULE.next_power_of_2(head_dimension)
        configs = list(MODULE._fmha_bwd_autotune_configs(head_dimension))
        preprocess_tile = configs[0].TILE_M
        padded_query = math.ceil(query_length / max(c.TILE_M for c in configs)) * max(
            c.TILE_M for c in configs
        )
        if padded_query != query_length:
            raise NotImplementedError(
                "attention backward adapter currently requires an unpadded query length"
            )
        dkdv_key = (
            batch,
            query_heads,
            kv_heads,
            query_length,
            key_length,
            head_dimension,
            head_group,
            causal,
            q.dtype,
            str(q.device),
        )
        dq_key = (
            batch,
            query_heads,
            query_length,
            key_length,
            head_dimension,
            head_group,
            causal,
            q.dtype,
            str(q.device),
        )
        dkdv_config, dkdv_kernel = MODULE._fmha_bwd_dkdv_tune_cache[dkdv_key]
        dq_config, dq_kernel = MODULE._fmha_bwd_dq_tune_cache[dq_key]
        STATE[key] = {
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_group": head_group,
            "tile_d": tile_d,
            "padded_query": padded_query,
            "preprocess_tile": preprocess_tile,
            "preprocess_grid": (
                math.ceil(query_length / preprocess_tile),
                batch * query_heads,
                1,
            ),
            "dkdv_config": dkdv_config,
            "dkdv_kernel": dkdv_kernel,
            "dkdv_grid": (
                math.ceil(key_length / dkdv_config.TILE_N),
                batch * kv_heads,
                1,
            ),
            "dq_config": dq_config,
            "dq_kernel": dq_kernel,
            "dq_grid": (
                math.ceil(query_length / dq_config.TILE_M),
                batch * query_heads,
                1,
            ),
            "delta": torch.empty(
                (batch * query_heads * padded_query,),
                device=q.device,
                dtype=torch.float32,
            ),
            "grad_q": torch.empty_like(q),
            "grad_k": torch.empty_like(k),
            "grad_v": torch.empty_like(v),
        }
    return _launch(STATE[key], q, k, v, output, grad_output, lse, scale, causal)


def main():
    batch, query_heads, kv_heads = 2, 8, 2
    sequence, head_dimension = 1024, 64
    scale = head_dimension**-0.5
    q = torch.randn(
        (batch, query_heads, sequence, head_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    k = torch.randn(
        (batch, kv_heads, sequence, head_dimension),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    repeated_k = k.repeat_interleave(query_heads // kv_heads, dim=1)
    repeated_v = v.repeat_interleave(query_heads // kv_heads, dim=1)
    scores = torch.matmul(q.float(), repeated_k.float().transpose(-1, -2)) * scale
    causal = torch.ones(
        (sequence, sequence), device="cuda", dtype=torch.bool
    ).tril()
    scores.masked_fill_(~causal, -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.matmul(torch.softmax(scores, dim=-1), repeated_v.float()).half()
    grad_output = torch.randn_like(output)
    arguments = (q, k, v, output, grad_output, lse, scale, True)
    gradients = upstream(arguments)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    gradients = upstream(arguments)
    end.record()
    torch.cuda.synchronize()
    print(
        f"B={batch} QH={query_heads} KVH={kv_heads} S={sequence} "
        f"D={head_dimension} dtype={q.dtype} causal=True"
    )
    print(
        f"gradient_shapes={[tuple(value.shape) for value in gradients]} "
        f"finite={all(torch.isfinite(value).all().item() for value in gradients)}"
    )
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
