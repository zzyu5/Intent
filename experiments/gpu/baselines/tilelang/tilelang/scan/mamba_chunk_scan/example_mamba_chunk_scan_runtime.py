import ast
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("example_mamba_chunk_scan.py")
TREE = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
TREE.body = [node for node in TREE.body if not isinstance(node, ast.If)]
NAMESPACE = {"__file__": str(SOURCE), "__name__": "intent_tilelang_mamba_chunk_scan"}
exec(compile(TREE, str(SOURCE), "exec"), NAMESPACE)
STATE = {}


def upstream(arguments):
    cb, x, dt, dA_cumsum, state_matrix, previous_states, residual_scale = arguments
    shape = (
        cb.shape[0],
        x.shape[1],
        cb.shape[3],
        cb.shape[2],
        x.shape[2],
        x.shape[3],
        state_matrix.shape[3],
    )
    if shape not in STATE:
        kernel = NAMESPACE["chunk_scan_fwd"].compile(
            batch=shape[0],
            seqlen=shape[1],
            chunk_size=shape[2],
            ngroups=shape[3],
            nheads=shape[4],
            headdim=shape[5],
            dstate=shape[6],
            block_M=64,
            block_N=64,
            block_K=64,
            block_Dstate=128,
            num_stages=2,
            threads=128,
        )
        STATE[shape] = (
            kernel.adapter._get_executable(),
            torch.empty_like(x),
        )
    executable, output = STATE[shape]
    executable(
        cb,
        x,
        dt,
        dA_cumsum,
        state_matrix,
        previous_states,
        residual_scale,
        output,
    )
    return output


def main():
    batch, seqlen, heads, groups = 1, 2048, 32, 8
    head_dimension, state_dimension, chunk_size = 64, 128, 256
    chunks = seqlen // chunk_size
    cb = torch.randn(
        (batch, chunks, groups, chunk_size, chunk_size),
        device="cuda",
        dtype=torch.float16,
    ) * 0.05
    x = torch.randn(
        (batch, seqlen, heads, head_dimension),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    dt = torch.rand(
        (batch, heads, chunks, chunk_size),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    dA_cumsum = -torch.rand_like(dt) * 0.1
    state_matrix = torch.randn(
        (batch, seqlen, groups, state_dimension),
        device="cuda",
        dtype=torch.float16,
    ) * 0.05
    previous_states = torch.randn(
        (batch, chunks, heads, head_dimension, state_dimension),
        device="cuda",
        dtype=torch.float16,
    ) * 0.05
    residual_scale = torch.randn(
        (heads,), device="cuda", dtype=torch.float16
    ) * 0.1
    arguments = (
        cb,
        x,
        dt,
        dA_cumsum,
        state_matrix,
        previous_states,
        residual_scale,
    )
    output = upstream(arguments)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = upstream(arguments)
    end.record()
    torch.cuda.synchronize()
    print(
        f"B={batch} S={seqlen} H={heads} G={groups} P={head_dimension} "
        f"N={state_dimension} C={chunk_size} dtype={x.dtype}"
    )
    print(f"output={tuple(output.shape)} finite={torch.isfinite(output).all().item()}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
