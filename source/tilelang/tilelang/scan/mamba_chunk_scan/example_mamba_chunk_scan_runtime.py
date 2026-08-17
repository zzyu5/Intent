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
