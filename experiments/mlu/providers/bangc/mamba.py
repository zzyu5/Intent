import importlib

import torch

from kernels.streaming.mamba import mamba_chunk_state_bf16_fwd, mamba_state_passing_fwd
from kernels.streaming.selective_scan import mamba_chunk_scan_bf16_fwd
from experiments._common.loading import load_module
from experiments._common.model import Tolerance
from .common import RemoteSequence


def _source(context, module, function):
    load_module(context.project_root /
        ("experiments/gpu/baselines/triton/state-spaces-mamba/mamba_ssm/ops/triton/" + module + "_runtime.py"),
        "intent_bangc_" + module)
    return getattr(importlib.import_module("mamba_ssm.ops.triton." + module), function)


def chunk_state(context):
    x = torch.randn((1, 2048, 32, 64), device="cuda", dtype=torch.bfloat16)
    basis = torch.randn((1, 2048, 8, 128), device="cuda", dtype=torch.bfloat16)
    dt = torch.rand((1, 32, 8, 256), device="cuda", dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    sequence = RemoteSequence(context)
    output = sequence.add(mamba_chunk_state_bf16_fwd,
        {"state_basis": basis, "x": x, "dt": dt, "cumulative_decay": decay},
        constexprs={"HEAD_GROUP": 4})["states"]
    source = _source(context, "ssd_chunk_state", "_chunk_state_fwd")
    return sequence.comparison(output, lambda: source(basis, x, dt, decay, states_in_fp32=True),
        Tolerance(atol=2e-2, rtol=2e-2))


def state_passing(context):
    states = torch.randn((1, 8, 32, 8192), device="cuda", dtype=torch.float32) * 0.01
    decay = -torch.rand((1, 32, 8), device="cuda", dtype=torch.float32) * 0.1
    initial = torch.zeros((1, 32, 8192), device="cuda", dtype=torch.float32)
    sequence = RemoteSequence(context)
    outputs = sequence.add(mamba_state_passing_fwd,
        {"chunk_states": states, "chunk_decay": decay, "initial_states": initial})
    source = _source(context, "ssd_state_passing", "_state_passing_fwd")
    return sequence.comparison((outputs["states_before_chunk"], outputs["final_states"]),
        lambda: source(states, decay, initial), Tolerance(atol=1e-5, rtol=1e-5))


def chunk_scan(context):
    x = torch.randn((1, 2048, 32, 64), device="cuda", dtype=torch.bfloat16)
    matrix = torch.randn((1, 2048, 8, 128), device="cuda", dtype=torch.bfloat16)
    cb = torch.randn((1, 8, 8, 256, 256), device="cuda", dtype=torch.bfloat16) * 0.01
    dt = torch.rand((1, 32, 8, 256), device="cuda", dtype=torch.float32) * 0.01
    decay = -torch.rand_like(dt).cumsum(-1) * 0.01
    previous = torch.randn((1, 8, 32, 64, 128), device="cuda", dtype=torch.float32) * 0.01
    residual = torch.randn((32,), device="cuda", dtype=torch.float32) * 0.01
    sequence = RemoteSequence(context)
    output = sequence.add(mamba_chunk_scan_bf16_fwd,
        {"cb": cb, "x": x, "dt": dt, "dA_cumsum": decay, "state_matrix": matrix,
         "previous_states": previous, "residual_scale": residual},
        constexprs={"HEADS_PER_GROUP": 4})["output"]
    source = _source(context, "ssd_chunk_scan", "_chunk_scan_fwd")
    return sequence.comparison(output, lambda: source(cb, x, dt, decay, matrix, previous, D=residual)[0],
        Tolerance(atol=1e-1, rtol=5e-2))


CASES = {"mamba_chunk_state": chunk_state, "mamba_state_passing": state_passing,
         "mamba_chunk_scan": chunk_scan}
