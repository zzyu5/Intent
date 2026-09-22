import torch

from kernels.factorization.triangular_solve import batched_lower_triangular_solve
from kernels.ragged.jagged_mean import jagged_mean
from ...loading import load_module
from ...model import Tolerance
from .common import RemoteSequence


def jagged_mean_case(context):
    batch, features, max_length = 512, 128, 128
    lengths = (torch.arange(batch, device="cuda", dtype=torch.int32) % max_length) + 1
    offsets = torch.empty(batch + 1, device="cuda", dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths, dim=0)
    values = torch.randn(int(offsets[-1].item()), features, device="cuda", dtype=torch.float32)
    sequence = RemoteSequence(context)
    output = sequence.add(jagged_mean, {"values": values, "offsets": offsets})["output"]
    runtime = load_module(context.project_root /
        "source/triton/tritonbench/ragged/jagged_mean/jagged_mean_runtime.py", "intent_bangc_jagged_mean")
    return sequence.comparison(output, lambda: runtime.upstream((values, offsets)),
        Tolerance(atol=2e-5, rtol=1e-5))


def triangular_solve(context):
    batch, size = 4096, 16
    lower = torch.tril(torch.randn((batch, size, size), device="cuda", dtype=torch.float32))
    lower.diagonal(dim1=-2, dim2=-1).add_(2.0)
    initial = torch.randn((batch, size), device="cuda", dtype=torch.float32)
    solution = initial.clone()
    sequence = RemoteSequence(context)
    sequence.add(batched_lower_triangular_solve, {"lower": lower, "solution": solution})
    runtime = load_module(context.project_root /
        "source/triton/flag-gems/factorization/triangular_solve/linalg_solve_triangular_runtime.py",
        "intent_bangc_triangular_solve")
    return sequence.comparison(solution, lambda: runtime.upstream((lower, initial.clone())),
        Tolerance(atol=1e-4, rtol=1e-4))


CASES = {"jagged_mean": jagged_mean_case, "flaggems_triangular_solve": triangular_solve}
