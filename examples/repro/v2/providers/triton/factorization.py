from __future__ import annotations

import torch

from kernels.factorization.triangular_solve import batched_lower_triangular_solve

from ...loading import load_module
from ...measurement import compile_single
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def triangular_solve(context: Context) -> PreparedComparison:
    batch, size = 4096, 16
    lower = torch.tril(
        torch.randn((batch, size, size), device="cuda", dtype=torch.float32)
    )
    lower.diagonal(dim1=-2, dim2=-1).add_(2.0)
    initial = torch.randn((batch, size), device="cuda", dtype=torch.float32)
    generated_solution = initial.clone()
    _, generated_base = compile_single(
        context,
        batched_lower_triangular_solve,
        (lower, generated_solution),
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: generated_solution,
        prepare=lambda: generated_solution.copy_(initial),
    )
    runtime = load_module(
        context.project_root
        / "source/triton/flag-gems/factorization/triangular_solve/linalg_solve_triangular_runtime.py",
        "intent_v2_triton_flaggems_triangular_solve",
    )
    source_solution = initial.clone()
    source = PreparedLaunch(
        launch=lambda: runtime.upstream((lower, source_solution)),
        outputs=lambda: source_solution,
        prepare=lambda: source_solution.copy_(initial),
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-4, rtol=1e-4),
        cuda_graph=False,
    )


CASES = {"flaggems_triangular_solve": triangular_solve}
