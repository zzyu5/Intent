import importlib.util
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("linalg_solve_triangular.py")
SPEC = importlib.util.spec_from_file_location(
    "intent_flaggems_triangular_solve",
    SOURCE,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
STATE = {}


def upstream(arguments):
    lower, solution = arguments
    batch, size, _ = lower.shape
    if size > 16:
        raise ValueError("this adapter covers the upstream small-diagonal path")
    if solution.shape != (batch, size):
        raise ValueError("triangular solve source ABI mismatch")
    if MODULE.HAS_TLE:
        MODULE._small_diag_kernel[(batch, 1)](
            lower,
            solution,
            size,
            1,
            32,
            False,
            False,
            lower.dtype == torch.float64,
            lower.stride(1),
            solution.stride(1),
        )
    else:
        key = (batch, size, lower.dtype, lower.device)
        if key not in STATE:
            STATE[key] = torch.zeros(
                batch * 16,
                device=lower.device,
                dtype=lower.dtype,
            )
        MODULE._small_diag_kernel_notle[(batch, 1)](
            lower,
            solution,
            STATE[key],
            size,
            1,
            1,
            32,
            False,
            False,
            lower.stride(1),
            solution.stride(1),
        )
    return solution
