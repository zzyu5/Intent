import intent
import torch

from kernels.factorization.triangular_solve import batched_lower_triangular_solve
from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance


def triangular_solve(context):
    lower = torch.tril(torch.randn((4096, 16, 16), dtype=torch.float32))
    lower.diagonal(dim1=-2, dim2=-1).add_(2.0)
    initial = torch.randn((4096, 16), dtype=torch.float32)
    report_stage("generated_compilation")
    artifact = intent.compile(batched_lower_triangular_solve, target=context.target,
                              compiler=context.compiler, tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        solution = initial.clone()
        return PreparedLaunch(lambda: function(lower, solution), lambda: solution,
                              prepare=lambda: solution.copy_(initial))

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.triangular_solve), Tolerance(atol=1e-4, rtol=1e-4),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有 B4096-N16 f32 lower-triangular solve；PyTorch eager CPU reference 保持逐列有序减法；单 NUMA 8 核；双方完整 host 调用，每次恢复相同 RHS，恢复不计时。",
    )


CASES = {"flaggems_triangular_solve": triangular_solve}
