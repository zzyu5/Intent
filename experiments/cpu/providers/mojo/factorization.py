import intent
import torch

from kernels.factorization.triangular_solve import batched_lower_triangular_solve
from kernels.factorization.cholesky import batched_cholesky_lower
from kernels.factorization.householder_qr import batched_householder_qr
from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import PreparedComparison, PreparedLaunch, Tolerance


def triangular_solve(context):
    lower = torch.tril(torch.randn((4096, 16, 16), dtype=torch.float32))
    lower.diagonal(dim1=-2, dim2=-1).add_(2.0)
    initial = torch.randn((4096, 16), dtype=torch.float32)
    report_stage("generated_compilation")
    artifact = intent.compile(batched_lower_triangular_solve, target=context.target,
                              compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

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


def _factorization(context, definition, initial, reference, tolerance):
    report_stage("generated_compilation")
    artifact = intent.compile(definition, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, options=context.compile_options)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        matrices = initial.clone()
        result = {}

        def launch():
            result["output"] = function(matrices)

        return PreparedLaunch(launch, lambda: result["output"],
                              prepare=lambda: matrices.copy_(initial))

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(getattr(runtime, reference)), tolerance,
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有batched factorization相同算法/输入规模/f32；单NUMA8核，PyTorch CPU reference写回相同InOut ABI；双方完整host调用，含内部临时量与输出分配，每次恢复输入矩阵且恢复不计时。",
    )


def cholesky(context):
    seed = torch.randn((256, 16, 16), dtype=torch.float32)
    matrices = seed @ seed.transpose(1, 2)
    matrices.add_(torch.eye(16)[None], alpha=16)
    return _factorization(context, batched_cholesky_lower, matrices, "cholesky",
                          Tolerance(atol=2e-4))


def householder_qr(context):
    matrices = torch.randn((128, 32, 16), dtype=torch.float32)
    return _factorization(context, batched_householder_qr, matrices, "householder_qr",
                          (Tolerance(atol=3e-4), Tolerance(atol=3e-4)))


CASES = {"flaggems_triangular_solve": triangular_solve,
         "cholesky": cholesky, "householder_qr": householder_qr}
