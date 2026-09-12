import intent
import torch

from kernels.backward.group_norm_silu import group_norm_silu_backward
from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance


def group_norm(context):
    x = torch.randn((32, 256, 1024), dtype=torch.bfloat16)
    upstream = torch.randn_like(x)
    weight = torch.randn((256,), dtype=torch.float32)
    bias = torch.randn_like(weight)
    grouped = x.float().reshape(32, 32, 8, 1024)
    mean = grouped.mean(dim=(2, 3))
    rstd = torch.rsqrt(grouped.var(dim=(2, 3), unbiased=False) + 1e-5)
    report_stage("generated_compilation")
    artifact = intent.compile(group_norm_silu_backward, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        dweight, dbias = torch.zeros_like(weight), torch.zeros_like(bias)
        state = {}

        def launch():
            state["output"] = function(x, upstream, weight, bias, mean, rstd, dweight, dbias, 1.0 / 8192)

        def prepare():
            dweight.zero_()
            dbias.zero_()

        return PreparedLaunch(launch, lambda: state["output"], prepare=prepare)

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.group_norm_silu_backward),
        (Tolerance(atol=4e-2), Tolerance(atol=3e-2), Tolerance(atol=3e-2)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有B32-C256-S1024-G32 bf16 GroupNorm-SiLU backward，dx及scatter-reduce参数梯度；单NUMA8核，PyTorch CPU reference，完整host调用；每次清零梯度且清零不计时。",
    )


CASES = {"group_norm_silu_backward": group_norm}
