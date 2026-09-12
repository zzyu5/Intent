import intent
import torch

from kernels.quantization.fp8 import bf16_groupwise_fp8_quantize
from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance


def fp8_groupwise(context):
    x = torch.randn((8192, 4096), dtype=torch.bfloat16)
    report_stage("generated_compilation")
    artifact = intent.compile(bf16_groupwise_fp8_quantize, target=context.target,
                              compiler=context.compiler, tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        scales = torch.zeros((8192, 32), dtype=torch.float32)
        state = {}

        def launch():
            state["outputs"] = function(x, scales)

        return PreparedLaunch(launch, lambda: state["outputs"], prepare=lambda: scales.zero_())

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.fp8_groupwise_quantize),
        (Tolerance(atol=16.0, rtol=0.125), Tolerance(atol=1e-6, rtol=1e-4)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有8192x4096 bf16、group128、E4M3FN量化；原输出及scale容差；独立scale存储，单NUMA8核，PyTorch CPU reference，完整host调用。",
    )


CASES = {"fp8_groupwise_quantize": fp8_groupwise}
