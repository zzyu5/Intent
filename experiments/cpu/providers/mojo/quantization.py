import intent
import torch

from kernels.quantization.fp8 import bf16_groupwise_fp8_quantize, f32_groupwise_fp8_quantize
from kernels.quantization.nvfp4 import nvfp4_quantize as nvfp4_quantize_definition
from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget


def fp8_groupwise(context):
    x = torch.randn((8192, 4096), dtype=torch.bfloat16)
    report_stage("generated_compilation")
    artifact = intent.compile(bf16_groupwise_fp8_quantize, target=context.target,
                              compiler=context.compiler, tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

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


def per_token_fp8(context):
    x = torch.randn((8192, 8192), dtype=torch.float32)
    report_stage("generated_compilation")
    artifact = intent.compile(f32_groupwise_fp8_quantize, target=context.target,
                              compiler=context.compiler, tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        scales = torch.zeros((8192, 64), dtype=torch.float32)
        state = {}

        def launch():
            state["outputs"] = function(x, scales)

        return PreparedLaunch(launch, lambda: state["outputs"], prepare=lambda: scales.zero_())

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.f32_groupwise_fp8_quantize),
        (Tolerance(atol=16.0, rtol=0.125), Tolerance(atol=1e-6, rtol=1e-4)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有 8192x8192 f32 group128 E4M3FN quantize；absmax 最小值1e-4，保留原输出和scale容差；独立scale存储，PyTorch CPU reference，完整 host 调用，单 NUMA 8 核。",
    )


def nvfp4_quantize(context):
    configure_cpu_budget()
    rows, columns = 8192, 4096
    x = torch.randn((rows, columns), dtype=torch.bfloat16)
    global_scale = torch.ones((1,), dtype=torch.float32)
    packed = torch.empty((rows, columns // 2), dtype=torch.uint8)
    row_blocks, column_blocks = rows // 128, columns // 64
    scales_shape = (row_blocks, column_blocks, 32, 4, 4)
    generated_packed = packed.clone()
    generated_scales = torch.empty(scales_shape, dtype=torch.uint8)
    source_packed = packed.clone()
    source_scales = torch.empty_like(generated_scales)

    report_stage("generated_compilation")
    artifact = intent.compile(
        nvfp4_quantize_definition,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    runtime = load_module(
        context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py",
        "intent_cpu_reference_nvfp4_quantize",
    )

    def generated_launch():
        artifact.run(
            x,
            global_scale,
            generated_packed,
            generated_scales,
        )

    def generated_outputs():
        return generated_packed, generated_scales.view(-1)

    def source_launch():
        runtime.nvfp4_quantize(
            x,
            global_scale,
            source_packed,
            source_scales,
        )

    def source_outputs():
        return source_packed, source_scales.view(-1)

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(generated_launch, generated_outputs),
        PreparedLaunch(source_launch, source_outputs),
        Tolerance(atol=0.0),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有 N8192-K4096 BF16 NVFP4 group16；保留 E2M1 nibble packing、E4M3 per-block scale 和原 packed/scales 输出 ABI；双方计完整 host 调用，输出 storage 分配不计时。",
    )


CASES = {
    "fp8_groupwise_quantize": fp8_groupwise,
    "per_token_fp8": per_token_fp8,
    "nvfp4_quantize": nvfp4_quantize,
}
