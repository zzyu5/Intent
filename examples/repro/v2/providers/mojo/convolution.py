import intent
import torch

from kernels.backward.causal_conv import (
    BATCH as BWD_CAUSAL_CONV_BATCH,
    CHANNELS as BWD_CAUSAL_CONV_CHANNELS,
    LENGTH as BWD_CAUSAL_CONV_LENGTH,
    WIDTH as BWD_CAUSAL_CONV_WIDTH,
    causal_conv1d_backward_partials,
    causal_conv1d_backward_reduce,
)
from kernels.convolution.direct import (
    CAUSAL_CONV_BATCH,
    CAUSAL_CONV_CHANNELS,
    CAUSAL_CONV_LENGTH,
    CAUSAL_CONV_WIDTH,
    CONV2D_BATCH,
    CONV2D_HEIGHT,
    CONV2D_WIDTH,
    CONV2D_FILTER_HEIGHT,
    CONV2D_FILTER_WIDTH,
    causal_depthwise_conv1d,
    causal_depthwise_conv1d_bf16,
    conv1d_same,
    conv2d_same,
    conv2d_nhwc,
    causal_depthwise_conv1d_update,
    causal_depthwise_conv1d_update_bf16,
)
from kernels.convolution.varlen import varlen_aligned_causal_depthwise_conv1d, varlen_causal_conv1d_final_state
from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance
from .common import prepare_host_comparison


def conv1d(context):
    x = torch.randn((64, 16384), dtype=torch.float16)
    weight = torch.randn((5,), dtype=torch.float16)
    return prepare_host_comparison(context, conv1d_same, (x, weight), "conv1d_same",
                                   Tolerance(atol=2e-2, rtol=1e-2))


def causal_conv1d(context):
    x = torch.randn(
        (CAUSAL_CONV_BATCH, CAUSAL_CONV_CHANNELS, CAUSAL_CONV_LENGTH),
        dtype=torch.float16,
    ) * 0.1
    weight = torch.randn(
        (CAUSAL_CONV_CHANNELS, CAUSAL_CONV_WIDTH), dtype=torch.float16
    ) * 0.1
    bias = torch.randn((CAUSAL_CONV_CHANNELS,), dtype=torch.float16) * 0.1
    return prepare_host_comparison(
        context,
        causal_depthwise_conv1d,
        (x, weight, bias),
        "causal_conv1d",
        Tolerance(atol=3e-3),
        constexprs={"SILU": True},
    )


def conv2d(context):
    x = torch.randn(
        (CONV2D_BATCH, CONV2D_HEIGHT, CONV2D_WIDTH), dtype=torch.float16
    ) * 0.1
    weight = torch.randn(
        (CONV2D_FILTER_HEIGHT, CONV2D_FILTER_WIDTH), dtype=torch.float16
    ) * 0.1
    return prepare_host_comparison(
        context,
        conv2d_same,
        (x, weight),
        "conv2d",
        Tolerance(atol=3e-3),
    )


def causal_conv1d_bf16(context):
    x = torch.randn((4, 4096, 4096), dtype=torch.bfloat16).transpose(1, 2)
    weight = torch.randn((4096, 4), dtype=torch.bfloat16)
    bias = torch.randn((4096,), dtype=torch.bfloat16)
    return prepare_host_comparison(
        context, causal_depthwise_conv1d_bf16, (x, weight, bias), "causal_conv1d",
        Tolerance(atol=5e-2, rtol=5e-2), constexprs={"SILU": True},
    )


def conv2d_channels_last(context):
    x = torch.randn((32, 128, 128, 256), dtype=torch.float16)
    weight = torch.randn((3, 3, 256, 512), dtype=torch.float16)
    return prepare_host_comparison(context, conv2d_nhwc, (x, weight), "conv2d_nhwc",
                                   Tolerance(atol=2e-2, rtol=2e-2))


def causal_conv1d_backward(context):
    shape = (
        BWD_CAUSAL_CONV_BATCH,
        BWD_CAUSAL_CONV_CHANNELS,
        BWD_CAUSAL_CONV_LENGTH,
    )
    x = torch.randn(shape, dtype=torch.float16) * 0.5
    weight = torch.randn(
        (BWD_CAUSAL_CONV_CHANNELS, BWD_CAUSAL_CONV_WIDTH),
        dtype=torch.float16,
    ) * 0.1
    grad_output = torch.randn_like(x) * 0.05
    report_stage("generated_compilation")
    partial = intent.compile(
        causal_conv1d_backward_partials,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    reduce = intent.compile(
        causal_conv1d_backward_reduce,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_causal_conv1d_backward",
    )
    generated_state = {}
    source_state = {}

    def generated_launch():
        grad_x, partial_weight, partial_bias = partial.run(x, weight, grad_output)
        grad_weight, grad_bias = reduce.run(partial_weight, partial_bias)
        generated_state["output"] = (grad_x, grad_weight, grad_bias)

    def source_launch():
        source_state["output"] = runtime.causal_conv1d_backward(
            x, weight, grad_output
        )

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(generated_launch, lambda: generated_state["output"]),
        PreparedLaunch(source_launch, lambda: source_state["output"]),
        (
            Tolerance(atol=2.5e-3),
            Tolerance(atol=0.25),
            Tolerance(atol=0.25),
        ),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有B8-D2048-L4096-W4 f16 causal conv backward；partial kernel后接reduce kernel，PyTorch CPU reference同算法，双方完整host调用。",
    )


def varlen_conv1d(context):
    lengths = (2048, 1536, 1024, 512)
    offsets = torch.tensor((0, 2048, 3584, 4608, 5120), dtype=torch.int64)
    chunks = torch.tensor([(sequence, chunk) for sequence, length in enumerate(lengths)
                           for chunk in range((length + 63) // 64)], dtype=torch.int32)
    x = torch.randn((sum(lengths), 4096), dtype=torch.bfloat16)
    weight = torch.randn((4096, 4), dtype=torch.float32)
    bias = torch.randn((4096,), dtype=torch.float32)
    report_stage("generated_compilation")
    forward = intent.compile(varlen_aligned_causal_depthwise_conv1d, target=context.target,
                             compiler=context.compiler, tuning_config=context.tuning_config)
    final_state = intent.compile(varlen_causal_conv1d_final_state, target=context.target,
                                 compiler=context.compiler, tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        state = {}

        def launch():
            state["output"] = function()

        return PreparedLaunch(launch, lambda: state["output"])

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(lambda: (forward.run(x, offsets, chunks, weight, bias), final_state.run(x, offsets))),
        side(lambda: runtime.varlen_causal_conv1d(x, offsets, weight, bias)),
        (Tolerance(atol=5e-2, rtol=5e-2), Tolerance(atol=0.0)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有变长卷积及final-state两kernel完整调用；原长度2048/1536/1024/512、D4096、bf16、chunk64；同算法PyTorch eager CPU reference，单NUMA8核；含ABI、输出分配与同步。",
    )


def causal_conv_update(context):
    x = torch.randn((64, 4096), dtype=torch.float16) * 0.1
    initial = torch.randn((64, 4096, 4), dtype=torch.float16) * 0.1
    weight = torch.randn((4096, 4), dtype=torch.float16) * 0.1
    bias = torch.randn((4096,), dtype=torch.float16) * 0.1
    report_stage("generated_compilation")
    artifact = intent.compile(causal_depthwise_conv1d_update, target=context.target,
                              compiler=context.compiler, tuning_config=context.tuning_config,
                              constexprs={"SILU": True})
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        state = initial.clone()
        result = {}

        def launch():
            result["output"] = function(x, state, weight, bias)

        return PreparedLaunch(launch, lambda: result["output"], prepare=lambda: state.copy_(initial))

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.causal_conv_update),
        (Tolerance(atol=0.0), Tolerance(atol=2e-3)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有B64-D4096-W4 f16缓存卷积更新及SiLU；单NUMA8核，PyTorch eager同算法，完整host调用；每次恢复同一state且恢复不计时。",
    )


def causal_conv_update_bf16(context):
    x = torch.randn((32, 4096), dtype=torch.bfloat16)
    initial = torch.randn((32, 4096, 4), dtype=torch.bfloat16)
    weight = torch.randn((4096, 4), dtype=torch.float32)
    bias = torch.randn((4096,), dtype=torch.float32)
    report_stage("generated_compilation")
    artifact = intent.compile(causal_depthwise_conv1d_update_bf16, target=context.target,
                              compiler=context.compiler, tuning_config=context.tuning_config,
                              constexprs={"SILU": True})
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        state = initial.clone()
        result = {}

        def launch():
            result["output"] = function(x, state, weight, bias)

        return PreparedLaunch(launch, lambda: result["output"], prepare=lambda: state.copy_(initial))

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.causal_conv_update),
        (Tolerance(atol=0.0), Tolerance(atol=5e-2, rtol=5e-2)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有 B32-D4096-W4 bf16 state/input、f32 weight/bias、SiLU；完整缓存更新及输出，PyTorch CPU reference，沿用原容差；单 NUMA 8 核，完整 host 调用，每次恢复 state 且恢复不计时。",
    )


CASES = {
    "flaggems_conv1d": conv1d,
    "causal_conv1d": causal_conv1d,
    "causal_conv1d_bf16": causal_conv1d_bf16,
    "conv2d": conv2d,
    "conv2d_nhwc": conv2d_channels_last,
    "causal_conv1d_backward": causal_conv1d_backward,
    "varlen_causal_conv1d": varlen_conv1d,
    "causal_conv_update": causal_conv_update,
    "causal_conv_update_bf16": causal_conv_update_bf16,
}
