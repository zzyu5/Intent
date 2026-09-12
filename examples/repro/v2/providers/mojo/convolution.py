import intent
import torch

from kernels.convolution.direct import conv1d_same, causal_depthwise_conv1d_update
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


CASES = {"flaggems_conv1d": conv1d, "varlen_causal_conv1d": varlen_conv1d,
         "causal_conv_update": causal_conv_update}
