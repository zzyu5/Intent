import intent
import torch

from kernels.loss.cross_entropy import fused_cross_entropy, fused_cross_entropy_bf16, flash_cross_entropy_bf16
from kernels.loss.fused_linear_cross_entropy import (
    TOKENS, HIDDEN, VOCABULARY, CHUNK_SIZE,
    linear_logits_chunk, cross_entropy_probability_chunk, cross_entropy_mean,
)
from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance
from .common import prepare_host_comparison


def cross_entropy(context):
    initial = torch.randn((4096, 16384), dtype=torch.float32) * 0.5
    labels = torch.randint(0, 16384, (4096,), dtype=torch.int32)
    labels[::17] = -100
    report_stage("generated_compilation")
    artifact = intent.compile(fused_cross_entropy, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, constexprs={"IGNORE_INDEX": -100})
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        logits = initial.clone()
        state = {}

        def launch():
            state["outputs"] = function(logits, labels)

        return PreparedLaunch(launch, lambda: state["outputs"], prepare=lambda: logits.copy_(initial))

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.fused_cross_entropy),
        (Tolerance(atol=2e-5), Tolerance(atol=2e-4), Tolerance(atol=0.0)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有4096x16384 f32 fused cross entropy，含原地logits gradient、loss和prediction；原ignore_index -100；单NUMA8核，PyTorch CPU reference，完整host调用；每次恢复logits且恢复不计时。",
    )


def linear_cross_entropy(context):
    hidden = torch.randn((TOKENS, HIDDEN), dtype=torch.bfloat16)
    weight = torch.randn((VOCABULARY, HIDDEN), dtype=torch.bfloat16)
    target = torch.randint(0, VOCABULARY, (TOKENS,), dtype=torch.int64)
    logits = torch.empty((CHUNK_SIZE, VOCABULARY), dtype=torch.bfloat16)
    loss = torch.empty((TOKENS,), dtype=torch.float32)
    mean = torch.empty((1,), dtype=torch.float32)
    report_stage("generated_compilation")
    linear = intent.compile(linear_logits_chunk, target=context.target, compiler=context.compiler,
                            tuning_config=context.tuning_config)
    probability = intent.compile(cross_entropy_probability_chunk, target=context.target, compiler=context.compiler,
                                 tuning_config=context.tuning_config)
    reduction = intent.compile(cross_entropy_mean, target=context.target, compiler=context.compiler,
                               tuning_config=context.tuning_config)

    def launch():
        for begin in range(0, TOKENS, CHUNK_SIZE):
            end = min(begin + CHUNK_SIZE, TOKENS)
            chunk = logits[:end - begin]
            linear(hidden[begin:end], weight, chunk)
            probability(chunk, target[begin:end], loss[begin:end])
        reduction(loss, mean)

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(launch, lambda: (loss, mean, logits)), None, None,
        cuda_graph=False, device_type="cpu", cpu_host_timing=True, status="run_only",
        note="作者 TOKENS2048/HIDDEN4096/VOCABULARY32768/CHUNK_SIZE1024；按 token 分块执行 linear、原地 probability/loss 和全局 mean 三 kernel，保持 bf16 logits 与 f32 loss；复用预分配 logits scratch。无同合同 CPU reference，仅验证编译、执行和结果可访问；单 NUMA 8 核，计完整 pipeline host 调用。",
    )


def cross_entropy_bf16(context):
    initial = torch.randn((8192, 32768), dtype=torch.bfloat16)
    labels = torch.randint(0, 32768, (8192,), dtype=torch.int64)
    labels[::17] = -100
    report_stage("generated_compilation")
    artifact = intent.compile(fused_cross_entropy_bf16, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, constexprs={"IGNORE_INDEX": -100})
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        logits = initial.clone()
        state = {}

        def launch():
            state["outputs"] = function(logits, labels)

        return PreparedLaunch(launch, lambda: state["outputs"], prepare=lambda: logits.copy_(initial))

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.fused_cross_entropy_bf16),
        (Tolerance(atol=5e-2, rtol=5e-2), Tolerance(atol=5e-2, rtol=1e-2), Tolerance(atol=0.0)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有 8192x32768 bf16 logits、i64 labels；f32 loss、i64 prediction、原地 bf16 gradient；IGNORE_INDEX=-100，沿用各输出原容差；PyTorch CPU reference，单 NUMA 8 核，完整 host 调用；恢复 logits 不计时。",
    )


def flash_cross_entropy(context):
    logits = torch.randn((8192, 32768), dtype=torch.bfloat16)
    labels = torch.randint(0, 32768, (8192,), dtype=torch.int64)
    return prepare_host_comparison(
        context, flash_cross_entropy_bf16, (logits, labels), "flash_cross_entropy_bf16",
        (Tolerance(atol=5e-2, rtol=1e-2), Tolerance(atol=5e-2, rtol=1e-2)),
        constexprs={"IGNORE_INDEX": -100, "Z_LOSS_SCALE": 1e-4},
    )


CASES = {"fused_cross_entropy": cross_entropy, "fused_linear_cross_entropy": linear_cross_entropy,
         "fused_cross_entropy_bf16": cross_entropy_bf16, "flash_cross_entropy_bf16": flash_cross_entropy}
