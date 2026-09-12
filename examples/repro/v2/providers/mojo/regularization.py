from dataclasses import replace
import intent
import torch

from kernels.normalization.dropout_residual_rms_norm import (
    FEATURES,
    KEEP_PROBABILITY,
    ROWS,
    SEED,
    dropout_residual_rms_norm_backward_data,
    dropout_residual_rms_norm_forward,
)
from kernels.simulation.monte_carlo import barrier_option_paths

from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def dropout_residual_rms_norm(context):
    configure_cpu_budget()
    inverse_features = 1.0 / FEATURES
    inverse_keep_probability = 1.0 / KEEP_PROBABILITY
    epsilon = 1.0e-6
    weight_offset = 1.0
    shape = (ROWS, FEATURES)
    x = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    residual = torch.randn_like(x) * 0.5
    weight = torch.randn((FEATURES,), dtype=torch.bfloat16)
    dnormalized = torch.randn_like(x) * 0.25
    dresidual = torch.randn_like(x) * 0.25
    forward_arguments = (
        x,
        residual,
        weight,
        SEED,
        KEEP_PROBABILITY,
        inverse_keep_probability,
        inverse_features,
        epsilon,
        weight_offset,
    )
    backward_arguments = (
        x,
        residual,
        weight,
        dnormalized,
        dresidual,
        SEED,
        KEEP_PROBABILITY,
        inverse_keep_probability,
        inverse_features,
        epsilon,
        weight_offset,
    )

    report_stage("generated_compilation")
    forward = intent.compile(
        dropout_residual_rms_norm_forward,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    backward = intent.compile(
        dropout_residual_rms_norm_backward_data,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    runtime = load_module(
        context.project_root / "source/pytorch/cpu_runtime.py",
        "intent_cpu_reference_dropout_residual_rms_norm",
    )
    generated_outputs = {}
    source_outputs = {}

    def generated_launch():
        generated_forward = forward.run(*forward_arguments)
        generated_backward = backward.run(*backward_arguments)
        generated_outputs["output"] = (*generated_forward, *generated_backward)

    def source_launch():
        source_outputs["output"] = runtime.dropout_residual_rms_norm(
            x,
            residual,
            weight,
            dnormalized,
            dresidual,
            SEED,
            KEEP_PROBABILITY,
            inverse_keep_probability,
            inverse_features,
            epsilon,
            weight_offset,
        )

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(generated_launch, lambda: generated_outputs["output"]),
        PreparedLaunch(source_launch, lambda: source_outputs["output"]),
        tuple(Tolerance(atol=6.5e-2) for _ in range(4)),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有dropout residual RMSNorm forward/backward完整调用；bf16、seed17、keep0.9；CPU reference为NumPy Philox加PyTorch算术，调用内生成随机掩码，双方计完整host调用。",
    )


def barrier_option(context):
    configure_cpu_budget()
    comparison = prepare_host_comparison(
        context,
        barrier_option_paths,
        (17, 100.0, 95.0, 150.0, 0.0002, 0.01),
        "barrier_option_paths",
        Tolerance(atol=3.0e-4),
    )
    return replace(comparison, note="既有262144 paths/64 steps barrier option，原seed17及f32参数/容差；CPU reference为NumPy Philox加PyTorch路径更新，双方完整host调用，单NUMA8核。")


CASES = {
    "dropout_residual_rms_norm": dropout_residual_rms_norm,
    "barrier_option_paths": barrier_option,
}
