from __future__ import annotations

import torch

from kernels.activation.swiglu import swiglu_forward
from kernels.backward.group_norm import group_norm_backward_dx
from kernels.backward.group_norm import group_norm_backward_weight_bias
from kernels.backward.softmax import softmax_backward as softmax_backward_kernel
from kernels.normalization.batch_norm import batch_norm_training as batch_norm_training_kernel
from kernels.normalization.fused_add_rms_norm import fused_add_rms_norm
from kernels.normalization.layer_norm import layer_norm_f16
from kernels.normalization.layer_norm import layer_norm_bf16
from kernels.normalization.logsumexp import row_logsumexp
from kernels.normalization.rms_norm import rms_norm_bf16
from kernels.normalization.softmax import stable_softmax_f16

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def _runtime(context: Context, path: str, name: str):
    return load_module(context.project_root / path, name)


def fused_softmax(context: Context) -> PreparedComparison:
    x = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
    _, generated = compile_single(
        context,
        stable_softmax_f16,
        (x,),
    )
    runtime = _runtime(
        context,
        "source/triton/triton/normalization/softmax/02-fused-softmax_runtime.py",
        "intent_v2_triton_fused_softmax",
    )
    source_function = runtime.load_softmax()
    source = functional_launch(lambda: source_function(x))
    return PreparedComparison(generated, source, Tolerance(atol=1e-2, rtol=1e-2), cuda_graph=True)


def layer_norm(context: Context) -> PreparedComparison:
    hidden = 4096
    x = torch.randn((8192, hidden), device="cuda", dtype=torch.float16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.float16)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.float16)
    arguments = (x, weight, bias, 1.0 / hidden, 1e-5)
    _, generated = compile_single(
        context,
        layer_norm_f16,
        arguments,
    )
    runtime = _runtime(
        context,
        "source/triton/triton/normalization/layer_norm/05-layer-norm_runtime.py",
        "intent_v2_triton_layer_norm",
    )
    source_function = runtime.load_layer_norm()
    source = functional_launch(
        lambda: source_function(x, (hidden,), weight, bias, 1e-5)
    )
    return PreparedComparison(generated, source, Tolerance(atol=1e-2), cuda_graph=True)


def flash_layer_norm(context: Context) -> PreparedComparison:
    hidden = 4096
    x = torch.randn((8192, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    arguments = (x, weight, bias, 1.0 / hidden, 1e-5)
    _, generated = compile_single(
        context,
        layer_norm_bf16,
        arguments,
    )
    runtime = _runtime(
        context,
        "source/triton/flash-attention/normalization/layer_norm/layer_norm_runtime.py",
        "intent_v2_triton_flash_layer_norm",
    )
    source = functional_launch(lambda: runtime.upstream(arguments))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=True,
    )


def swiglu(context: Context) -> PreparedComparison:
    shape = (8192, 14336)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    _, generated = compile_single(
        context,
        swiglu_forward,
        (gate, up),
    )
    runtime = _runtime(
        context,
        "source/triton/liger-kernel/activation/swiglu/swiglu_runtime.py",
        "intent_v2_triton_swiglu",
    )
    source = functional_launch(lambda: runtime.upstream((gate, up)))
    return PreparedComparison(generated, source, Tolerance(atol=2e-2, rtol=1e-2), cuda_graph=True)


def fused_add_rms(context: Context) -> PreparedComparison:
    hidden = 4096
    shape = (8192, hidden)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    arguments = (x, residual, weight, 1.0 / hidden, 1e-6, 0.0)
    _, generated = compile_single(
        context,
        fused_add_rms_norm,
        arguments,
    )
    runtime = _runtime(
        context,
        "source/triton/liger-kernel/normalization/fused_add_rms_norm/fused_add_rms_norm_runtime.py",
        "intent_v2_triton_fused_add_rms_norm",
    )
    source = functional_launch(lambda: runtime.upstream(arguments))
    return PreparedComparison(generated, source, Tolerance(atol=2e-2, rtol=1e-2), cuda_graph=True)


def rms_norm(context: Context) -> PreparedComparison:
    hidden = 4096
    x = torch.randn((8192, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    arguments = (x, weight, 1.0 / hidden, 1e-6)
    _, generated = compile_single(
        context,
        rms_norm_bf16,
        arguments,
    )
    runtime = _runtime(
        context,
        "source/triton/liger-kernel/normalization/rms_norm/rms_norm_runtime.py",
        "intent_v2_triton_rms_norm",
    )
    source = functional_launch(lambda: runtime.upstream(arguments))
    return PreparedComparison(generated, source, Tolerance(atol=2e-2, rtol=1e-2), cuda_graph=True)


def xformers_rms_norm(context: Context) -> PreparedComparison:
    hidden = 4096
    x = torch.randn((8192, hidden), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16)
    arguments = (x, weight, 1.0 / hidden, 1e-6)
    _, generated = compile_single(
        context,
        rms_norm_bf16,
        arguments,
    )
    runtime = _runtime(
        context,
        "source/triton/xformers/normalization/rms_norm/rmsnorm_kernels_runtime.py",
        "intent_v2_triton_xformers_rms_norm",
    )
    source_module = runtime.load_source()
    source = functional_launch(
        lambda: source_module._rms_norm_forward(x, weight, 1e-6)
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=True,
    )


def flaggems_batch_norm_training(context: Context) -> PreparedComparison:
    batch, channels, spatial = 32, 64, 4096
    x = torch.randn(
        (batch, channels, spatial), device="cuda", dtype=torch.float16
    )
    weight = torch.randn((channels,), device="cuda", dtype=torch.float32)
    bias = torch.randn((channels,), device="cuda", dtype=torch.float32)
    initial_mean = torch.randn((channels,), device="cuda", dtype=torch.float32)
    initial_variance = (
        torch.rand((channels,), device="cuda", dtype=torch.float32) + 1.0
    )
    generated_mean = initial_mean.clone()
    generated_variance = initial_variance.clone()
    _, generated_base = compile_single(
        context,
        batch_norm_training_kernel,
        (x, weight, bias, generated_mean, generated_variance, 1e-5, 0.1),
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: (*generated_base.outputs(), generated_mean, generated_variance),
        prepare=lambda: (
            generated_mean.copy_(initial_mean),
            generated_variance.copy_(initial_variance),
        ),
    )
    runtime = _runtime(
        context,
        "source/triton/flag-gems/normalization/batch_norm/batch_norm_runtime.py",
        "intent_v2_triton_flaggems_batch_norm",
    )
    source_mean = initial_mean.clone()
    source_variance = initial_variance.clone()
    state: dict[str, object] = {}

    def source_launch():
        state["outputs"] = runtime.upstream(
            (x, weight, bias, source_mean, source_variance, 1e-5, 0.1)
        )

    source_launch()
    source = PreparedLaunch(
        launch=source_launch,
        outputs=lambda: (*state["outputs"], source_mean, source_variance),
        prepare=lambda: (
            source_mean.copy_(initial_mean),
            source_variance.copy_(initial_variance),
        ),
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=2e-2, rtol=1e-2),
            Tolerance(atol=1e-4, rtol=1e-4),
            Tolerance(atol=1e-4, rtol=1e-4),
            Tolerance(atol=1e-4, rtol=1e-4),
            Tolerance(atol=1e-4, rtol=1e-4),
        ),
        cuda_graph=False,
    )


def flaggems_group_norm_backward(context: Context) -> PreparedComparison:
    batch, channels, spatial, groups = 32, 256, 1024, 32
    x = torch.randn(
        (batch, channels, spatial), device="cuda", dtype=torch.float16
    )
    grad_y = torch.randn_like(x)
    weight = torch.randn((channels,), device="cuda", dtype=torch.float16)
    mean = torch.randn((batch, groups), device="cuda", dtype=torch.float16)
    rstd = (
        torch.rand((batch, groups), device="cuda", dtype=torch.float16) + 0.5
    )
    _, generated_dx = compile_single(
        context,
        group_norm_backward_dx,
        (x, grad_y, weight, mean, rstd, 1.0 / ((channels // groups) * spatial)),
    )
    _, generated_weight_bias = compile_single(
        context,
        group_norm_backward_weight_bias,
        (x, grad_y, mean, rstd),
    )
    generated = PreparedLaunch(
        launch=lambda: (generated_dx.launch(), generated_weight_bias.launch()),
        outputs=lambda: (generated_dx.outputs(), *generated_weight_bias.outputs()),
    )
    runtime = _runtime(
        context,
        "source/triton/flag-gems/normalization/group_norm/groupnorm_runtime.py",
        "intent_v2_triton_flaggems_group_norm_backward",
    )
    source = functional_launch(
        lambda: runtime.upstream((x, grad_y, weight, mean, rstd))
    )
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=5e-2, rtol=2e-2),
        cuda_graph=True,
    )


def flaggems_logsumexp(context: Context) -> PreparedComparison:
    x = torch.randn((8192, 8192), device="cuda", dtype=torch.float32)
    _, generated = compile_single(context, row_logsumexp, (x,))
    runtime = _runtime(
        context,
        "source/triton/flag-gems/normalization/logsumexp/logsumexp_runtime.py",
        "intent_v2_triton_flaggems_logsumexp",
    )
    source = functional_launch(lambda: runtime.upstream((x,)))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-4, rtol=1e-5),
        cuda_graph=True,
    )


def flaggems_softmax_backward(context: Context) -> PreparedComparison:
    probabilities = torch.randn(
        (4096, 4097), device="cuda", dtype=torch.float32
    )
    gradient = torch.randn_like(probabilities)
    _, generated = compile_single(
        context,
        softmax_backward_kernel,
        (probabilities, gradient),
    )
    runtime = _runtime(
        context,
        "source/triton/flag-gems/normalization/softmax/softmax_runtime.py",
        "intent_v2_triton_flaggems_softmax_backward",
    )
    source = functional_launch(lambda: runtime.upstream((probabilities, gradient)))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=1e-4, rtol=1e-5),
        cuda_graph=True,
    )


CASES = {
    "fused_softmax": fused_softmax,
    "layer_norm": layer_norm,
    "flash_layer_norm": flash_layer_norm,
    "swiglu": swiglu,
    "fused_add_rms_norm": fused_add_rms,
    "rms_norm": rms_norm,
    "xformers_rms_norm": xformers_rms_norm,
    "flaggems_batch_norm_training": flaggems_batch_norm_training,
    "flaggems_group_norm_backward": flaggems_group_norm_backward,
    "flaggems_logsumexp": flaggems_logsumexp,
    "flaggems_softmax_backward": flaggems_softmax_backward,
}
