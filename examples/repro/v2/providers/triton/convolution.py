from __future__ import annotations

import torch

from kernels.convolution.direct import causal_depthwise_conv1d_bf16
from kernels.convolution.direct import causal_depthwise_conv1d_update_bf16
from kernels.convolution.direct import conv1d_same
from kernels.convolution.varlen import varlen_causal_conv1d_final_state
from kernels.convolution.varlen import varlen_aligned_causal_depthwise_conv1d

from ...loading import load_module
from ...measurement import compile_single
from ...measurement import functional_launch
from ...measurement import TRITON_PARAMETER_OWNERSHIP_N
from ...measurement import triton_parameter_value
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance


def causal_conv1d(context: Context) -> PreparedComparison:
    batch, channels, sequence, width = 4, 4096, 4096, 4
    x = torch.randn(
        (batch, sequence, channels), device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    weight = torch.randn((channels, width), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((channels,), device="cuda", dtype=torch.bfloat16)
    source_tiles = {
        (128, 256),
        (128, 128),
        (128, 64),
        (128, 32),
        (64, 128),
        (64, 64),
        (64, 32),
        (32, 256),
        (32, 128),
        (32, 64),
        (32, 32),
    }
    _, generated = compile_single(
        context,
        causal_depthwise_conv1d_bf16,
        (x, weight, bias),
        constexprs={"SILU": True},
        triton_config_filter=lambda config: (
            triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=1
            ),
            triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=3
            ),
        )
        in source_tiles
        and config.num_warps == 8
        and config.num_stages == 3
        and config.num_ctas == 1,
    )
    source_module = load_module(
        context.project_root
        / "source/triton/meta-applied-ai/convolution/causal_conv1d/causal_1d_conv.py",
        "intent_v2_triton_causal_conv1d",
    )
    source = functional_launch(
        lambda: source_module.causal_conv1d_fwd(
            x,
            weight,
            bias=bias,
            activation="silu",
        )
    )
    return PreparedComparison(generated, source, Tolerance(atol=5e-2, rtol=5e-2), cuda_graph=True)


def flaggems_conv1d(context: Context) -> PreparedComparison:
    batch, length, width = 64, 16384, 5
    x = torch.randn((batch, length), device="cuda", dtype=torch.float16)
    weight = torch.randn((width,), device="cuda", dtype=torch.float16)
    source_candidates = {
        (32, 4, 2),
        (64, 4, 2),
        (128, 4, 2),
        (32, 4, 3),
        (64, 4, 3),
        (128, 4, 3),
        (256, 8, 3),
        (64, 4, 4),
        (128, 4, 4),
        (256, 4, 4),
        (64, 2, 5),
    }
    _, generated = compile_single(
        context,
        conv1d_same,
        (x, weight),
        triton_config_filter=lambda config: (
            triton_parameter_value(config, TRITON_PARAMETER_OWNERSHIP_N),
            config.num_warps,
            config.num_stages,
        )
        in source_candidates
        and config.num_ctas == 1,
    )
    runtime = load_module(
        context.project_root
        / "source/triton/flag-gems/convolution/conv1d/conv1d_runtime.py",
        "intent_v2_triton_flaggems_conv1d",
    )
    source = functional_launch(lambda: runtime.upstream((x, weight)))
    return PreparedComparison(
        generated,
        source,
        Tolerance(atol=2e-2, rtol=1e-2),
        cuda_graph=True,
    )


def varlen_causal_conv1d(context: Context) -> PreparedComparison:
    lengths = (2048, 1536, 1024, 512)
    offsets = torch.tensor(
        (0, 2048, 3584, 4608, 5120),
        device="cuda",
        dtype=torch.int64,
    )
    hidden, width, chunk_size = 4096, 4, 64
    x = torch.randn(
        (sum(lengths), hidden), device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn((hidden, width), device="cuda", dtype=torch.float32)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.float32)
    runtime = load_module(
        context.project_root
        / "source/triton/fla/conv/causal1d/causal_conv_varlen_runtime.py",
        "intent_v2_triton_varlen_causal_conv_runtime",
    )
    chunk_indices = runtime.RUNTIME._prepare_chunk_indices(offsets, chunk_size)
    _, generated_forward = compile_single(
        context,
        varlen_aligned_causal_depthwise_conv1d,
        (x, offsets, chunk_indices, weight, bias),
        triton_config_filter=lambda config: (
            triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=2
            )
            in (16, 32, 64, 128)
            and config.num_warps in (4, 8, 16, 32)
            and config.num_stages == 3
            and config.num_ctas == 1
        ),
    )
    _, generated_state = compile_single(
        context,
        varlen_causal_conv1d_final_state,
        (x, offsets),
        triton_config_filter=lambda config: (
            triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=1
            )
            == 256
            and config.kwargs.get("FRAGMENT_S8") == 1
            and config.kwargs.get("FRAGMENT_S19") == 1
            and config.num_warps == 4
            and config.num_stages == 3
            and config.num_ctas == 1
        ),
    )

    def generated_launch():
        generated_forward.launch()
        generated_state.launch()

    generated = PreparedLaunch(
        launch=generated_launch,
        outputs=lambda: (
            generated_forward.outputs(),
            generated_state.outputs(),
        ),
    )
    source_module = runtime.RUNTIME.load_causal_conv(
        context.project_root / "source/triton/fla/conv/causal1d/ops.py"
    )
    source_base = functional_launch(
        lambda: source_module.causal_conv1d_fwd(
            x.unsqueeze(0),
            weight,
            bias,
            None,
            output_final_state=True,
            activation="silu",
            cu_seqlens=offsets,
            chunk_indices=chunk_indices,
            BT=chunk_size,
        )
    )
    source = PreparedLaunch(
        launch=source_base.launch,
        outputs=lambda: (
            source_base.outputs()[0].squeeze(0),
            source_base.outputs()[1].transpose(1, 2),
        ),
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=5e-2, rtol=5e-2),
            Tolerance(atol=0.0),
        ),
        cuda_graph=False,
    )


def causal_conv1d_update(context: Context) -> PreparedComparison:
    batch, hidden, width = 32, 4096, 4
    x = torch.randn((batch, hidden), device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(
        (batch, hidden, width), device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn((hidden, width), device="cuda", dtype=torch.float32)
    bias = torch.randn((hidden,), device="cuda", dtype=torch.float32)
    generated_state = initial.clone()
    _, generated_base = compile_single(
        context,
        causal_depthwise_conv1d_update_bf16,
        (x, generated_state, weight, bias),
        constexprs={"SILU": True},
        triton_config_filter=lambda config: (
            triton_parameter_value(
                config, TRITON_PARAMETER_OWNERSHIP_N, dimension=1
            )
            in (8, 16, 32, 64, 128, 256)
            and config.num_warps in (4, 8, 16, 32)
            and config.num_stages == 3
            and config.num_ctas == 1
        ),
    )
    generated = PreparedLaunch(
        launch=generated_base.launch,
        outputs=lambda: (generated_base.outputs(), generated_state),
        prepare=lambda: generated_state.copy_(initial),
    )
    runtime = load_module(
        context.project_root
        / "source/triton/fla/conv/causal1d/causal_conv_update_runtime.py",
        "intent_v2_triton_causal_conv_update_runtime",
    )
    source_module = runtime.RUNTIME.load_causal_conv(
        context.project_root / "source/triton/fla/conv/causal1d/ops.py"
    )
    source_state = initial.clone()
    state: dict[str, object] = {}

    def source_launch():
        output, updated = source_module.causal_conv1d_update(
            x,
            source_state,
            weight=weight,
            bias=bias,
            activation="silu",
        )
        state["output"] = output
        state["updated"] = updated

    source_launch()
    source = PreparedLaunch(
        launch=source_launch,
        outputs=lambda: (state["output"], state["updated"]),
        prepare=lambda: source_state.copy_(initial),
    )
    return PreparedComparison(
        generated,
        source,
        (
            Tolerance(atol=5e-2, rtol=5e-2),
            Tolerance(atol=0.0),
        ),
        cuda_graph=False,
    )


CASES = {
    "flaggems_conv1d": flaggems_conv1d,
    "causal_conv1d": causal_conv1d,
    "varlen_causal_conv1d": varlen_causal_conv1d,
    "causal_conv1d_update": causal_conv1d_update,
}
