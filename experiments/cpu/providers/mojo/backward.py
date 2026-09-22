
import intent
import torch

from kernels.backward.attention import (
    BATCH as ATTENTION_BATCH,
    HEAD_DIMENSION as ATTENTION_HEAD_DIMENSION,
    HEAD_GROUP as ATTENTION_HEAD_GROUP,
    KV_HEADS as ATTENTION_KV_HEADS,
    QUERY_HEADS as ATTENTION_QUERY_HEADS,
    SCALE as ATTENTION_SCALE,
    SEQUENCE as ATTENTION_SEQUENCE,
    attention_backward_delta,
    attention_backward_dkdv,
    attention_backward_dq,
)
from kernels.backward.group_norm_silu import group_norm_silu_backward
from kernels.backward.group_norm import (
    BATCH as GROUP_NORM_BATCH,
    CHANNELS as GROUP_NORM_CHANNELS,
    CHANNELS_PER_GROUP as GROUP_NORM_CHANNELS_PER_GROUP,
    GROUPS as GROUP_NORM_GROUPS,
    SPATIAL as GROUP_NORM_SPATIAL,
    group_norm_backward_dx,
    group_norm_backward_weight_bias,
)
from kernels.backward.layer_norm import (
    FEATURES as LAYER_NORM_FEATURES,
    PARTIAL_GROUPS as LAYER_NORM_PARTIAL_GROUPS,
    ROWS as LAYER_NORM_ROWS,
    layer_norm_backward_reduce,
    layer_norm_backward_rows,
)
from kernels.backward.sparse_mla import (
    sparse_mla_backward_delta,
    sparse_mla_backward_main,
    sparse_mla_grad_kv_cast,
)
from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import Context, PreparedComparison, PreparedLaunch, Tolerance


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
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

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


def layer_norm_backward(context):
    epsilon = 1.0e-5
    inverse_features = 1.0 / LAYER_NORM_FEATURES
    shape = (LAYER_NORM_ROWS, LAYER_NORM_FEATURES)
    x = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    dy = torch.randn(shape, dtype=torch.bfloat16) * 0.05
    weight = torch.randn((LAYER_NORM_FEATURES,), dtype=torch.bfloat16)
    x_f32 = x.float()
    mean = x_f32.mean(dim=1)
    centered = x_f32 - mean[:, None]
    rstd = torch.rsqrt(centered.square().mean(dim=1) + epsilon)

    report_stage("generated_compilation")
    rows_artifact = intent.compile(
        layer_norm_backward_rows,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    reduce_artifact = intent.compile(
        layer_norm_backward_reduce,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    runtime = load_module(
        context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py",
        "intent_cpu_reference_layer_norm_backward",
    )
    dx = torch.empty_like(x)
    partial_shape = (LAYER_NORM_PARTIAL_GROUPS, LAYER_NORM_FEATURES)
    dw_partial = torch.zeros(partial_shape, dtype=torch.bfloat16)
    db_partial = torch.zeros_like(dw_partial)
    dw = torch.empty((LAYER_NORM_FEATURES,), dtype=torch.float32)
    db = torch.empty_like(dw)
    generated_state = {}
    source_state = {}

    def generated_launch():
        dw_partial.zero_()
        db_partial.zero_()
        rows_artifact(
            x,
            dy,
            weight,
            mean,
            rstd,
            dx,
            dw_partial,
            db_partial,
            inverse_features,
        )
        reduce_artifact(dw_partial, db_partial, dw, db)
        generated_state["output"] = (dx, dw, db)

    def source_launch():
        source_state["output"] = runtime.layer_norm_backward(
            x, dy, weight, mean, rstd
        )

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(generated_launch, lambda: generated_state["output"]),
        PreparedLaunch(source_launch, lambda: source_state["output"]),
        (
            Tolerance(atol=5.0e-2),
            Tolerance(atol=1.25e-1),
            Tolerance(atol=1.25e-1),
        ),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有B4096-N4096 bf16 LayerNorm backward；rows partial后接reduce，完整返回dx/dweight/dbias，PyTorch CPU reference同算法。",
    )


def group_norm_backward(context):
    shape = (GROUP_NORM_BATCH, GROUP_NORM_CHANNELS, GROUP_NORM_SPATIAL)
    x = torch.randn(shape, dtype=torch.float16) * 0.5
    grad_y = torch.randn(shape, dtype=torch.float16) * 0.05
    weight = torch.randn((GROUP_NORM_CHANNELS,), dtype=torch.float16)
    grouped = x.float().reshape(
        GROUP_NORM_BATCH,
        GROUP_NORM_GROUPS,
        GROUP_NORM_CHANNELS_PER_GROUP,
        GROUP_NORM_SPATIAL,
    )
    mean = grouped.mean(dim=(2, 3)).to(torch.float16)
    centered = grouped - mean.float()[:, :, None, None]
    rstd = torch.rsqrt(centered.square().mean(dim=(2, 3)) + 1.0e-5).to(
        torch.float16
    )
    inverse_group_elements = 1.0 / (
        GROUP_NORM_CHANNELS_PER_GROUP * GROUP_NORM_SPATIAL
    )

    report_stage("generated_compilation")
    dx_artifact = intent.compile(
        group_norm_backward_dx,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    affine_artifact = intent.compile(
        group_norm_backward_weight_bias,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    runtime = load_module(
        context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py",
        "intent_cpu_reference_group_norm_backward",
    )
    grad_x = torch.empty_like(x)
    grad_weight = torch.empty_like(weight)
    grad_bias = torch.empty_like(weight)
    generated_state = {}
    source_state = {}

    def generated_launch():
        dx_artifact(
            x,
            grad_y,
            weight,
            mean,
            rstd,
            grad_x,
            inverse_group_elements,
        )
        affine_artifact(
            x,
            grad_y,
            mean,
            rstd,
            grad_weight,
            grad_bias,
        )
        generated_state["output"] = (grad_x, grad_weight, grad_bias)

    def source_launch():
        source_state["output"] = runtime.group_norm_backward(
            x, grad_y, weight, mean, rstd
        )

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(generated_launch, lambda: generated_state["output"]),
        PreparedLaunch(source_launch, lambda: source_state["output"]),
        (
            Tolerance(atol=2.0e-2),
            Tolerance(atol=2.5e-1),
            Tolerance(atol=2.5e-1),
        ),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有B32-C256-S1024-G32 f16 GroupNorm backward；dx与weight/bias梯度两个kernel完整调用，PyTorch CPU reference同算法。",
    )


def attention_backward_case(context):
    shape_q = (
        ATTENTION_BATCH,
        ATTENTION_QUERY_HEADS,
        ATTENTION_SEQUENCE,
        ATTENTION_HEAD_DIMENSION,
    )
    shape_kv = (
        ATTENTION_BATCH,
        ATTENTION_KV_HEADS,
        ATTENTION_SEQUENCE,
        ATTENTION_HEAD_DIMENSION,
    )
    q = torch.randn(shape_q, dtype=torch.float16) * 0.5
    k = torch.randn(shape_kv, dtype=torch.float16) * 0.5
    v = torch.randn(shape_kv, dtype=torch.float16) * 0.5
    grad_output = torch.randn_like(q) * 0.05

    runtime = load_module(
        context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py",
        "intent_cpu_reference_attention_backward",
    )
    output, lse, source_backward = runtime.prepare_attention_backward(
        q, k, v, grad_output, ATTENTION_SCALE
    )

    report_stage("generated_compilation")
    delta_artifact = intent.compile(
        attention_backward_delta,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    constexprs = {"HEAD_GROUP": ATTENTION_HEAD_GROUP, "CAUSAL": True}
    dkdv_artifact = intent.compile(
        attention_backward_dkdv,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
        constexprs=constexprs,
    )
    dq_artifact = intent.compile(
        attention_backward_dq,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
        constexprs=constexprs,
    )
    delta = torch.empty(shape_q[:-1], dtype=torch.float32)
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    generated_state = {}
    source_state = {}

    def generated_launch():
        delta_artifact(output, grad_output, delta)
        dkdv_artifact(
            q,
            k,
            v,
            grad_output,
            lse,
            delta,
            grad_k,
            grad_v,
            ATTENTION_SCALE,
        )
        dq_artifact(
            q,
            k,
            v,
            grad_output,
            lse,
            delta,
            grad_q,
            ATTENTION_SCALE,
        )
        generated_state["output"] = (grad_q, grad_k, grad_v)

    def source_launch():
        source_state["output"] = source_backward()

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(generated_launch, lambda: generated_state["output"]),
        PreparedLaunch(source_launch, lambda: source_state["output"]),
        (
            Tolerance(atol=1.25e-1),
            Tolerance(atol=2.5e-1),
            Tolerance(atol=1.25e-1),
        ),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有B2-HQ8-HK2-S1024-D64 f16 causal attention backward；delta、dkdv、dq三kernel完整调用；CPU source为原f32 autograd参考，保留前向图及中间量，生成端仅接收output/lse并重算概率，双方前向准备不计时。",
    )


def sparse_mla_backward_case(context: Context) -> PreparedComparison:
    batch, sequence, key_value_sequence = 1, 4096, 8192
    heads, key_value_groups = 64, 1
    query_dimension, value_dimension, topk = 576, 512, 2048
    query = torch.randn(
        (batch, sequence, heads, query_dimension), dtype=torch.bfloat16
    )
    key_value = torch.randn(
        (batch, key_value_sequence, key_value_groups, query_dimension),
        dtype=torch.bfloat16,
    )
    grad_output = torch.randn(
        (batch, sequence, heads, value_dimension), dtype=torch.bfloat16
    )
    positions = torch.arange(sequence, dtype=torch.int32)
    offsets = torch.arange(topk, dtype=torch.int32)
    causal_indices = positions[:, None] - offsets[None, :] - 1
    causal_indices = torch.where(
        offsets[None, :] < positions[:, None],
        causal_indices,
        torch.full_like(causal_indices, key_value_sequence),
    )
    selected_indices = causal_indices.view(
        batch, sequence, key_value_groups, topk
    ).contiguous()
    scale = query_dimension**-0.5

    runtime = load_module(
        context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py",
        "intent_cpu_reference_sparse_mla_backward",
    )
    output, lse, source_backward = runtime.prepare_sparse_mla_backward(
        query,
        key_value,
        grad_output,
        selected_indices,
        scale,
    )

    report_stage("generated_compilation")
    delta_artifact = intent.compile(
        sparse_mla_backward_delta,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    constexprs = {"HEAD_GROUP": heads // key_value_groups}
    main_artifact = intent.compile(
        sparse_mla_backward_main,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
        constexprs=constexprs,
    )
    cast_artifact = intent.compile(
        sparse_mla_grad_kv_cast,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    delta = torch.empty((batch, sequence, heads), dtype=torch.float32)
    grad_key_value = torch.zeros_like(key_value, dtype=torch.float32)
    grad_query = torch.empty_like(query)
    grad_key_value_bf16 = torch.empty_like(key_value)
    generated_state = {}
    source_state = {}

    def generated_launch() -> None:
        grad_key_value.zero_()
        delta_artifact(output, grad_output, delta)
        main_artifact(
            query,
            key_value,
            grad_output,
            selected_indices,
            lse,
            delta,
            grad_key_value,
            grad_query,
            scale,
        )
        cast_artifact(grad_key_value, grad_key_value_bf16)
        generated_state["output"] = (grad_query, grad_key_value_bf16)

    def source_launch() -> None:
        source_state["output"] = source_backward()

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(
            generated_launch,
            lambda: generated_state["output"],
        ),
        PreparedLaunch(source_launch, lambda: source_state["output"]),
        (
            Tolerance(atol=2.5e-1, rtol=5.0e-2),
            Tolerance(atol=2.5e-1, rtol=5.0e-2),
        ),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有B1-S4096-SKV8192-H64-DQ576-DV512-topk2048 bf16 sparse MLA backward；计入 f32 gradKV workspace 清零及 delta/main/gradKV cast 三kernel；CPU reference 同样重算概率并保持 bf16 dScore/P 转换，前向 output/LSE 准备不计入反向时间。",
    )


CASES = {
    "group_norm_silu_backward": group_norm,
    "layer_norm_backward": layer_norm_backward,
    "group_norm_backward": group_norm_backward,
    "attention_backward": attention_backward_case,
    "sparse_mla_backward": sparse_mla_backward_case,
}
