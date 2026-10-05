from __future__ import annotations

import intent
import torch
from programs.composition import MoEAlignment

from kernels.routing.mhc import mhc_apply_residual as mhc_apply_residual_definition
from kernels.routing.mhc import mhc_gemm_rms_partial, mhc_gemm_rms_finalize
from kernels.routing.mhc import mhc_pre_gemm_sqrsum, mhc_pre_fuse, mhc_sinkhorn
from kernels.variants.decomposition import moe_count_routes_product_domain
from kernels.routing.moe_align import BLOCK_SIZE
from kernels.routing.moe_align import EXPERTS
from kernels.routing.moe_align import PADDED_ROUTES
from kernels.routing.moe_align import ROUTES
from kernels.routing.moe_align import moe_count_routes
from kernels.routing.moe_align import moe_mark_expert_blocks
from kernels.routing.moe_align import moe_prefix_routes
from kernels.routing.moe_align import moe_scatter_routes
from kernels.routing.mqa_logits import fp8_mqa_logits

from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import Context, PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison


def _canonical_alignment(
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    total: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_value = int(total.reshape(-1)[0].item())
    blocks = total_value // BLOCK_SIZE
    canonical_ids = sorted_ids[:total_value].clone()
    active_experts = expert_ids[:blocks]
    for expert in range(EXPERTS):
        expert_blocks = torch.nonzero(active_experts == expert).flatten()
        if expert_blocks.numel() == 0:
            continue
        begin = int(expert_blocks[0].item()) * BLOCK_SIZE
        end = (int(expert_blocks[-1].item()) + 1) * BLOCK_SIZE
        canonical_ids[begin:end] = torch.sort(canonical_ids[begin:end]).values
    return canonical_ids, active_experts, total


def moe_alignment(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    tokens, topk = 4096, 2
    source_ids = (
        torch.arange(tokens * topk, dtype=torch.int32)
        .remainder(EXPERTS)
        .reshape(tokens, topk)
        .contiguous()
    )
    expert_counts = torch.zeros((EXPERTS,), dtype=torch.int32)
    expert_cursors = torch.zeros_like(expert_counts)
    sorted_routes = torch.full((PADDED_ROUTES,), ROUTES, dtype=torch.int32)

    report_stage("generated_compilation")
    count = intent.compile(
        moe_count_routes,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config, options=context.compile_options,
    )
    prefix = intent.compile(
        moe_prefix_routes,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config, options=context.compile_options,
    )
    scatter = intent.compile(
        moe_scatter_routes,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config, options=context.compile_options,
    )
    mark = intent.compile(
        moe_mark_expert_blocks,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config, options=context.compile_options,
    )
    generated_state: dict[str, tuple[torch.Tensor, ...]] = {}
    program = MoEAlignment(count, prefix, scatter, mark)

    def generated_launch() -> None:
        generated_state["output"] = program.run_into(
            source_ids, expert_counts, expert_cursors, sorted_routes)

    runtime = load_module(
        context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py",
        "intent_cpu_reference_moe_alignment",
    )
    source_state: dict[str, tuple[torch.Tensor, ...]] = {}

    def source_launch() -> None:
        source_state["output"] = runtime.moe_align_block_size(
            source_ids,
            BLOCK_SIZE,
            EXPERTS,
        )

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(
            generated_launch,
            lambda: _canonical_alignment(*generated_state["output"]),
        ),
        PreparedLaunch(source_launch, lambda: _canonical_alignment(*source_state["output"])),
        tuple(Tolerance(atol=0.0) for _ in range(3)),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有MoE alignment；T4096、top-k2、E64、block128；完整计数/前缀/散射/标记调用，辅助workspace初始化计入调用，输出为canonical三元组。",
    )


def mhc_post(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    tokens, hidden, streams = 4096, 2560, 4
    layer_output = torch.randn((tokens, hidden), dtype=torch.bfloat16)
    residual = torch.randn((tokens, streams, hidden), dtype=torch.bfloat16)
    post_mix = torch.randn((tokens, streams), dtype=torch.float32)
    residual_mix = torch.randn((tokens, streams, streams), dtype=torch.float32)
    return prepare_host_comparison(
        context,
        mhc_apply_residual_definition,
        (residual, layer_output, post_mix, residual_mix),
        "mhc_apply_residual",
        Tolerance(atol=5e-2, rtol=2e-2),
    )


def moe_count_routes_product_domain_case(context):
    ids = torch.randint(0, 64, (4096, 2), dtype=torch.int32)
    report_stage("generated_compilation")
    artifact = intent.compile(moe_count_routes_product_domain, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, options=context.compile_options)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        counts = torch.zeros((64,), dtype=torch.int32)
        return PreparedLaunch(lambda: function(ids, counts), lambda: counts, prepare=lambda: counts.zero_())

    report_stage("adapter_preparation")
    return PreparedComparison(side(artifact.run), side(runtime.moe_count_routes), Tolerance(atol=0.0),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="原product-domain MoE count T4096/top2/E64、随机i32 IDs，relaxed i32 atomic add；独立counts每次清零且清零不计时，CPU bincount参考，精确比较，完整host调用。")


def mhc_gemm_rms_scale(context):
    x = torch.randn((2048, 16384), dtype=torch.bfloat16)
    weight = torch.randn((16384, 24), dtype=torch.bfloat16)
    bias = torch.randn((24,), dtype=torch.bfloat16)
    report_stage("generated_compilation")
    partial = intent.compile(mhc_gemm_rms_partial, target=context.target, compiler=context.compiler,
                             tuning_config=context.tuning_config, options=context.compile_options, constexprs={"P": 16})
    finalize = intent.compile(mhc_gemm_rms_finalize, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, options=context.compile_options,
                              constexprs={"P": 16, "STREAMS": 4, "ALPHA_PRE": 1.0,
                                          "ALPHA_POST": 1.0, "ALPHA_RESIDUAL": 1.0})
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")
    generated, source = {}, {}

    def launch():
        linear, square_sum = partial.run(x, weight)
        generated["output"] = finalize.run(linear, square_sum, bias, x.shape[1])

    def reference():
        source["output"] = runtime.mhc_gemm_rms_scale(x, weight, bias)

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(launch, lambda: generated["output"]), PreparedLaunch(reference, lambda: source["output"]),
        (Tolerance(atol=5e-2, rtol=2e-2), Tolerance(atol=5e-3, rtol=1e-3)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有 T2048/H4096/streams4/P16 bf16 MHC GEMM+RMS；partial/finalize 两 kernel 完整 host 调用，含中间及输出分配；PyTorch CPU reference，保留 split16、bf16 mixed、f32 rms 与原容差，单 NUMA 8 核。",
    )


def mhc_pre(context):
    factors = 1 + torch.arange(4).mul(0.01).view(1, -1, 1)
    residual = (torch.randn((2048, 4, 4096), dtype=torch.float32) * factors).bfloat16()
    weight = (torch.randn((24, 4, 4096), dtype=torch.float32) * 1e-4 * factors).flatten(1, 2)
    scale = torch.randn((3,), dtype=torch.float32) * 0.1
    base = torch.randn((24,), dtype=torch.float32) * 0.1
    report_stage("generated_compilation")
    first = intent.compile(mhc_pre_gemm_sqrsum, target=context.target, compiler=context.compiler,
                           tuning_config=context.tuning_config, options=context.compile_options)
    second = intent.compile(mhc_pre_fuse, target=context.target, compiler=context.compiler,
                            tuning_config=context.tuning_config, options=context.compile_options,
                            constexprs={"RMS_EPS": 1e-6, "PRE_EPS": 1e-6, "SINKHORN_EPS": 1e-6,
                                        "POST_MULTIPLIER": 1.0, "SINKHORN_REPEATS": 10})
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")
    generated, source = {}, {}

    def launch():
        mixes, square_sum = first.run(residual.flatten(1, 2), weight)
        generated["output"] = second.run(mixes, square_sum, scale, base, residual)

    def reference():
        source["output"] = runtime.mhc_pre(residual, weight, scale, base)

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(launch, lambda: generated["output"]), PreparedLaunch(reference, lambda: source["output"]),
        (Tolerance(atol=1e-2, rtol=1e-3), Tolerance(atol=1e-2, rtol=1e-3), Tolerance(atol=5e-2, rtol=2e-2)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有 T2048/H4096/streams4 MHC pre 两 kernel；weight 先转 bf16，f32 GEMM/RMS/混合，Sinkhorn10 次、原 epsilon 和容差；PyTorch CPU reference 沿用 source 的 mhc_pre_ref 算法，计完整 host 调用及分配，单 NUMA 8 核。",
    )


def mhc_sinkhorn_case(context):
    initial = torch.randn((8192, 4, 4), dtype=torch.float32)
    report_stage("generated_compilation")
    artifact = intent.compile(mhc_sinkhorn, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config, options=context.compile_options)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        values = initial.clone()
        return PreparedLaunch(lambda: function(values), lambda: values, prepare=lambda: values.copy_(initial))

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.mhc_sinkhorn), Tolerance(atol=2e-4, rtol=1e-4),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有 T8192/streams4 f32 MHC Sinkhorn；exp 后固定 20 次行列归一化、无 epsilon；每次恢复独立输入且恢复不计时，PyTorch CPU reference、原容差，完整 host 调用，单 NUMA 8 核。",
    )


def flaggems_fp8_mqa_logits(context: Context) -> PreparedComparison:
    configure_cpu_budget()
    queries, keys, heads, dimension = 256, 4096, 32, 128
    q = (
        torch.randn((queries, heads, dimension), dtype=torch.float16) * 0.25
    ).to(torch.float8_e4m3fn)
    kv = (
        torch.randn((keys, dimension), dtype=torch.float16) * 0.25
    ).to(torch.float8_e4m3fn)
    kv_scale = 0.5 + torch.rand((keys,), dtype=torch.float32)
    head_weight = torch.rand((queries, heads), dtype=torch.float32) * 0.1
    indices = torch.arange(queries, dtype=torch.int32)
    key_start = indices % 17
    key_end = keys - indices % 19
    return prepare_host_comparison(
        context,
        fp8_mqa_logits,
        (q, kv, kv_scale, head_weight, key_start, key_end),
        "fp8_mqa_logits",
        Tolerance(atol=2.0e-2, rtol=2.0e-2),
    )


CASES = {
    "moe_alignment": moe_alignment,
    "mhc_post": mhc_post,
    "mhc_gemm_rms_scale": mhc_gemm_rms_scale,
    "mhc_pre": mhc_pre,
    "mhc_sinkhorn": mhc_sinkhorn_case,
    "moe_count_routes_product_domain": moe_count_routes_product_domain_case,
    "flaggems_fp8_mqa_logits": flaggems_fp8_mqa_logits,
}
