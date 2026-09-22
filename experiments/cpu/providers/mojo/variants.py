import torch

import intent

from kernels.contraction.gemm import Activation
from kernels.contraction.gemm import K as GEMM_K
from kernels.contraction.gemm import M as GEMM_M
from kernels.contraction.gemm import N as GEMM_N
from kernels.factorization.cholesky import BATCH as CHOLESKY_BATCH
from kernels.factorization.cholesky import SIZE as CHOLESKY_SIZE
from kernels.variants.activation import swiglu_forward_helper
from kernels.variants.contraction import gemm_loop_interchange
from kernels.variants.contraction import conv2d_reduce_order
from kernels.variants.decomposition import batched_cholesky_right_looking
from kernels.variants.decomposition import matrix_transpose_product_domains
from kernels.variants.decomposition import ordered_prefix_nested
from kernels.variants.indexing import rotary_embedding_equivalent_index
from kernels.variants.layout import matrix_transpose_scalar_domains
from kernels.variants.normalization import stable_softmax_online
from kernels.variants.normalization import weighted_layer_norm_second_moment
from kernels.variants.streaming import streamed_online_softmax_inline
from kernels.variants.streaming import flash_attention_inline_fwd, flash_attention_select_fwd, flash_attention_full_causal_stream_fwd
from kernels.activation.swiglu import FEATURES as SWIGLU_FEATURES
from kernels.activation.swiglu import TOKENS as SWIGLU_TOKENS
from kernels.layout.transpose import COLUMNS as TRANSPOSE_COLUMNS
from kernels.layout.transpose import ROWS as TRANSPOSE_ROWS
from kernels.position.rope import HALF_DIMENSION
from kernels.position.rope import HEAD_DIMENSION
from kernels.streaming.online_softmax import COLUMNS as ONLINE_COLUMNS
from kernels.streaming.online_softmax import ROWS as ONLINE_ROWS

from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_host_comparison, prepare_host_run_only


def swiglu_helper(context):
    configure_cpu_budget()
    shape = (SWIGLU_TOKENS, SWIGLU_FEATURES)
    gate = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    up = torch.randn_like(gate) * 0.5
    return prepare_host_comparison(
        context,
        swiglu_forward_helper,
        (gate, up),
        "swiglu_float_intermediate",
        Tolerance(atol=5.0e-2),
    )


def gemm_loop(context):
    configure_cpu_budget()
    a = torch.randn((GEMM_M, GEMM_K), dtype=torch.float16) * 0.1
    b = torch.randn((GEMM_K, GEMM_N), dtype=torch.float16) * 0.1
    return prepare_host_comparison(
        context,
        gemm_loop_interchange,
        (a, b),
        "matmul",
        Tolerance(atol=3.0e-2),
        constexprs={"ACTIVATION": Activation.NONE},
    )


def rotary_embedding_index(context):
    configure_cpu_budget()
    heads, tokens = 32, 2048
    rows = heads * tokens
    values = torch.randn((rows, HEAD_DIMENSION), dtype=torch.float16)
    cosine = torch.randn((tokens, HALF_DIMENSION), dtype=torch.float16)
    sine = torch.randn_like(cosine)
    return prepare_host_comparison(
        context,
        rotary_embedding_equivalent_index,
        (values, cosine, sine),
        "rotary_embedding_flat",
        Tolerance(atol=1.0e-2),
        constexprs={"HEADS": heads},
    )


def transpose_scalar_domains(context):
    configure_cpu_budget()
    x = torch.randn((TRANSPOSE_ROWS, TRANSPOSE_COLUMNS), dtype=torch.float16)
    return prepare_host_comparison(
        context,
        matrix_transpose_scalar_domains,
        (x,),
        "transpose",
        Tolerance(atol=0.0),
    )


def transpose_product_domains(context):
    x = torch.randn((TRANSPOSE_ROWS, TRANSPOSE_COLUMNS), dtype=torch.float16)
    return prepare_host_comparison(context, matrix_transpose_product_domains, (x,),
                                   "transpose", Tolerance(atol=0.0))


def conv2d_reduction(context):
    x = torch.randn((16, 256, 256), dtype=torch.float16) * 0.1
    weight = torch.randn((3, 3), dtype=torch.float16) * 0.1
    return prepare_host_comparison(context, conv2d_reduce_order, (x, weight), "conv2d", Tolerance(atol=3e-3))


def layer_norm_second_moment(context):
    x = torch.randn((8192, 4096), dtype=torch.float32)
    weight = torch.randn((4096,), dtype=torch.float32)
    bias = torch.randn_like(weight)
    return prepare_host_comparison(context, weighted_layer_norm_second_moment,
                                   (x, weight, bias, 1.0 / 4096, 1e-5), "layer_norm", Tolerance(atol=5e-5))


def prefix_nested(context):
    x = torch.randn((512, 17, 31), dtype=torch.float32) * 0.1
    return prepare_host_comparison(context, ordered_prefix_nested, (x,), "ordered_prefix", Tolerance(atol=2e-5))


def softmax_online(context):
    configure_cpu_budget()
    x = torch.randn((ONLINE_ROWS, ONLINE_COLUMNS), dtype=torch.float32)
    return prepare_host_comparison(
        context,
        stable_softmax_online,
        (x,),
        "softmax",
        Tolerance(atol=1.0e-5),
    )


def online_softmax_inline(context):
    configure_cpu_budget()
    x = torch.randn((ONLINE_ROWS, ONLINE_COLUMNS), dtype=torch.float32)
    return prepare_host_comparison(
        context,
        streamed_online_softmax_inline,
        (x,),
        "softmax",
        Tolerance(atol=1.0e-5),
    )


def cholesky_right_looking(context):
    configure_cpu_budget()
    seed = torch.randn(
        (CHOLESKY_BATCH, CHOLESKY_SIZE, CHOLESKY_SIZE), dtype=torch.float32
    )
    initial = seed @ seed.transpose(1, 2)
    initial.add_(torch.eye(CHOLESKY_SIZE)[None], alpha=CHOLESKY_SIZE)
    generated_matrix = initial.clone()
    source_matrix = initial.clone()

    report_stage("generated_compilation")
    artifact = intent.compile(
        batched_cholesky_right_looking,
        target=context.target,
        compiler=context.compiler,
        tuning_config=context.tuning_config,
    )
    runtime = load_module(
        context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py",
        "intent_cpu_reference_batched_cholesky_right_looking",
    )

    def generated_launch():
        artifact.run(generated_matrix)

    def source_launch():
        runtime.cholesky(source_matrix)

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(
            generated_launch,
            lambda: generated_matrix,
            prepare=lambda: generated_matrix.copy_(initial),
        ),
        PreparedLaunch(
            source_launch,
            lambda: source_matrix,
            prepare=lambda: source_matrix.copy_(initial),
        ),
        Tolerance(atol=3.0e-4),
        cuda_graph=False,
        device_type="cpu",
        cpu_host_timing=True,
        note="既有 B256-N16 f32 Cholesky left-looking versus right-looking；生成端与 CPU reference 使用独立 InOut 矩阵，每次调用前恢复且恢复不计时；双方计完整 host 调用。",
    )


def _flash_variant(context, definition):
    q = torch.randn((4, 32, 4096, 128), dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    return prepare_host_run_only(context, definition, (q, k, v, 128**-0.5), constexprs={"CAUSAL": True},
        note="该attention变体无独立原production runner/reference/容差，复用base flash已有B4/H32/S4096/D128 f16 causal输入规模；保留作者helper与概率cast，仅验证完整host调用可运行，不作数值或相对性能结论。")


def flash_inline(context):
    return _flash_variant(context, flash_attention_inline_fwd)


def flash_select(context):
    return _flash_variant(context, flash_attention_select_fwd)


def flash_full_causal(context):
    return _flash_variant(context, flash_attention_full_causal_stream_fwd)


CASES = {
    "swiglu_forward_helper": swiglu_helper,
    "gemm_loop_interchange": gemm_loop,
    "rotary_embedding_equivalent_index": rotary_embedding_index,
    "matrix_transpose_scalar_domains": transpose_scalar_domains,
    "matrix_transpose_product_domains": transpose_product_domains,
    "conv2d_reduce_order": conv2d_reduction,
    "weighted_layer_norm_second_moment": layer_norm_second_moment,
    "ordered_prefix_nested": prefix_nested,
    "stable_softmax_online": softmax_online,
    "streamed_online_softmax_inline": online_softmax_inline,
    "batched_cholesky_right_looking": cholesky_right_looking,
    "flash_attention_inline": flash_inline,
    "flash_attention_select": flash_select,
    "flash_attention_full_causal_stream": flash_full_causal,
}
