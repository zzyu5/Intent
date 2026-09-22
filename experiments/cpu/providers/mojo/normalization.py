import intent
import torch
from kernels.normalization.rms_norm import weighted_rms_norm, rms_norm_f32, rms_norm_bf16
from kernels.normalization.softmax import stable_softmax, stable_softmax_f16, chunked_softmax_bf16
from kernels.normalization.layer_norm import weighted_layer_norm, layer_norm_f16, layer_norm_bf16
from kernels.normalization.logsumexp import row_logsumexp
from kernels.normalization.batch_norm import batch_norm_training
from kernels.normalization.fused_add_rms_norm import fused_add_rms_norm
from kernels.backward.softmax import softmax_backward as backward_definition
from experiments._common.loading import load_module
from experiments._common.measurement import report_stage
from experiments._common.model import PreparedComparison, PreparedLaunch, Tolerance
from .common import configure_cpu_budget, prepare_comparison, prepare_host_comparison


def rms_norm(context):
    x = torch.randn((8192, 4096), dtype=torch.float32)
    weight = torch.randn((4096,), dtype=torch.float32)
    return prepare_comparison(context, weighted_rms_norm, (x, weight, 1.0 / 4096, 1e-6),
        "experiments/cpu/baselines/mojo/modular/normalization/rms_norm/rms_norm_runtime.py", Tolerance(5e-5),
        "Source 调用安装的 Modular/MAX RMSNorm；双方归约括号与中间数学实现可能不同。")


def softmax(context):
    x = torch.randn((8192, 8192), dtype=torch.float32)
    return prepare_comparison(context, stable_softmax, (x,),
        "experiments/cpu/baselines/mojo/modular/normalization/softmax/softmax_runtime.py", Tolerance(1e-6, 1e-5),
        "Source 调用安装的 Modular/MAX CPU Softmax；双方 max/exp/sum 后显式以倒数乘法归一化，归约树仍可能不同。")


def layer_norm(context):
    x = torch.randn((8192, 4096), dtype=torch.float32)
    weight = torch.randn((4096,), dtype=torch.float32)
    bias = torch.randn((4096,), dtype=torch.float32)
    return prepare_comparison(context, weighted_layer_norm, (x, weight, bias, 1.0 / 4096, 1e-6),
        "experiments/cpu/baselines/mojo/modular/normalization/layer_norm/layer_norm_runtime.py", Tolerance(5e-5, 1e-5),
        "Source 使用 Modular/MAX rowwise 同矩统计算法 E[x²]-E[x]²；不是现成 Welford layer_norm。")


def logsumexp(context):
    configure_cpu_budget()
    x = torch.randn((8192, 8192), dtype=torch.float32)
    return prepare_host_comparison(context, row_logsumexp, (x,), "logsumexp", Tolerance(atol=1e-4, rtol=1e-5))


def half_softmax(context):
    configure_cpu_budget()
    x = torch.randn((8192, 8192), dtype=torch.float16)
    return prepare_host_comparison(context, stable_softmax_f16, (x,), "softmax", Tolerance(1e-2, 1e-2))


def bfloat_softmax(context):
    configure_cpu_budget()
    x = torch.randn((8192, 32768), dtype=torch.bfloat16)
    return prepare_host_comparison(context, chunked_softmax_bf16, (x,), "softmax", Tolerance(2e-2, 1e-2))


def mixed_layer_norm(context, definition, dtype, tolerance):
    configure_cpu_budget()
    x = torch.randn((8192, 4096), dtype=dtype)
    weight = torch.randn((4096,), dtype=dtype)
    bias = torch.randn((4096,), dtype=dtype)
    return prepare_host_comparison(context, definition, (x, weight, bias, 1.0 / 4096, 1e-5),
                                   "layer_norm", tolerance)


def plain_rms_norm(context):
    configure_cpu_budget()
    x = torch.randn((8192, 4096), dtype=torch.float32)
    return prepare_host_comparison(context, rms_norm_f32, (x, 1.0 / 4096, 1e-5),
                                   "rms_norm", Tolerance(1e-2, 1e-2))


def bfloat_rms_norm(context):
    configure_cpu_budget()
    x = torch.randn((8192, 4096), dtype=torch.bfloat16)
    weight = torch.randn((4096,), dtype=torch.bfloat16)
    return prepare_host_comparison(context, rms_norm_bf16, (x, weight, 1.0 / 4096, 1e-6),
                                   "weighted_rms_norm", Tolerance(2e-2, 1e-2))


def softmax_backward(context):
    configure_cpu_budget()
    probabilities = torch.randn((4096, 4097), dtype=torch.float32)
    upstream = torch.randn_like(probabilities)
    return prepare_host_comparison(context, backward_definition, (probabilities, upstream),
                                   "softmax_backward", Tolerance(1e-4, 1e-5))


def batch_norm(context):
    x = torch.randn((32, 64, 4096), dtype=torch.float16)
    weight = torch.randn((64,), dtype=torch.float32)
    bias = torch.randn_like(weight)
    initial_mean = torch.randn_like(weight) * 0.1
    initial_variance = torch.rand_like(weight) + 0.5
    epsilon, momentum = 1e-5, 0.1
    report_stage("generated_compilation")
    artifact = intent.compile(batch_norm_training, target=context.target, compiler=context.compiler,
                              tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "experiments/cpu/baselines/pytorch/cpu_runtime.py", "intent_cpu_reference")

    def side(function):
        mean, variance = initial_mean.clone(), initial_variance.clone()
        state = {}

        def prepare():
            mean.copy_(initial_mean)
            variance.copy_(initial_variance)

        def launch():
            state["output"] = function(x, weight, bias, mean, variance, epsilon, momentum)

        return PreparedLaunch(launch, lambda: state["output"], prepare=prepare)

    report_stage("adapter_preparation")
    return PreparedComparison(
        side(artifact.run), side(runtime.batch_norm_training),
        (Tolerance(atol=2e-5), Tolerance(atol=2e-5), Tolerance(atol=1e-2),
         Tolerance(atol=2e-5), Tolerance(atol=2e-5)),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有B32-C64-S4096 f16训练态batch normalization；输出含running mean/variance、归一化值、saved mean/rstd；单NUMA8核，PyTorch CPU reference，完整host调用；运行统计每次恢复且恢复不计时。",
    )


def fused_add_rms_norm_case(context):
    configure_cpu_budget()
    shape = (8192, 4096)
    x = torch.randn(shape, dtype=torch.bfloat16) * 0.5
    residual = torch.randn_like(x) * 0.5
    weight = torch.randn((4096,), dtype=torch.bfloat16)
    return prepare_host_comparison(
        context,
        fused_add_rms_norm,
        (x, residual, weight, 1.0 / 4096, 1.0e-6, 1.0),
        "fused_add_rms_norm",
        (Tolerance(atol=5e-2), Tolerance(atol=5e-2)),
    )


CASES = {"weighted_rms_norm": rms_norm, "stable_softmax": softmax, "weighted_layer_norm": layer_norm,
         "flaggems_logsumexp": logsumexp, "fused_softmax": half_softmax, "chunked_softmax": bfloat_softmax,
         "layer_norm_f16": lambda context: mixed_layer_norm(context, layer_norm_f16, torch.float16, Tolerance(1e-2)),
         "layer_norm_bf16": lambda context: mixed_layer_norm(context, layer_norm_bf16, torch.bfloat16, Tolerance(2e-2, 1e-2)),
         "rms_norm_f32": plain_rms_norm, "rms_norm_bf16": bfloat_rms_norm,
         "flaggems_softmax_backward": softmax_backward, "batch_norm_training": batch_norm,
         "fused_add_rms_norm": fused_add_rms_norm_case}
