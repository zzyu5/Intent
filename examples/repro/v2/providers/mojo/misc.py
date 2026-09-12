import intent
import torch

from kernels.clustering.kmeans import CLUSTERS
from kernels.clustering.kmeans import FEATURES
from kernels.clustering.kmeans import POINTS
from kernels.clustering.kmeans import kmeans_assign
from kernels.reduction.boolean import COLUMNS as BOOLEAN_COLUMNS
from kernels.reduction.boolean import ROWS as BOOLEAN_ROWS
from kernels.reduction.boolean import row_boolean_reduction
from kernels.reduction.two_pass import partitioned_max_partial, partitioned_max_reduce

from ...loading import load_module
from ...measurement import report_stage
from ...model import PreparedComparison, PreparedLaunch
from ...model import Tolerance
from .common import prepare_host_comparison


def boolean_reduction(context):
    shape_source = torch.empty(
        (BOOLEAN_ROWS, BOOLEAN_COLUMNS), dtype=torch.float32
    )
    return prepare_host_comparison(
        context,
        row_boolean_reduction,
        (shape_source,),
        "row_boolean_reduction",
        Tolerance(atol=0.0),
    )


def kmeans(context):
    points = torch.randn((POINTS, FEATURES), dtype=torch.float16) * 0.1
    centroids = torch.randn((CLUSTERS, FEATURES), dtype=torch.float16) * 0.1
    return prepare_host_comparison(
        context,
        kmeans_assign,
        (points, centroids),
        "kmeans_assign",
        (Tolerance(atol=0.0), Tolerance(atol=2.0e-4)),
    )


def partitioned_max(context):
    x = -torch.rand((128, 257), dtype=torch.float32)
    report_stage("generated_compilation")
    partial = intent.compile(partitioned_max_partial, target=context.target, compiler=context.compiler,
                             tuning_config=context.tuning_config)
    reduce = intent.compile(partitioned_max_reduce, target=context.target, compiler=context.compiler,
                            tuning_config=context.tuning_config)
    runtime = load_module(context.project_root / "source/pytorch/cpu_runtime.py", "intent_cpu_reference")
    state = {}

    def generated_launch():
        workspace = torch.full((128, 300), -torch.inf, dtype=torch.float32)
        partial(x, workspace)
        state["generated"] = reduce.run(workspace)

    def source_launch():
        state["source"] = runtime.partitioned_max(x)

    report_stage("adapter_preparation")
    return PreparedComparison(
        PreparedLaunch(generated_launch, lambda: state["generated"]),
        PreparedLaunch(source_launch, lambda: state["source"]), Tolerance(atol=0.0),
        cuda_graph=False, device_type="cpu", cpu_host_timing=True,
        note="既有M128-N257-P300两阶段max，包含空子区域；原负输入、exact输出；单NUMA8核，完整host含workspace分配/初始化及两kernel同步，PyTorch amax reference。",
    )


CASES = {
    "boolean_reduction": boolean_reduction,
    "kmeans_assign": kmeans,
    "partitioned_two_pass_max": partitioned_max,
}
