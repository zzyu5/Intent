import torch

from kernels.clustering.kmeans import CLUSTERS
from kernels.clustering.kmeans import FEATURES
from kernels.clustering.kmeans import POINTS
from kernels.clustering.kmeans import kmeans_assign
from kernels.reduction.boolean import COLUMNS as BOOLEAN_COLUMNS
from kernels.reduction.boolean import ROWS as BOOLEAN_ROWS
from kernels.reduction.boolean import row_boolean_reduction

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


CASES = {
    "boolean_reduction": boolean_reduction,
    "kmeans_assign": kmeans,
}
