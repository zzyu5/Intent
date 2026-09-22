import torch
from kernels.vision.max_pool import max_pool2d
from kernels.vision.max_pool_with_indices import max_pool2d_with_indices
from kernels.vision.nms import greedy_nms
from kernels.vision.roi_align import roi_align_center_sample

from experiments._common.model import Tolerance
from .common import prepare_host_comparison


def roi_align(context):
    batch_size, channels, height, width = 8, 64, 128, 128
    rois_count = 2048
    pooled_height, pooled_width = 7, 7
    feature = torch.randn((batch_size, channels, height, width), dtype=torch.float32)
    batch = torch.randint(0, batch_size, (rois_count, 1)).float()
    upper_left = torch.rand((rois_count, 2)) * (height - 16)
    size = torch.rand((rois_count, 2)) * 48 + 8
    lower_right = torch.minimum(
        upper_left + size,
        torch.tensor([width - 1, height - 1]),
    )
    rois = torch.cat((batch, upper_left, lower_right), dim=1)
    return prepare_host_comparison(context, roi_align_center_sample,
        (feature, rois), "roi_align_center_sample", Tolerance(atol=1.0e-4))


def max_pool(context):
    x = torch.randn((8, 32, 128, 128), dtype=torch.float16)
    return prepare_host_comparison(context, max_pool2d, (x,), "max_pool2d",
        Tolerance(atol=0.0))


def max_pool_with_indices(context):
    x = torch.randn((8, 32, 128, 128), dtype=torch.float16)
    return prepare_host_comparison(
        context,
        max_pool2d_with_indices,
        (x,),
        "max_pool2d_with_indices",
        (Tolerance(atol=0.0), Tolerance(atol=0.0)),
    )


def nms(context):
    upper_left = torch.rand((32, 1024, 2), dtype=torch.float32) * 0.8
    size = torch.rand_like(upper_left) * 0.2 + 0.01
    boxes = torch.cat((upper_left, upper_left + size), dim=2)
    threshold = 0.5
    return prepare_host_comparison(context, greedy_nms, (boxes, threshold),
        "greedy_nms", Tolerance(atol=0.0))


CASES = {"roi_align_center_sample": roi_align, "max_pool2d": max_pool,
         "flaggems_max_pool2d_with_indices": max_pool_with_indices,
         "greedy_nms": nms}
