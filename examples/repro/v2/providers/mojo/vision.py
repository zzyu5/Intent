import torch

from kernels.vision.roi_align import roi_align_center_sample

from ...model import Tolerance
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


CASES = {"roi_align_center_sample": roi_align}
