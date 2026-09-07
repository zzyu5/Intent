from __future__ import annotations

import cuda.tile as ct
import torch

from kernels.quantization.nvfp4 import nvfp4_quantize as nvfp4_kernel

from ...measurement import compile_single
from ...measurement import functional_launch
from ...model import Context
from ...model import PreparedComparison
from ...model import PreparedLaunch
from ...model import Tolerance
from .common import tilegym_source


def nvfp4_quantize(context: Context) -> PreparedComparison:
    rows, columns = 8192, 4096
    x = torch.randn((rows, columns), device="cuda", dtype=torch.bfloat16)
    global_scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    packed = torch.empty((rows, columns // 2), device="cuda", dtype=torch.uint8)
    row_blocks, column_blocks = rows // 128, columns // 64
    scales = torch.empty(
        (row_blocks, column_blocks, 32, 4, 4), device="cuda", dtype=torch.uint8
    )
    _, generated_base = compile_single(
        context, nvfp4_kernel, (x, global_scale, packed, scales)
    )
    generated = PreparedLaunch(
        generated_base.launch, lambda: (packed, scales.view(-1))
    )
    source_module = tilegym_source(
        context,
        "source/cutile/tilegym/quantization/nvfp4/nvfp4_quantize.py",
        "nvfp4_quantize",
    )
    source_packed = torch.empty_like(packed)
    source_scales = torch.empty_like(scales).view(-1)
    source_kernel = source_module._make_nvfp4_kernel(occupancy=6)

    def launch_source():
        ct.launch(
            torch.cuda.current_stream(),
            (column_blocks, row_blocks, 1),
            source_kernel,
            (
                x, source_scales, source_packed, global_scale,
                column_blocks, row_blocks, columns, rows, False, False,
            ),
        )
        return source_packed, source_scales

    return PreparedComparison(
        generated,
        functional_launch(launch_source),
        Tolerance(atol=0.0),
        cuda_graph=True,
        note="同算法、group16；双方计时均为一次 kernel launch，不含输出分配",
    )


CASES = {"nvfp4_quantize": nvfp4_quantize}
