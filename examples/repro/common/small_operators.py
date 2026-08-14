from __future__ import annotations

import torch
import torch.nn.functional as F

import intent
from intent.targets.base import Target
from kernels.backward.softmax import COLUMNS as SOFTMAX_COLUMNS
from kernels.backward.softmax import ROWS as SOFTMAX_ROWS
from kernels.backward.softmax import softmax_backward
from kernels.factorization.triangular_solve import BATCH as TRIANGULAR_BATCH
from kernels.factorization.triangular_solve import SIZE as TRIANGULAR_SIZE
from kernels.factorization.triangular_solve import batched_lower_triangular_solve
from kernels.normalization.batch_norm import BATCH as BATCH_NORM_BATCH
from kernels.normalization.batch_norm import CHANNELS as BATCH_NORM_CHANNELS
from kernels.normalization.batch_norm import SPATIAL as BATCH_NORM_SPATIAL
from kernels.normalization.batch_norm import batch_norm_training
from kernels.sparse.csr_spmm import COLUMNS as SPMM_COLUMNS
from kernels.sparse.csr_spmm import FEATURES as SPMM_FEATURES
from kernels.sparse.csr_spmm import NONZEROS as SPMM_NONZEROS
from kernels.sparse.csr_spmm import NONZEROS_PER_ROW as SPMM_NONZEROS_PER_ROW
from kernels.sparse.csr_spmm import ROWS as SPMM_ROWS
from kernels.sparse.csr_spmm import csr_spmm
from kernels.vision.max_pool import BATCH as MAX_POOL_BATCH
from kernels.vision.max_pool import CHANNELS as MAX_POOL_CHANNELS
from kernels.vision.max_pool import HEIGHT as MAX_POOL_HEIGHT
from kernels.vision.max_pool import KERNEL_HEIGHT
from kernels.vision.max_pool import PADDING as MAX_POOL_PADDING
from kernels.vision.max_pool import STRIDE as MAX_POOL_STRIDE
from kernels.vision.max_pool import WIDTH as MAX_POOL_WIDTH
from kernels.vision.max_pool import max_pool2d

from .evaluation import Runner
from .evaluation import report_pipeline
from .evaluation import require_close
from .evaluation import run_generated
from .support import prepare_kernel_call


def _run_max_pool2d(compiler: str, target: Target, target_name: str) -> None:
    x = torch.randn(
        (MAX_POOL_BATCH, MAX_POOL_CHANNELS, MAX_POOL_HEIGHT, MAX_POOL_WIDTH),
        device="cuda",
        dtype=torch.float16,
    )
    run_generated(
        definition=max_pool2d,
        arguments=(x,),
        reference=lambda: F.max_pool2d(
            x,
            kernel_size=KERNEL_HEIGHT,
            stride=MAX_POOL_STRIDE,
            padding=MAX_POOL_PADDING,
        ),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="2D max pooling",
        tolerance=0.0,
    )


def _run_softmax_backward(
    compiler: str,
    target: Target,
    target_name: str,
) -> None:
    logits = torch.randn(
        (SOFTMAX_ROWS, SOFTMAX_COLUMNS),
        device="cuda",
        dtype=torch.float32,
    )
    probabilities = torch.softmax(logits, dim=1)
    upstream = torch.randn_like(probabilities)
    run_generated(
        definition=softmax_backward,
        arguments=(probabilities, upstream),
        reference=lambda: probabilities
        * (upstream - (probabilities * upstream).sum(dim=1, keepdim=True)),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="softmax backward",
        tolerance=2.0e-6,
    )


def _run_csr_spmm(compiler: str, target: Target, target_name: str) -> None:
    row_offsets = (
        torch.arange(SPMM_ROWS + 1, device="cuda", dtype=torch.int32)
        * SPMM_NONZEROS_PER_ROW
    )
    column_indices = torch.randint(
        0,
        SPMM_COLUMNS,
        (SPMM_NONZEROS,),
        device="cuda",
        dtype=torch.int32,
    )
    values = torch.randn(
        (SPMM_NONZEROS,), device="cuda", dtype=torch.float32
    ) * 0.1
    dense = torch.randn(
        (SPMM_COLUMNS, SPMM_FEATURES),
        device="cuda",
        dtype=torch.float32,
    )

    def reference() -> torch.Tensor:
        gathered = dense[column_indices.long()].reshape(
            SPMM_ROWS,
            SPMM_NONZEROS_PER_ROW,
            SPMM_FEATURES,
        )
        coefficients = values.reshape(SPMM_ROWS, SPMM_NONZEROS_PER_ROW, 1)
        return (coefficients * gathered).sum(dim=1)

    run_generated(
        definition=csr_spmm,
        arguments=(row_offsets, column_indices, values, dense),
        reference=reference,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="CSR sparse matrix-dense matrix product",
        tolerance=3.0e-5,
        cuda_graph=False,
    )


def _run_batch_norm_training(
    compiler: str,
    target: Target,
    target_name: str,
) -> None:
    x = torch.randn(
        (BATCH_NORM_BATCH, BATCH_NORM_CHANNELS, BATCH_NORM_SPATIAL),
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.randn(
        (BATCH_NORM_CHANNELS,), device="cuda", dtype=torch.float32
    )
    bias = torch.randn_like(weight)
    epsilon = 1.0e-5

    def reference() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = x.float()
        mean = values.mean(dim=(0, 2))
        variance = values.var(dim=(0, 2), unbiased=False)
        rstd = torch.rsqrt(variance + epsilon)
        output = (
            (values - mean[None, :, None])
            * rstd[None, :, None]
            * weight[None, :, None]
            + bias[None, :, None]
        ).half()
        return output, mean, rstd

    run_generated(
        definition=batch_norm_training,
        arguments=(x, weight, bias, epsilon),
        reference=reference,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="training batch normalization",
        tolerance=(1.0e-2, 2.0e-4, 2.0e-4),
        cuda_graph=False,
    )


def _run_triangular_solve(
    compiler: str,
    target: Target,
    target_name: str,
) -> None:
    lower = torch.randn(
        (TRIANGULAR_BATCH, TRIANGULAR_SIZE, TRIANGULAR_SIZE),
        device="cuda",
        dtype=torch.float32,
    ).tril()
    diagonal = torch.arange(TRIANGULAR_SIZE, device="cuda")
    lower[:, diagonal, diagonal] = (
        lower[:, diagonal, diagonal] + TRIANGULAR_SIZE
    )
    right_hand_side = torch.randn(
        (TRIANGULAR_BATCH, TRIANGULAR_SIZE),
        device="cuda",
        dtype=torch.float32,
    )
    expected = torch.linalg.solve_triangular(
        lower,
        right_hand_side.unsqueeze(2),
        upper=False,
    ).squeeze(2)
    solution = right_hand_side.clone()
    artifact = intent.compile(
        batched_lower_triangular_solve,
        target=target,
        compiler=compiler,
    )
    artifact.run(lower, solution)
    torch.cuda.synchronize()
    measured_errors = require_close(
        actual=solution,
        expected=expected,
        tolerance=2.0e-5,
        target_name=target_name,
        kernel_name="batched lower-triangular solve",
    )
    launch = prepare_kernel_call(artifact, (lower, solution), ())
    report_pipeline(
        artifacts=(artifact,),
        launch=launch,
        errors=measured_errors,
        target_name=target_name,
        kernel_name="batched lower-triangular solve",
        prepare=lambda: solution.copy_(right_hand_side),
        performance_scope="kernel-only",
    )


SMALL_OPERATOR_RUNNERS: dict[str, Runner] = {
    "batch_norm_training": _run_batch_norm_training,
    "csr_spmm": _run_csr_spmm,
    "max_pool2d": _run_max_pool2d,
    "softmax_backward": _run_softmax_backward,
    "triangular_solve": _run_triangular_solve,
}
