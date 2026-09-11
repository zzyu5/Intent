import torch
import triton
import triton.language as tl


_N_ELEMENTS = 1 << 20
_MAX_PARTIALS = 256
_FINAL_BLOCK = _MAX_PARTIALS


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 4096, "N_PARTIALS": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 4096, "N_PARTIALS": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 4096, "N_PARTIALS": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 4096, "N_PARTIALS": 256}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK": 8192, "N_PARTIALS": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 8192, "N_PARTIALS": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 8192, "N_PARTIALS": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 8192, "N_PARTIALS": 128}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK": 16384, "N_PARTIALS": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 16384, "N_PARTIALS": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 16384, "N_PARTIALS": 64}, num_warps=16, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _mean_partial(
    input_tensor,
    partial,
    BLOCK: tl.constexpr,
    N_PARTIALS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_tensor + offsets)
    tl.store(partial + pid, tl.sum(values, axis=0))

    if pid == 0:
        tail = tl.arange(0, 256)
        tl.store(partial + tail, 0.0, mask=tail >= N_PARTIALS)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _mean_final(partial, output, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(partial + offsets)
    total = tl.sum(values, axis=0)
    tl.store(output, total * 9.5367431640625e-7)


def launch(input_tensor, output):
    partial = torch.empty(
        (_MAX_PARTIALS,), device=input_tensor.device, dtype=torch.float32
    )
    _mean_partial[lambda META: (META["N_PARTIALS"],)](
        input_tensor,
        partial,
    )
    _mean_final[(1,)](
        partial,
        output,
    )


def run(input_tensor):
    output = torch.empty((), device=input_tensor.device, dtype=torch.float32)
    launch(input_tensor, output)
    return output
