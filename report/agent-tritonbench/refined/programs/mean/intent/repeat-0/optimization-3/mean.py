import torch
import triton
import triton.language as tl


_N_ELEMENTS = 1 << 20
_PARTIAL_BLOCK = 4096
_N_PARTIALS = _N_ELEMENTS // _PARTIAL_BLOCK
_FINAL_BLOCK = _N_PARTIALS
_INV_N_ELEMENTS = 1.0 / _N_ELEMENTS


@triton.jit
def _mean_partial(input_tensor, partial, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_tensor + offsets)
    tl.store(partial + pid, tl.sum(values, axis=0))


@triton.jit
def _mean_final(partial, output, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(partial + offsets)
    total = tl.sum(values, axis=0)
    tl.store(output, total * _INV_N_ELEMENTS)


def launch(input_tensor, output):
    partial = torch.empty(
        (_N_PARTIALS,), device=input_tensor.device, dtype=torch.float32
    )
    _mean_partial[(_N_PARTIALS,)](
        input_tensor,
        partial,
        BLOCK=_PARTIAL_BLOCK,
        num_warps=8,
        num_stages=2,
    )
    _mean_final[(1,)](
        partial,
        output,
        BLOCK=_FINAL_BLOCK,
        num_warps=4,
        num_stages=2,
    )


def run(input_tensor):
    output = torch.empty((), device=input_tensor.device, dtype=torch.float32)
    launch(input_tensor, output)
    return output
