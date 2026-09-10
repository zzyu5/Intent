import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

def _intent_tensor_descriptor_legal(
    tensor, shape, strides, source_rank, flattened_contiguous_axes,
    aligned_stride_axes, unit_stride_axes, require_positive_shape,
    require_positive_strides, alignment, maximum_shape_extent,
):
    if tensor.ndim != source_rank:
        return False
    if tensor.data_ptr() % alignment != 0:
        return False
    if require_positive_shape and any(extent <= 0 for extent in shape):
        return False
    if any(extent > maximum_shape_extent for extent in shape):
        return False
    if require_positive_strides and any(stride <= 0 for stride in strides):
        return False
    if strides[-1] != 1:
        return False
    if any(tensor.stride(axis) != 1 for axis in unit_stride_axes):
        return False
    if any((tensor.stride(axis) * tensor.element_size()) % alignment != 0 for axis in aligned_stride_axes):
        return False
    if any(tensor.stride(axis) != tensor.stride(axis + 1) * tensor.shape[axis + 1] for axis in flattened_contiguous_axes):
        return False
    return True

def _intent_tensor_descriptor_block_shape_legal(
    block_shape, element_size, minimum_contiguous_bytes,
    require_power_of_two, maximum_block_elements,
):
    elements = 1
    for extent in block_shape:
        if extent <= 0 or (require_power_of_two and extent & (extent - 1)):
            return False
        elements *= extent
    return (
        elements <= maximum_block_elements
        and block_shape[-1] * element_size >= minimum_contiguous_bytes
    )

def _intent_prune_tensor_descriptor_configs(configs, named_args, **kwargs):
    if not named_args["TENSOR_DESCRIPTOR_ELIGIBLE"]:
        return [config for config in configs if not config.kwargs["USE_TENSOR_DESCRIPTOR"]]
    retained = []
    for config in configs:
        if not config.kwargs["USE_TENSOR_DESCRIPTOR"]:
            retained.append(config)
            continue
        args = dict(named_args)
        args.update(config.kwargs)
        if _intent_tensor_descriptor_shapes_legal(args):
            retained.append(config)
    return retained

def _intent_tensor_descriptor_shapes_legal(args):
    descriptors = (
        ([1024, 1024], args["output"].element_size(), 16, True, 1048576),
    )
    for contract in descriptors:
        if not _intent_tensor_descriptor_block_shape_legal(*contract):
            return False
    return True

def _intent_tensor_descriptor_allocator(size, alignment, stream):
    buffer = torch.empty(size, dtype=torch.int8, device="cuda")
    if buffer.data_ptr() % alignment != 0:
        raise RuntimeError("Triton descriptor allocator returned a misaligned buffer")
    return buffer

def _intent_host_tensor_descriptor_pre_hook(args):
    if not _intent_tensor_descriptor_shapes_legal(args):
        return
    if not isinstance(args["_intent_descriptor_0"], TensorDescriptor):
        return
    args["_intent_descriptor_0"].block_shape = [1024, 1024]

_intent_tuning_hooks = TuningHooks(("a", "b", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"USE_TENSOR_DESCRIPTOR": 0}, num_warps=1, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 1}, num_warps=1, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 0}, num_warps=2, num_stages=1, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 1}, num_warps=2, num_stages=1, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 0}, num_warps=2, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 1}, num_warps=2, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 0}, num_warps=2, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 1}, num_warps=2, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 0}, num_warps=8, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 1}, num_warps=8, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=1, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=1, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
    ],
    key=["D1", "D2", "D3", "D6", "D7", "D8", "D13", "D14", "S0_0", "S0_1", "S0_2", "S1_0", "S1_1", "S1_2", "S2_0", "S2_1", "TENSOR_DESCRIPTOR_ELIGIBLE"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
    prune_configs_by={"early_config_prune": _intent_prune_tensor_descriptor_configs},
)
@triton.jit
def _intent_kernel(a, b, output, _intent_descriptor_0, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr, D13: tl.constexpr, D14: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, TENSOR_DESCRIPTOR_ELIGIBLE: tl.constexpr, USE_TENSOR_DESCRIPTOR: tl.constexpr):
    v0 = 1024
    v1 = 1024
    v2 = 1024
    v3 = 1024
    v4 = tl.program_id(0)
    v5 = (v4 % 1)
    v6 = (0 + tl.arange(0, 1024) * 1)
    v7 = (0 + tl.arange(0, 8) * 1)
    v8 = (0 + tl.arange(0, 8) * 1)
    v9 = tl.full((1024,), 1024, tl.int64)
    v10 = (v6 < v9)
    v11 = tl.broadcast_to(v10[:, None, None], (1024, 8, 8))
    v12 = tl.full((8,), 8, tl.int64)
    v13 = (v7 < v12)
    v14 = tl.broadcast_to(v13[None, :, None], (1024, 8, 8))
    v15 = (v11 & v14)
    v16 = tl.full((8,), 8, tl.int64)
    v17 = (v8 < v16)
    v18 = tl.broadcast_to(v17[None, None, :], (1024, 8, 8))
    v19 = (v15 & v18)
    v20 = tl.full((1024, 8, 8), 0.0, tl.float16)
    v21 = (0 + tl.arange(0, 8) * 1)
    v22 = (0 + tl.arange(0, 8) * 1)
    v23 = (0 + tl.arange(0, 1024) * 1)
    v24 = tl.full((8,), 8, tl.int64)
    v25 = (v21 < v24)
    v26 = tl.broadcast_to(v25[:, None, None], (8, 8, 1024))
    v27 = tl.full((8,), 8, tl.int64)
    v28 = (v22 < v27)
    v29 = tl.broadcast_to(v28[None, :, None], (8, 8, 1024))
    v30 = (v26 & v29)
    v31 = tl.full((1024,), 1024, tl.int64)
    v32 = (v23 < v31)
    v33 = tl.broadcast_to(v32[None, None, :], (8, 8, 1024))
    v34 = (v30 & v33)
    v35 = tl.full((8, 8, 1024), 0.0, tl.float16)
    v36 = tl.full((1024, 1024), 0.0, tl.float32)
    v37 = tl.load(tl.make_block_ptr(base=(a + tl.cast(0, tl.int64) * S0_0 + tl.cast(0, tl.int64) * S0_1 + tl.cast(0, tl.int64) * S0_2), shape=((1024 - tl.cast(0, tl.int64)), (8 - tl.cast(0, tl.int64)), (8 - tl.cast(0, tl.int64))), strides=(S0_0, S0_1, S0_2), offsets=(0, 0, 0), block_shape=(1024, 8, 8), order=(2, 1, 0)))
    v38 = tl.reshape(v37, (1024, 64), can_reorder=False)
    v39 = tl.load(tl.make_block_ptr(base=(b + tl.cast(0, tl.int64) * S1_0 + tl.cast(0, tl.int64) * S1_1 + tl.cast(0, tl.int64) * S1_2), shape=((8 - tl.cast(0, tl.int64)), (8 - tl.cast(0, tl.int64)), (1024 - tl.cast(0, tl.int64))), strides=(S1_0, S1_1, S1_2), offsets=(0, 0, 0), block_shape=(8, 8, 1024), order=(2, 1, 0)))
    v40 = tl.reshape(v39, (64, 1024), can_reorder=False)
    v41 = tl.dot(v38, v40, v36, input_precision="ieee")
    v42 = tl.cast(v41, tl.float16)
    v43 = (0 + tl.arange(0, 1024) * 1)
    v44 = (0 + tl.arange(0, 1024) * 1)
    v45 = tl.broadcast_to(v42, (1024, 1024))
    v46 = tl.full((1024,), 1024, tl.int64)
    v47 = (v43 < v46)
    v48 = tl.broadcast_to(v47[:, None], (1024, 1024))
    v49 = tl.full((1024,), 1024, tl.int64)
    v50 = (v44 < v49)
    v51 = tl.broadcast_to(v50[None, :], (1024, 1024))
    v52 = (v48 & v51)
    v53 = (0 * S2_0)
    v54 = ((v53 // S2_0) - (((v53 % S2_0) != 0) & (((v53 % S2_0) < 0) != (S2_0 < 0))))
    if USE_TENSOR_DESCRIPTOR:
        _intent_descriptor_0.store([tl.cast(v54, tl.int32), tl.cast(0, tl.int32)], tl.cast(v45, tl.float16))
    else:
        tl.store(tl.make_block_ptr(base=(output + tl.cast(0, tl.int64) * S2_0 + tl.cast(0, tl.int64) * S2_1), shape=((1024 - tl.cast(0, tl.int64)), (1024 - tl.cast(0, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(1024, 1024), order=(1, 0)), tl.cast(v45, tl.float16))

def launch(a, b, output):
    triton.set_allocator(_intent_tensor_descriptor_allocator)
    D1 = a.shape[0]
    D2 = a.shape[1]
    D3 = a.shape[2]
    D6 = b.shape[0]
    D7 = b.shape[1]
    D8 = b.shape[2]
    D13 = output.shape[0]
    D14 = output.shape[1]
    S0_0 = a.stride(0)
    S0_1 = a.stride(1)
    S0_2 = a.stride(2)
    S1_0 = b.stride(0)
    S1_1 = b.stride(1)
    S1_2 = b.stride(2)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    TENSOR_DESCRIPTOR_ELIGIBLE = (_intent_tensor_descriptor_legal(output, [1024, 1024], [S2_0, 1], 2, (), (0,), (1,), True, True, 16, 2147483647))
    _intent_descriptor_0 = (TensorDescriptor(output, shape=[1024, 1024], strides=[S2_0, 1], block_shape=[1, 8], padding="zero") if TENSOR_DESCRIPTOR_ELIGIBLE else output)
    grid = lambda META: (1,)
    return _intent_kernel[grid](a, b, output, _intent_descriptor_0, D1, D2, D3, D6, D7, D8, D13, D14, S0_0, S0_1, S0_2, S1_0, S1_1, S1_2, S2_0, S2_1, TENSOR_DESCRIPTOR_ELIGIBLE)

def run(a, b):
    output = torch.empty((1024, 1024), device=a.device, dtype=torch.float16)
    launch(a, b, output)
    return output
