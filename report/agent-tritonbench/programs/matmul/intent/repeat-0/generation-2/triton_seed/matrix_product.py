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
        ([args["FRAGMENT_D1"], args["FRAGMENT_D3"]], args["output"].element_size(), 16, True, 1048576),
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
    args["_intent_descriptor_0"].block_shape = [args["FRAGMENT_D1"], args["FRAGMENT_D3"]]

_intent_tuning_hooks = TuningHooks(("input", "other", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=8, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=8, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=4, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=4, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=8, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=8, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=4, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 128, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=4, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=8, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=8, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=4, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=4, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=8, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=8, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=4, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 64, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=4, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 128, "FRAGMENT_D3": 256, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 128, "FRAGMENT_D3": 256, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=2, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 128, "FRAGMENT_D3": 256, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=8, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 128, "FRAGMENT_D3": 256, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=8, num_stages=3, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 128, "FRAGMENT_D3": 256, "USE_TENSOR_DESCRIPTOR": 0}, num_warps=4, num_stages=4, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 128, "FRAGMENT_D3": 256, "USE_TENSOR_DESCRIPTOR": 1}, num_warps=4, num_stages=4, num_ctas=1, pre_hook=_intent_host_tensor_descriptor_pre_hook),
    ],
    key=["D1", "D2", "D3", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1", "TENSOR_DESCRIPTOR_ELIGIBLE"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
    prune_configs_by={"early_config_prune": _intent_prune_tensor_descriptor_configs},
)
@triton.jit
def _intent_kernel(input, other, output, _intent_descriptor_0, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, TENSOR_DESCRIPTOR_ELIGIBLE: tl.constexpr, USE_TENSOR_DESCRIPTOR: tl.constexpr, BLOCK_K_1_2: tl.constexpr, FRAGMENT_D3: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = D1
    v1 = D3
    v2 = FRAGMENT_D1
    v3 = FRAGMENT_D3
    v4 = (FRAGMENT_D1 - 1)
    v5 = (D1 + v4)
    v6 = ((v5 // FRAGMENT_D1) - (((v5 % FRAGMENT_D1) != 0) & (((v5 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v7 = (FRAGMENT_D3 - 1)
    v8 = (D3 + v7)
    v9 = ((v8 // FRAGMENT_D3) - (((v8 % FRAGMENT_D3) != 0) & (((v8 % FRAGMENT_D3) < 0) != (FRAGMENT_D3 < 0))))
    v10 = tl.program_id(0)
    v11 = tl.program_id(1)
    v12 = (v10 * v9)
    v13 = (v12 + v11)
    v14 = ((v13 // v9) % v6)
    v15 = (v13 % v9)
    v16 = (v14 * FRAGMENT_D1)
    v17 = (v16 * 1)
    v18 = (0 + v17)
    v19 = (v18 + tl.arange(0, FRAGMENT_D1) * 1)
    v20 = D1
    v21 = (v19 < v20)
    v22 = D1
    v23 = tl.full((FRAGMENT_D1,), v22, tl.int64)
    v24 = (v19 < v23)
    v25 = D2
    v26 = (v15 * FRAGMENT_D3)
    v27 = (v26 * 1)
    v28 = (0 + v27)
    v29 = (v28 + tl.arange(0, FRAGMENT_D3) * 1)
    v30 = D3
    v31 = (v29 < v30)
    v32 = D2
    v33 = D3
    v34 = tl.full((FRAGMENT_D3,), v33, tl.int64)
    v35 = (v29 < v34)
    v36 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float32)
    v37 = v36
    for iv38 in range(0, D2, BLOCK_K_1_2):
        v39 = (iv38 + tl.arange(0, BLOCK_K_1_2) * 1)
        v40 = (iv38 - 0)
        v41 = (0 + v40)
        v42 = (v41 + tl.arange(0, BLOCK_K_1_2) * 1)
        v43 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v44 = (v39 < v43)
        v45 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v46 = (v42 < v45)
        v47 = tl.full((BLOCK_K_1_2,), v25, tl.int64)
        v48 = (v39 < v47)
        v49 = v48[None, :]
        v50 = v24[:, None]
        v51 = (v50 & v49)
        v52 = v21[:, None]
        v53 = (v51 & v52)
        v54 = v44[None, :]
        v55 = (v53 & v54)
        v56 = tl.full((FRAGMENT_D1, BLOCK_K_1_2), 0.0, tl.float16)
        v57 = tl.full((BLOCK_K_1_2,), v32, tl.int64)
        v58 = (v42 < v57)
        v59 = v58[:, None]
        v60 = v35[None, :]
        v61 = (v59 & v60)
        v62 = v31[None, :]
        v63 = (v61 & v62)
        v64 = v46[:, None]
        v65 = (v63 & v64)
        v66 = tl.full((BLOCK_K_1_2, FRAGMENT_D3), 0.0, tl.float16)
        v67 = tl.load(tl.make_block_ptr(base=(other + tl.cast(v41, tl.int64) * S1_0 + tl.cast(v28, tl.int64) * S1_1), shape=((D2 - tl.cast(v41, tl.int64)), (D3 - tl.cast(v28, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(BLOCK_K_1_2, FRAGMENT_D3), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v68 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v18, tl.int64) * S0_0 + tl.cast(iv38, tl.int64) * S0_1), shape=((D1 - tl.cast(v18, tl.int64)), (D2 - tl.cast(iv38, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v69 = tl.dot(v68, v67, v37, input_precision="ieee")
        v37 = v69
    v70 = tl.cast(v37, tl.float16)
    v71 = FRAGMENT_D1
    v72 = (0 + tl.arange(0, FRAGMENT_D1) * 1)
    v73 = FRAGMENT_D3
    v74 = (0 + tl.arange(0, FRAGMENT_D3) * 1)
    v75 = tl.broadcast_to(v70, (FRAGMENT_D1, FRAGMENT_D3, ))
    v76 = tl.full((FRAGMENT_D1,), v0, tl.int64)
    v77 = (v72 < v76)
    v78 = v77[:, None]
    v79 = tl.full((FRAGMENT_D3,), v1, tl.int64)
    v80 = (v74 < v79)
    v81 = v80[None, :]
    v82 = (v78 & v81)
    v83 = (0 * S2_0)
    v84 = ((v83 // S2_0) - (((v83 % S2_0) != 0) & (((v83 % S2_0) < 0) != (S2_0 < 0))))
    if USE_TENSOR_DESCRIPTOR:
        _intent_descriptor_0.store([tl.cast(v84, tl.int32), tl.cast(0, tl.int32)], tl.cast(v75, tl.float16))
    else:
        tl.store(tl.make_block_ptr(base=(output + tl.cast(0, tl.int64) * S2_0 + tl.cast(0, tl.int64) * S2_1), shape=((D1 - tl.cast(0, tl.int64)), (D3 - tl.cast(0, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D3), order=(1, 0)), tl.cast(v75, tl.float16), boundary_check=(0, 1))

def launch(input, other, output):
    triton.set_allocator(_intent_tensor_descriptor_allocator)
    D1 = input.shape[0]
    D2 = input.shape[1]
    D3 = other.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = other.stride(0)
    S1_1 = other.stride(1)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    TENSOR_DESCRIPTOR_ELIGIBLE = (_intent_tensor_descriptor_legal(output, [D1, D3], [S2_0, 1], 2, (), (0,), (1,), True, True, 16, 2147483647))
    _intent_descriptor_0 = (TensorDescriptor(output, shape=[D1, D3], strides=[S2_0, 1], block_shape=[1, 8], padding="zero") if TENSOR_DESCRIPTOR_ELIGIBLE else output)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]), triton.cdiv(D3, META["FRAGMENT_D3"]))
    return _intent_kernel[grid](input, other, output, _intent_descriptor_0, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1, TENSOR_DESCRIPTOR_ELIGIBLE)

def run(input, other):
    output = torch.empty((input.shape[0], other.shape[1]), device=input.device, dtype=torch.float16)
    launch(input, other, output)
    return output
