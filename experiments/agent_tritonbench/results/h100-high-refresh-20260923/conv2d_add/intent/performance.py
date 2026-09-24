import torch
import intent
import intent.language as I


@intent.kernel
def _conv2d_add_kernel(
    input: I.In[I.f32, (16, 32, 32, 32)],
    weight: I.In[I.f32, (32, 32, 3, 3)],
    bias: I.In[I.f32, (32,)],
    other: I.In[I.f32, (16, 32, 32, 32)],
    out: I.Out[I.f32, (16, 32, 32, 32)],
    alpha: I.f32,
    other_scalar: I.f32,
    HAS_BIAS: I.Constexpr[bool],
    HAS_OTHER: I.Constexpr[bool],
    OTHER_TENSOR: I.Constexpr[bool],
):
    n = I.domain(0, 16)
    oc = I.domain(0, 32)
    oh = I.domain(0, 32)
    ow = I.domain(0, 32)
    ci = I.domain(0, 32)
    kh = I.domain(0, 3)
    kw = I.domain(0, 3)

    # The six free/reduction coordinates form one logical convolution window.
    n6 = I.reshape(I.indices(n), (16, 1, 1, 1, 1, 1))
    oh6 = I.reshape(I.indices(oh), (1, 32, 1, 1, 1, 1))
    ow6 = I.reshape(I.indices(ow), (1, 1, 32, 1, 1, 1))
    ci6 = I.reshape(I.indices(ci), (1, 1, 1, 32, 1, 1))
    kh6 = I.reshape(I.indices(kh), (1, 1, 1, 1, 3, 1))
    kw6 = I.reshape(I.indices(kw), (1, 1, 1, 1, 1, 3))

    ih = oh6 + kh6 - 1
    iw = ow6 + kw6 - 1
    valid = (ih >= 0) & (ih < 32) & (iw >= 0) & (iw < 32)
    lhs = I.gather(
        input,
        (n6, ci6, ih, iw),
        valid=valid,
        fill=I.cast(0.0, I.f32),
    )

    ocw = I.reshape(I.indices(oc), (32, 1, 1, 1))
    ciw = I.reshape(I.indices(ci), (1, 32, 1, 1))
    khw = I.reshape(I.indices(kh), (1, 1, 3, 1))
    kww = I.reshape(I.indices(kw), (1, 1, 1, 3))
    rhs = weight[ocw, ciw, khw, kww]

    # lhs is [N, OH, OW, CI, KH, KW], rhs is [OC, CI, KH, KW].
    # The contraction therefore produces [N, OH, OW, OC].
    result = I.contract(
        lhs,
        rhs,
        reduce=((3, 1), (4, 2), (5, 3)),
        batch=(),
        acc_dtype=I.f32,
    )
    result = I.transpose(result, (0, 3, 1, 2))

    n4 = I.reshape(I.indices(n), (16, 1, 1, 1))
    oc4 = I.reshape(I.indices(oc), (1, 32, 1, 1))
    oh4 = I.reshape(I.indices(oh), (1, 1, 32, 1))
    ow4 = I.reshape(I.indices(ow), (1, 1, 1, 32))

    if HAS_BIAS:
        result = result + bias[oc4]
    if HAS_OTHER:
        if OTHER_TENSOR:
            result = result + alpha * other[n4, oc4, oh4, ow4]
        else:
            result = result + alpha * other_scalar

    out[n4, oc4, oh4, ow4] = result


@intent.kernel
def _copy_output_kernel(
    source: I.In[I.f32, (16, 32, 32, 32)],
    destination: I.Out[I.f32, (16, 32, 32, 32)],
):
    n = I.domain(0, 16)
    oc = I.domain(0, 32)
    oh = I.domain(0, 32)
    ow = I.domain(0, 32)
    n4 = I.reshape(I.indices(n), (16, 1, 1, 1))
    oc4 = I.reshape(I.indices(oc), (1, 32, 1, 1))
    oh4 = I.reshape(I.indices(oh), (1, 1, 32, 1))
    ow4 = I.reshape(I.indices(ow), (1, 1, 1, 32))
    destination[n4, oc4, oh4, ow4] = source[n4, oc4, oh4, ow4]


def build(context):
    artifacts = {}
    for has_bias in (False, True):
        artifacts[(has_bias, False, False)] = context.compile(
            "conv2d_add_no_other_bias_%d" % int(has_bias),
            _conv2d_add_kernel,
            constexprs={
                "HAS_BIAS": has_bias,
                "HAS_OTHER": False,
                "OTHER_TENSOR": False,
            },
        )
        artifacts[(has_bias, True, True)] = context.compile(
            "conv2d_add_tensor_other_bias_%d" % int(has_bias),
            _conv2d_add_kernel,
            constexprs={
                "HAS_BIAS": has_bias,
                "HAS_OTHER": True,
                "OTHER_TENSOR": True,
            },
        )
        artifacts[(has_bias, True, False)] = context.compile(
            "conv2d_add_scalar_other_bias_%d" % int(has_bias),
            _conv2d_add_kernel,
            constexprs={
                "HAS_BIAS": has_bias,
                "HAS_OTHER": True,
                "OTHER_TENSOR": False,
            },
        )
    copy_artifact = context.compile("conv2d_add_copy_output", _copy_output_kernel)

    def conv2d_add(
        input,
        weight,
        bias=None,
        other=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        alpha=1,
        out=None,
    ):
        stride_pair = (stride, stride) if isinstance(stride, int) else tuple(stride)
        dilation_pair = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        padding_pair = (padding, padding) if isinstance(padding, int) else tuple(padding)
        if stride_pair != (1, 1) or dilation_pair != (1, 1) or padding_pair != (1, 1) or groups != 1:
            raise ValueError("candidate is specialized for stride=1, padding=1, dilation=1, groups=1")

        if out is None:
            out = torch.empty((16, 32, 32, 32), device=input.device, dtype=input.dtype)

        # A separate destination avoids write-after-read hazards when out is input.
        aliases_input = out.data_ptr() == input.data_ptr()
        launch_out = out
        if aliases_input:
            launch_out = torch.empty((16, 32, 32, 32), device=input.device, dtype=input.dtype)

        has_bias = bias is not None
        if other is None:
            has_other = False
            other_tensor = False
            other_arg = input
            other_scalar = 0.0
        elif isinstance(other, torch.Tensor):
            has_other = True
            other_tensor = True
            other_arg = other
            other_scalar = 0.0
        else:
            has_other = True
            other_tensor = False
            other_arg = input
            other_scalar = float(other)

        bias_arg = bias if has_bias else weight[:, 0, 0, 0]
        artifact = artifacts[(has_bias, has_other, other_tensor)]
        artifact(
            input,
            weight,
            bias_arg,
            other_arg,
            launch_out,
            float(alpha),
            other_scalar,
        )
        if aliases_input:
            copy_artifact(launch_out, out)
        return out

    return conv2d_add
