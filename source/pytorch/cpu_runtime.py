import numpy as np
import torch
import torch.nn.functional as F


def philox_uniform(seed, counters):
    counter = np.asarray(counters, dtype=np.uint64)
    block = counter >> np.uint64(2)
    mask = np.uint64(0xFFFFFFFF)
    c0, c1 = block & mask, block >> np.uint64(32)
    c2, c3 = np.zeros_like(block), np.zeros_like(block)
    k0, k1 = seed & 0xFFFFFFFF, seed >> 32
    for _ in range(10):
        p0 = c0 * np.uint64(0xD2511F53)
        p1 = c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ np.uint64(k0), p1 & mask,
                          (p0 >> np.uint64(32)) ^ c3 ^ np.uint64(k1), p0 & mask)
        k0 = (k0 + 0x9E3779B9) & 0xFFFFFFFF
        k1 = (k1 + 0xBB67AE85) & 0xFFFFFFFF
    word = counter & np.uint64(3)
    bits = c3.copy()
    for index, values in enumerate((c0, c1, c2)):
        np.copyto(bits, values, where=word == index)
    return torch.from_numpy((bits >> np.uint64(8)).astype(np.float32) * np.float32(2.0**-24))


def dropout_residual_rms_norm(x, residual, weight, dnormalized, dresidual_out,
                              seed, keep_probability, inverse_keep_probability,
                              inverse_features, epsilon, weight_offset):
    counters = np.arange(x.numel(), dtype=np.uint64).reshape(x.shape)
    keep = (philox_uniform(seed, counters) < keep_probability).float()
    summed = (x.float() * keep / keep_probability + residual.float()).to(x.dtype)
    values = summed.float()
    inverse_rms = torch.rsqrt(values.square().sum(dim=1, keepdim=True) * inverse_features + epsilon)
    normalized = (values * inverse_rms * (weight.float() + weight_offset)).to(x.dtype)
    gradient = dnormalized.float() * (weight.float() + weight_offset)
    projection = (gradient * values).sum(dim=1, keepdim=True) * inverse_features
    summed_gradient = gradient * inverse_rms - values * inverse_rms.pow(3) * projection + dresidual_out.float()
    dx = (summed_gradient * keep / keep_probability).to(x.dtype)
    return normalized, summed, dx, summed_gradient.to(x.dtype)


def barrier_option_paths(seed, initial_price, strike, barrier, drift, volatility):
    counters = np.arange(262144 * 64, dtype=np.uint64).reshape(262144, 64)
    uniform = philox_uniform(seed, counters)
    price = torch.full((262144,), initial_price, dtype=torch.float32)
    knocked_out = torch.zeros((262144,), dtype=torch.bool)
    for step in range(64):
        direction = torch.where(uniform[:, step] >= 0.5, volatility, -volatility)
        next_price = price * torch.exp(torch.tensor(drift, dtype=torch.float32) + direction)
        price = torch.where(knocked_out, price, next_price)
        knocked_out |= price >= barrier
    return torch.where(knocked_out, torch.zeros_like(price), torch.clamp_min(price - strike, 0.0))


def gelu(x):
    return F.gelu(x, approximate="tanh")


def fp8_groupwise_quantize(x, scales):
    values = x.float().reshape(x.shape[0], -1, 128)
    scale = values.abs().amax(dim=-1).clamp_min(1e-12) / 448.0
    quantized = (values / scale[:, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    scales.copy_(scale)
    return quantized.reshape_as(x), scales


def partitioned_max(x):
    return torch.amax(x, dim=1)


def relu(x):
    return torch.maximum(x, x.new_zeros(()))


def silu_and_mul_packed(packed):
    half = packed.shape[1] // 2
    return swiglu(packed[:, :half], packed[:, half:])


def geglu_tanh(x, output):
    half = output.shape[1]
    output.copy_((x[:, :half].float() * F.gelu(x[:, half:].float(), approximate="tanh")).to(x.dtype))
    return output


def fp8_matmul(lhs, rhs_transposed):
    return (lhs.float() @ rhs_transposed.float().T).to(lhs.dtype)


def quantized_gemm(lhs, rhs, bias, residual, scale):
    accumulator = lhs.float() @ rhs.float()
    fused = torch.relu(accumulator + bias) + residual.float()
    return (fused / scale).clamp(-128.0, 127.0).to(torch.int8)


def weight_only_int4_matmul(activation, packed, scales):
    positions = torch.arange(activation.shape[1])
    nibble = (packed[positions // 8] >> ((positions % 8) * 4)[:, None]) & 15
    signed = nibble - ((nibble & 8) << 1)
    dequantized = signed.float() * scales.float().repeat_interleave(64, dim=0)
    return (activation.float() @ dequantized).half()


def f32_groupwise_fp8_quantize(x, scales):
    values = x.reshape(x.shape[0], -1, 128)
    scale = values.abs().amax(dim=-1).clamp_min(1e-4) / 448.0
    scales.copy_(scale)
    output = (values / scale[:, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape_as(x)
    return output, scales


def qkv_projection(x, weights):
    values = x.float()
    return torch.stack(tuple((values @ weight.float()).to(x.dtype) for weight in weights))


def flash_attention_bf16_fwd(query, key, value, scale):
    group = query.shape[1] // key.shape[1]
    keys = key.repeat_interleave(group, dim=1).float()
    values = value.repeat_interleave(group, dim=1).float()
    return F.scaled_dot_product_attention(query.float(), keys, values, is_causal=True, scale=scale).to(query.dtype)


def swiglu(gate, up):
    values = gate.float()
    sigmoid = 1.0 / (1.0 + torch.exp(-values))
    sigmoid = torch.where(sigmoid < torch.finfo(torch.float32).tiny, sigmoid * 0.0, sigmoid)
    return (values * sigmoid).to(gate.dtype) * up


def swiglu_backward(dc, a, b):
    dc_f32 = dc.float()
    a_f32 = a.float()
    b_f32 = b.float()
    sigmoid = torch.sigmoid(a_f32)
    silu = a_f32 * sigmoid
    da = (dc_f32 * (silu * (1.0 - sigmoid) + sigmoid) * b_f32).to(torch.bfloat16)
    db = (dc_f32 * silu).to(torch.bfloat16)
    return da, db


def addcmul(x, scale, bias):
    return bias.unsqueeze(-1) + x * scale.unsqueeze(-1)


def logsumexp(x):
    return torch.logsumexp(x, dim=-1)


def softmax(x):
    values = x.float()
    numerator = torch.exp(values - values.amax(dim=-1, keepdim=True))
    return (numerator / numerator.sum(dim=-1, keepdim=True)).to(x.dtype)


def layer_norm(x, weight, bias, inverse_features, epsilon):
    values = x.float()
    mean = values.sum(dim=-1, keepdim=True) * inverse_features
    centered = values - mean
    variance = (centered * centered).sum(dim=-1, keepdim=True) * inverse_features
    return (centered * torch.rsqrt(variance + epsilon) * weight.float() + bias.float()).to(x.dtype)


def rms_norm(x, inverse_features, epsilon):
    mean_square = (x * x).sum(dim=-1, keepdim=True) * inverse_features
    return x * torch.rsqrt(mean_square + epsilon)


def weighted_rms_norm(x, weight, inverse_features, epsilon):
    values = x.float()
    return (rms_norm(values, inverse_features, epsilon) * weight.float()).to(x.dtype)


def fused_add_rms_norm(x, residual, weight, inverse_features, epsilon, weight_offset):
    summed = (x.float() + residual.float()).to(x.dtype)
    summed_f32 = summed.float()
    inverse_rms = torch.rsqrt(
        summed_f32.square().sum(dim=1, keepdim=True) * inverse_features + epsilon
    )
    normalized = (
        summed_f32 * inverse_rms * (weight.float() + weight_offset)
    ).to(x.dtype)
    return normalized, summed


def softmax_backward(probabilities, upstream):
    projection = (probabilities * upstream).sum(dim=-1, keepdim=True)
    return probabilities * (upstream - projection)


def fused_cross_entropy(logits, labels):
    loss = F.cross_entropy(logits, labels.long(), reduction="none", ignore_index=-100)
    valid = labels != -100
    prediction = torch.where(valid, logits.argmax(dim=1), -1).to(torch.int32)
    gradient = torch.softmax(logits, dim=1)
    safe_labels = torch.where(valid, labels, 0).long()
    gradient[torch.arange(logits.shape[0]), safe_labels] -= valid.float()
    gradient *= valid[:, None].float()
    logits.copy_(gradient)
    return logits, loss, prediction


def fused_cross_entropy_bf16(logits, labels):
    values = logits.float()
    loss = F.cross_entropy(values, labels, reduction="none", ignore_index=-100)
    valid = labels != -100
    prediction = torch.where(valid, values.argmax(dim=1), -1)
    gradient = torch.softmax(values, dim=1)
    safe_labels = torch.where(valid, labels, 0)
    gradient[torch.arange(logits.shape[0]), safe_labels] -= valid.float()
    gradient *= valid[:, None].float()
    logits.copy_(gradient)
    return logits, loss, prediction


def flash_cross_entropy_bf16(logits, labels):
    values = logits.float()
    valid = labels != -100
    safe_labels = torch.where(valid, labels, 0)
    lse = torch.logsumexp(values, dim=1)
    regularizer = (1e-4 * lse) * lse
    target = values[torch.arange(logits.shape[0]), safe_labels]
    return torch.where(valid, lse - target + regularizer, 0.0), torch.where(valid, regularizer, 0.0)


def group_norm_silu_backward(x, upstream, weight, bias, mean, rstd, dweight, dbias, inverse_group_elements):
    values = x.float()
    normalized = ((values.reshape(32, 32, 8, 1024) - mean[:, :, None, None])
                  * rstd[:, :, None, None]).reshape_as(values)
    affine = normalized * weight[None, :, None] + bias[None, :, None]
    sigmoid = torch.sigmoid(affine)
    gradient = upstream.float() * (sigmoid + affine * sigmoid * (1.0 - sigmoid))
    dx, dw, db = torch.ops.aten.native_group_norm_backward(
        gradient, values, mean, rstd, weight, 32, 256, 1024, 32, [True, True, True])
    dweight.add_(dw)
    dbias.add_(db)
    return dx.to(x.dtype), dweight, dbias


def causal_conv1d(x, weight, bias):
    values = F.conv1d(F.pad(x.float(), (weight.shape[1] - 1, 0)),
                      weight[:, None, :].float(), bias.float(), groups=x.shape[1])
    return F.silu(values).to(x.dtype)


def conv2d(x, weight):
    return F.conv2d(x[:, None].float(), weight[None, None].float(), padding=1)[:, 0].to(x.dtype)


def causal_conv1d_backward(x, weight, grad_output):
    batch, channels, length = x.shape
    width = weight.shape[1]
    gradient = grad_output.float()
    padded = F.pad(x.float(), (width - 1, 0))
    dx = torch.nn.grad.conv1d_input(padded.shape, weight[:, None].float(), gradient, groups=channels)
    dw = torch.nn.grad.conv1d_weight(padded, (channels, 1, width), gradient, groups=channels)
    db = gradient.sum(dim=(0, 2))
    return dx[:, :, width - 1:].to(x.dtype), dw[:, 0], db


def transpose(x):
    return x.T.contiguous()


def shifted_row_copy(x):
    return torch.cat((x[1:], x[:1]), dim=0)


def roll_rows_forward(x):
    return x[(torch.arange(x.numel()) + 4096) % x.numel()]


def grouped_query_head_add(query, key):
    heads = torch.arange(query.shape[0]) // (query.shape[0] // key.shape[0])
    return query + key[heads]


def alternating_signed_indices(shape_source):
    rows, columns = shape_source.shape
    signs = torch.where(torch.arange(rows, dtype=torch.int32) % 2 == 0, 1, -1).to(torch.int32)
    return signs[:, None] * torch.arange(columns, dtype=torch.int32)[None, :]


def row_boolean_reduction(shape_source):
    return torch.full((shape_source.shape[0],), 3, dtype=torch.int32)


def kmeans_assign(points, centroids):
    distances = (points.float().square().sum(dim=1, keepdim=True)
                 + centroids.float().square().sum(dim=1)[None, :]
                 - 2.0 * points.float() @ centroids.float().T)
    minimum, assignment = distances.min(dim=1)
    return assignment.to(torch.int32), minimum


def scalar_table_lookup(labels, table):
    return table[labels.long()]


def embedding_forward_lookup(embedding_table, indices):
    return embedding_table[indices.long()]


def max_pool2d(x):
    return F.max_pool2d(x, kernel_size=3, stride=2, padding=1)


def max_pool2d_with_indices(x):
    return F.max_pool2d(x, kernel_size=3, stride=2, padding=1, return_indices=True)


def integer_log2_floor(values):
    value = values.clone()
    result = torch.zeros_like(values)
    while bool((value > 1).any()):
        active = value > 1
        value = torch.where(active, torch.div(value, 2, rounding_mode="floor"), value)
        result = result + active.to(result.dtype)
    return result


def paired_sum_product(x, y):
    return (x + y) + (x * y)


def rope_qk(query, key, cosine, sine):
    half = 64
    query_first = query[..., :half].clone()
    query_second = query[..., half:].clone()
    key_first = key[..., :half].clone()
    key_second = key[..., half:].clone()
    cos = cosine[..., :half]
    sin = sine[..., :half]
    query[..., :half] = query_first * cos - query_second * sin
    query[..., half:] = query_second * cos + query_first * sin
    key[..., :half] = key_first * cos - key_second * sin
    key[..., half:] = key_second * cos + key_first * sin
    return query, key


def rope_qk_partial(query, key, cosine, sine):
    half = 32
    cos, sin = cosine[..., :half].float(), sine[..., :half].float()
    for values in (query, key):
        first = values[..., :half].float()
        second = values[..., half:2 * half].float()
        values[..., :half] = first * cos - second * sin
        values[..., half:2 * half] = second * cos + first * sin
    return query, key


def rotary_embedding_bf16(values, cosine, sine):
    first, second = values[..., :64].float(), values[..., 64:].float()
    cos, sin = cosine[None, :, None, :], sine[None, :, None, :]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(values.dtype)


def rope_qk_bf16(query, key, cosine, sine):
    cos, sin = cosine[..., :64].float(), sine[..., :64].float()
    for values in (query, key):
        first, second = values[..., :64].float(), values[..., 64:].float()
        values[..., :64] = first * cos - second * sin
        values[..., 64:] = second * cos + first * sin
    return query, key


def greedy_nms(boxes, threshold):
    keep = torch.zeros((32, 1024), dtype=torch.bool, device=boxes.device)
    for batch in range(32):
        suppressed = torch.zeros((1024,), dtype=torch.bool, device=boxes.device)
        for candidate in range(1024):
            if suppressed[candidate]:
                continue
            keep[batch, candidate] = True
            candidate_box = boxes[batch, candidate]
            remaining = boxes[batch, candidate + 1 :]
            if remaining.numel() == 0:
                continue
            upper_left = torch.maximum(candidate_box[:2], remaining[:, :2])
            lower_right = torch.minimum(candidate_box[2:], remaining[:, 2:])
            intersection = (lower_right - upper_left).clamp_min(0.0).prod(dim=1)
            candidate_area = (candidate_box[2:] - candidate_box[:2]).prod()
            remaining_area = (remaining[:, 2:] - remaining[:, :2]).prod(dim=1)
            overlap = intersection / (candidate_area + remaining_area - intersection)
            suppressed[candidate + 1 :] |= overlap > threshold
    return keep


def embedding_backward_atomic(indices, grad_output, grad_weight):
    grad_weight.index_add_(0, indices.long(), grad_output)
    return grad_weight


def claim_zero_slots(state):
    previous = state.clone()
    state.copy_(torch.where(state == 0, torch.ones_like(state), state))
    return state, previous


def cumsum(x):
    return torch.cumsum(x, dim=1)


def batch_norm_training(x, weight, bias, running_mean, running_variance, epsilon, momentum):
    values = x.float()
    variance, mean = torch.var_mean(values, dim=(0, 2), unbiased=False)
    rstd = torch.rsqrt(variance + epsilon)
    output = ((values - mean[None, :, None]) * rstd[None, :, None]
              * weight[None, :, None] + bias[None, :, None]).to(x.dtype)
    count = x.shape[0] * x.shape[2]
    running_mean.copy_((1.0 - momentum) * running_mean + momentum * mean)
    running_variance.copy_((1.0 - momentum) * running_variance
                           + momentum * variance * count / (count - 1))
    return running_mean, running_variance, output, mean, rstd


def histogram(samples):
    return torch.bincount(samples.to(torch.int64), minlength=256).to(torch.int32)


def nucleus(probabilities, threshold):
    cumulative = probabilities.cumsum(dim=1)
    cutoff = (cumulative < threshold).sum(dim=1, dtype=torch.int32) + 1
    return cumulative, cutoff


def ordered_prefix(x):
    return torch.cumsum(x.flatten(1), dim=1).reshape_as(x)


def compact_nonzero(values):
    flags = values != 0.0
    prefix = flags.to(torch.int32).cumsum(dim=1, dtype=torch.int32)
    output = torch.full_like(prefix, -1)
    rows, columns = torch.nonzero(flags, as_tuple=True)
    output[rows, prefix[rows, columns].long() - 1] = columns.to(torch.int32)
    return output, flags.sum(dim=1, dtype=torch.int32)


def csr_spmv(row_offsets, column_indices, values, vector):
    return (
        values.reshape(32768, 32)
        * vector[column_indices.long()].reshape(32768, 32)
    ).sum(dim=1)


def csr_spmm(row_offsets, column_indices, values, dense):
    gathered = dense[column_indices.long()].reshape(8192, 32, 128)
    coefficients = values.reshape(8192, 32, 1)
    return (coefficients * gathered).sum(dim=1)


def roi_align_center_sample(feature, rois):
    batch = rois[:, 0].long()
    start_x = rois[:, 1]
    start_y = rois[:, 2]
    end_x = rois[:, 3]
    end_y = rois[:, 4]
    bin_width = (end_x - start_x).clamp_min(1.0) / 7
    bin_height = (end_y - start_y).clamp_min(1.0) / 7
    output = torch.empty(
        (2048, 64, 7, 7),
        device=feature.device,
        dtype=torch.float32,
    )
    for pooled_y in range(7):
        sample_y = start_y + (pooled_y + 0.5) * bin_height
        y_low = sample_y.to(torch.int32).clamp(0, 127).long()
        y_high = (y_low + 1).clamp_max(127)
        y_fraction = sample_y - y_low.float()
        for pooled_x in range(7):
            sample_x = start_x + (pooled_x + 0.5) * bin_width
            x_low = sample_x.to(torch.int32).clamp(0, 127).long()
            x_high = (x_low + 1).clamp_max(127)
            x_fraction = sample_x - x_low.float()
            top = (
                feature[batch, :, y_low, x_low]
                * (1.0 - x_fraction[:, None])
                + feature[batch, :, y_low, x_high] * x_fraction[:, None]
            )
            bottom = (
                feature[batch, :, y_high, x_low]
                * (1.0 - x_fraction[:, None])
                + feature[batch, :, y_high, x_high] * x_fraction[:, None]
            )
            output[:, :, pooled_y, pooled_x] = (
                top * (1.0 - y_fraction[:, None])
                + bottom * y_fraction[:, None]
            )
    return output


def index_select(source, indices):
    return torch.index_select(source, 0, indices)


def gated_dual_gemm(x, gate_weight, value_weight):
    gate = x.float() @ gate_weight.float()
    value = x.float() @ value_weight.float()
    return (torch.relu(gate) * value).to(x.dtype)


def matmul(a, b):
    return torch.matmul(a.float(), b.float()).to(a.dtype)


def vector_dot(lhs, rhs):
    return torch.dot(lhs, rhs).reshape(1)


def vector_outer(lhs, rhs):
    return torch.outer(lhs, rhs)


def batched_gemm_tn(a, b):
    return torch.bmm(a.float().transpose(-1, -2), b.float()).to(torch.bfloat16)


def batched_gemm_nt(a, b):
    return torch.bmm(a.float(), b.float().transpose(-1, -2)).to(torch.bfloat16)


def batched_gemm_tt(a, b):
    return torch.bmm(a.float().transpose(-1, -2), b.float().transpose(-1, -2)).to(torch.bfloat16)


def mla_head_projection(source, weight):
    return torch.einsum("bqhi,hoi->bqho", source.float(), weight.float()).half()


def conv1d_same(x, weight):
    width = weight.numel()
    patches = F.pad(x.float(), (width // 2, width // 2)).unfold(-1, width, 1)
    return (patches * weight.float()).sum(dim=-1).to(x.dtype)


def varlen_causal_conv1d(x, offsets, weight, bias):
    width = weight.shape[1]
    outputs, states = [], []
    boundaries = offsets.tolist()
    for start, end in zip(boundaries, boundaries[1:]):
        values = x[start:end]
        padded = torch.nn.functional.pad(values.float(), (0, 0, width - 1, 0))
        accumulation = (padded.unfold(0, width, 1) * weight).sum(dim=-1) + bias
        outputs.append((accumulation * torch.sigmoid(accumulation)).to(x.dtype))
        state = torch.nn.functional.pad(values, (0, 0, max(0, width - values.shape[0]), 0))
        states.append(state[-width:])
    return torch.cat(outputs, dim=0), torch.stack(states, dim=0)


def triangular_solve(lower, solution):
    for row in range(solution.shape[-1]):
        residual = solution[:, row].clone()
        for column in range(row):
            residual = residual - lower[:, row, column] * solution[:, column]
        solution[:, row].copy_(residual / lower[:, row, row])
    return solution


def cholesky(matrices):
    matrices.copy_(torch.linalg.cholesky(matrices))
    return matrices


def householder_qr(matrices):
    factor, tau = torch.geqrf(matrices)
    matrices.copy_(factor)
    return matrices, tau


def causal_conv_update(x, state, weight, bias):
    updated = torch.cat((state[:, :, 1:], x[:, :, None]), dim=2)
    state.copy_(updated)
    accumulator = bias.float()[None, :].expand(x.shape[0], -1).clone()
    for tap in range(weight.shape[1]):
        accumulator = accumulator + state[:, :, tap].float() * weight[:, tap].float()
    output = (accumulator * torch.sigmoid(accumulator)).to(x.dtype)
    return state, output


def bitonic_sort(values):
    result = values.clone()
    indices = torch.arange(values.shape[1], device=values.device)
    sequence = 2
    while sequence <= values.shape[1]:
        stride = sequence // 2
        while stride > 0:
            partner = indices ^ stride
            other = result[:, partner]
            take_minimum = ((indices & sequence) == 0) == (indices < partner)
            swap = torch.where(take_minimum, result > other, result < other)
            result = torch.where(swap, other, result)
            stride //= 2
        sequence *= 2
    return result


def insertion_top_k(logits):
    values = torch.full((logits.shape[0], 8), -torch.inf, dtype=logits.dtype)
    indices = torch.full((logits.shape[0], 8), -1, dtype=torch.int32)
    for candidate in range(logits.shape[1]):
        carry_value = logits[:, candidate]
        carry_index = torch.full((logits.shape[0],), candidate, dtype=torch.int32)
        for slot in range(8):
            current_value = values[:, slot].clone()
            current_index = indices[:, slot].clone()
            swap = carry_value > current_value
            values[:, slot] = torch.where(swap, carry_value, current_value)
            indices[:, slot] = torch.where(swap, carry_index, current_index)
            carry_value = torch.where(swap, current_value, carry_value)
            carry_index = torch.where(swap, current_index, carry_index)
    return values, indices


def radix2_fft(input_real, input_imag, twiddle_real, twiddle_imag):
    size = input_real.shape[1]
    source = torch.arange(size, device=input_real.device)
    reversed_index = torch.zeros_like(source)
    for _ in range(size.bit_length() - 1):
        reversed_index = (reversed_index << 1) | (source & 1)
        source = source >> 1
    real = input_real[:, reversed_index]
    imag = input_imag[:, reversed_index]
    for stage in range(size.bit_length() - 1):
        span = 1 << (stage + 1)
        half = span // 2
        real_blocks = real.reshape(real.shape[0], -1, span)
        imag_blocks = imag.reshape(imag.shape[0], -1, span)
        even_real, odd_real = real_blocks[..., :half], real_blocks[..., half:]
        even_imag, odd_imag = imag_blocks[..., :half], imag_blocks[..., half:]
        weight_real = twiddle_real[stage, :half]
        weight_imag = twiddle_imag[stage, :half]
        rotated_real = odd_real * weight_real - odd_imag * weight_imag
        rotated_imag = odd_real * weight_imag + odd_imag * weight_real
        real = torch.cat((even_real + rotated_real, even_real - rotated_real), dim=-1).reshape_as(real)
        imag = torch.cat((even_imag + rotated_imag, even_imag - rotated_imag), dim=-1).reshape_as(imag)
    return real, imag


def viterbi(emissions, transitions):
    batch, time_steps, states = emissions.shape
    previous = emissions[:, 0]
    predecessors = torch.empty((batch, time_steps, states), dtype=torch.int64, device=emissions.device)
    for time in range(1, time_steps):
        values, sources = (previous[:, :, None] + transitions[None, :, :]).max(dim=1)
        predecessors[:, time] = sources
        previous = values + emissions[:, time]
    score, state = previous.max(dim=1)
    path = torch.empty((batch, time_steps), dtype=torch.int32, device=emissions.device)
    path[:, -1] = state.to(torch.int32)
    batch_index = torch.arange(batch, device=emissions.device)
    for time in range(time_steps - 1, 0, -1):
        state = predecessors[batch_index, time, state]
        path[:, time - 1] = state.to(torch.int32)
    return path, score


def smith_waterman(query, reference):
    previous = torch.zeros((query.shape[0], reference.shape[1] + 1), dtype=torch.int32, device=query.device)
    maximum = torch.zeros((query.shape[0],), dtype=torch.int32, device=query.device)
    for row in range(1, query.shape[1] + 1):
        current = torch.zeros_like(previous)
        for column in range(1, reference.shape[1] + 1):
            substitution = torch.where(query[:, row - 1] == reference[:, column - 1], 2, -1)
            cell = torch.maximum(torch.zeros_like(maximum), torch.maximum(
                previous[:, column - 1] + substitution,
                torch.maximum(previous[:, column] - 1, current[:, column - 1] - 1)))
            current[:, column] = cell
            maximum = torch.maximum(maximum, cell)
        previous = current
    return maximum.to(torch.int32)


def adafactor(gradient, parameter, row_state, column_state, row_mean, decay, learning_rate, epsilon):
    row_state.mul_(decay).add_(gradient.square().mean(dim=1), alpha=1.0 - decay)
    column_state.mul_(decay).add_(gradient.square().mean(dim=0), alpha=1.0 - decay)
    row_mean.copy_(row_state.mean().reshape(1))
    variance = row_state[:, None] * column_state[None, :] / row_mean
    parameter.add_(gradient * torch.rsqrt(variance + epsilon), alpha=-learning_rate)


def adamw(gradient, parameter, first_moment, second_moment, learning_rate,
          beta1, beta2, bias_correction1, bias_correction2, epsilon, weight_decay):
    first = beta1 * first_moment + (1.0 - beta1) * gradient
    second = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
    denominator = torch.sqrt(second / bias_correction2) + epsilon
    update = (first / bias_correction1) / denominator + weight_decay * parameter
    parameter.copy_(parameter - learning_rate * update)
    first_moment.copy_(first)
    second_moment.copy_(second)
    return parameter, first_moment, second_moment


def jagged_mean(values, offsets):
    return torch.segment_reduce(values, "mean", offsets=offsets)


def ragged_grouped_gemm(x, offsets, weight):
    output = torch.empty((x.shape[0], weight.shape[2]), dtype=x.dtype)
    for group in range(weight.shape[0]):
        begin, end = int(offsets[group]), int(offsets[group + 1])
        output[begin:end] = x[begin:end].float() @ weight[group].float()
    return output


def ragged_grouped_gemm_backward_weight(left, right, offsets, output):
    for group in range(output.shape[0]):
        begin, end = int(offsets[group]), int(offsets[group + 1])
        output[group] = left[begin:end].float().T @ right[begin:end].float()
    return output


def routed_expert_projection_bf16(x, offsets, member_routes, weight):
    top_k = member_routes.numel() // x.shape[0]
    output = torch.empty((x.shape[0] * top_k, weight.shape[1]), dtype=x.dtype)
    for expert in range(weight.shape[0]):
        routes = member_routes[int(offsets[expert]):int(offsets[expert + 1])].long()
        values = x[routes // top_k].float() @ weight[expert].float().T
        output[routes] = values.to(x.dtype)
    return output.reshape(x.shape[0], top_k, weight.shape[1])


def nested_jagged_mean_pool(document_offsets, sentence_offsets, values):
    sentence_lengths = sentence_offsets[1:] - sentence_offsets[:-1]
    document_lengths = document_offsets[1:] - document_offsets[:-1]
    sentence_ids = torch.repeat_interleave(torch.arange(sentence_lengths.numel()), sentence_lengths.long())
    sentence_sums = torch.zeros((sentence_lengths.numel(), values.shape[1]), dtype=values.dtype)
    sentence_sums.index_add_(0, sentence_ids, values)
    document_ids = torch.repeat_interleave(torch.arange(document_lengths.numel()), document_lengths.long())
    document_sums = torch.zeros((document_lengths.numel(), values.shape[1]), dtype=values.dtype)
    document_sums.index_add_(0, document_ids, sentence_sums)
    document_tokens = torch.zeros(document_lengths.shape, dtype=torch.int32)
    document_tokens.index_add_(0, document_ids, sentence_lengths)
    return sentence_sums / sentence_lengths[:, None], document_sums / document_tokens[:, None]


def selective_state_scan(x, decay, drive):
    state = torch.zeros((x.shape[0],), dtype=x.dtype)
    output = torch.empty_like(x)
    for position in range(x.shape[1]):
        state = decay[:, position] * state + drive[:, position] * x[:, position]
        output[:, position] = state
    return output


def streamed_online_softmax_f16(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def recurrent_gated_delta_fwd(query, key, value, gate, beta, scale):
    batch, sequence, heads, dimension = query.shape
    state = torch.zeros((batch, heads, dimension, value.shape[-1]), dtype=torch.float32)
    output = torch.empty_like(value)
    for position in range(sequence):
        q = query[:, position].float() * scale
        k = key[:, position].float()
        v = value[:, position].float()
        decayed = state * gate[:, position].float().exp()[..., None, None]
        remembered = (decayed * k[..., None]).sum(dim=-2)
        update = (v - remembered) * beta[:, position].float()[..., None]
        state = decayed + k[..., None] * update[..., None, :]
        output[:, position] = (state * q[..., None]).sum(dim=-2).to(value.dtype)
    return output, state


def mamba_chunk_scan_fwd(cb, x, dt, cumulative_decay, state_matrix, previous_states, residual_scale):
    batch, chunks, groups, chunk, _ = cb.shape
    heads, dimension = x.shape[2:]
    group_index = torch.arange(heads) // (heads // groups)
    chunk_x = x.reshape(batch, chunks, chunk, heads, dimension)
    state_c = state_matrix.reshape(batch, chunks, chunk, groups, state_matrix.shape[-1])[:, :, :, group_index]
    state = torch.einsum("bcshn,bchpn->bcshp", state_c.float(), previous_states.float())
    decay_values = cumulative_decay.float()
    state *= decay_values.exp().permute(0, 2, 3, 1)[..., None]
    decay = (decay_values[..., :, None] - decay_values[..., None, :]).exp().permute(0, 2, 3, 4, 1)
    coefficients = cb[:, :, group_index].permute(0, 1, 3, 4, 2).float()
    coefficients *= decay
    coefficients *= dt.float().permute(0, 2, 3, 1)[:, :, None]
    causal = torch.ones((chunk, chunk), dtype=torch.bool).tril()
    coefficients.masked_fill_(~causal[None, None, :, :, None], 0.0)
    scan = torch.einsum("bcskh,bckhp->bcshp", coefficients, chunk_x.float())
    result = state + scan + chunk_x.float() * residual_scale.float()[None, None, None, :, None]
    return result.reshape_as(x).to(x.dtype)


def layer_norm_backward(x, dy, weight, mean, rstd):
    normalized = (x.float() - mean[:, None]) * rstd[:, None]
    gradient = dy.float()
    weighted = gradient * weight.float()
    dx = (weighted - weighted.mean(dim=1, keepdim=True)
          - normalized * (weighted * normalized).mean(dim=1, keepdim=True)) * rstd[:, None]
    return dx.to(x.dtype), (gradient * normalized).sum(dim=0), gradient.sum(dim=0)


def group_norm_backward(x, grad_y, weight, mean, rstd):
    batch, channels, spatial = x.shape
    groups = mean.shape[1]
    grouped = x.float().reshape(batch, groups, channels // groups, spatial)
    normalized = (grouped - mean.float()[..., None, None]) * rstd.float()[..., None, None]
    gradient = grad_y.float().reshape_as(grouped)
    weighted = gradient * weight.float().reshape(groups, channels // groups)[None, :, :, None]
    inverse_elements = 1.0 / (channels // groups * spatial)
    total = weighted.sum(dim=(2, 3), keepdim=True)
    projection = (weighted * normalized).sum(dim=(2, 3), keepdim=True)
    dx = rstd.float()[..., None, None] * (weighted - total * inverse_elements - normalized * projection * inverse_elements)
    dw = (gradient * normalized).sum(dim=(0, 3)).reshape(-1)
    db = gradient.sum(dim=(0, 3)).reshape(-1)
    return dx.reshape_as(x).to(x.dtype), dw.to(weight.dtype), db.to(weight.dtype)


def dropout(x, mixed_seed, drop_probability, inverse_keep_probability):
    offsets = torch.arange(x.numel(), dtype=torch.int32).reshape(x.shape)
    hashed = offsets * 1103515245 + mixed_seed
    hashed = hashed ^ (hashed >> 16)
    hashed = hashed ^ (hashed << 8)
    hashed = hashed ^ (hashed >> 4)
    random = (hashed & 0x7FFFFFFF).float() / 2147483647.0
    scaled = x * torch.tensor(inverse_keep_probability, dtype=x.dtype)
    return torch.where(random > drop_probability, scaled, 0.0)


def unique_consecutive(values, run_lengths):
    starts = torch.ones_like(values, dtype=torch.bool)
    starts[:, 1:] = values[:, 1:] != values[:, :-1]
    groups = starts.to(torch.int32).cumsum(dim=1, dtype=torch.int32) - 1
    output = torch.zeros_like(values)
    rows, columns = torch.nonzero(starts, as_tuple=True)
    output[rows, groups[rows, columns].long()] = values[rows, columns]
    run_lengths.scatter_add_(1, groups.long(), torch.ones_like(values))
    counts = starts.sum(dim=1, dtype=torch.int32)
    return output, groups, counts


def prepare_attention_backward(query, key, value, grad_output, scale):
    q, k, v = (tensor.float().requires_grad_(True) for tensor in (query, key, value))
    group = query.shape[1] // key.shape[1]
    scores = (q @ k.repeat_interleave(group, dim=1).transpose(-2, -1)) * scale
    causal = torch.ones(scores.shape[-2:], dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, -float("inf"))
    output = torch.softmax(scores, dim=-1) @ v.repeat_interleave(group, dim=1)
    lse = torch.logsumexp(scores.detach(), dim=-1) * np.log2(np.e)

    def launch():
        gradients = torch.autograd.grad(output, (q, k, v), grad_output.float(), retain_graph=True)
        return tuple(gradient.to(tensor.dtype) for gradient, tensor in zip(gradients, (query, key, value)))

    return output.detach().to(query.dtype), lse, launch


def block_scaled_matmul(lhs, lhs_scale, rhs, rhs_scale):
    def scales(raw):
        decoded = torch.exp2(raw.float() - 127.0)
        return torch.where(raw == 255, float("nan"), decoded)

    left = lhs.float() * scales(lhs_scale)[..., None]
    right = rhs.float() * scales(rhs_scale)[:, None, :]
    return left.reshape(lhs.shape[0], -1) @ right.reshape(-1, rhs.shape[2])


def block_sparse_matmul(lhs, rhs, block_mask):
    rows, depth = lhs.shape
    columns = rhs.shape[1]
    block_m, block_n, block_k = (rows // block_mask.shape[0], columns // block_mask.shape[1],
                                 depth // block_mask.shape[2])
    left, right = lhs.float(), rhs.float()
    output = torch.empty((rows, columns), dtype=lhs.dtype)
    for row in range(block_mask.shape[0]):
        row_slice = slice(row * block_m, (row + 1) * block_m)
        for column in range(block_mask.shape[1]):
            column_slice = slice(column * block_n, (column + 1) * block_n)
            accumulator = torch.zeros((block_m, block_n), dtype=torch.float32)
            for block in torch.nonzero(block_mask[row, column], as_tuple=True)[0].tolist():
                reduction = slice(block * block_k, (block + 1) * block_k)
                accumulator += left[row_slice, reduction] @ right[reduction, column_slice]
            output[row_slice, column_slice] = accumulator.to(lhs.dtype)
    return output


def padded_rope_cache_update(packed_input, sequence_lengths, output_storage, theta, linear_scale):
    batch, _, dimension = packed_input.shape
    query_heads, kv_heads, padded_length = 32, 8, 8193
    half = dimension // 2
    positions = sequence_lengths.long() - 1
    powers = torch.arange(half, dtype=torch.float32) * (-2.0 / dimension)
    angles = positions.float()[:, None] * torch.pow(theta, powers)[None, :] / linear_scale
    cosine, sine = angles.cos()[:, None, :], angles.sin()[:, None, :]
    first = packed_input[:, :query_heads + kv_heads, :half].float()
    second = packed_input[:, :query_heads + kv_heads, half:].float()
    rotated = torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)
    query_rows = batch * query_heads
    cache_rows = batch * padded_length * kv_heads
    output_storage[:query_rows] = rotated[:, :query_heads].reshape(query_rows, dimension)
    key_cache = output_storage[query_rows:query_rows + cache_rows].view(batch, padded_length, kv_heads, dimension)
    value_cache = output_storage[query_rows + cache_rows:].view(batch, padded_length, kv_heads, dimension)
    rows = torch.arange(batch)
    key_cache[rows, positions] = rotated[:, query_heads:].to(packed_input.dtype)
    value_cache[rows, positions] = packed_input[:, query_heads + kv_heads:]
    return output_storage


def moe_align_block_size(topk_ids, block_size, num_experts):
    flat = topk_ids.flatten()
    counts = torch.bincount(flat.long(), minlength=num_experts)
    padded = (counts + block_size - 1) // block_size * block_size
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), padded.cumsum(0)))
    routes = torch.argsort(flat, stable=True).to(torch.int32)
    capacity = flat.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.full((capacity,), flat.numel(), dtype=torch.int32)
    expert_ids = torch.empty(((capacity + block_size - 1) // block_size,), dtype=torch.int32)
    source_offset = 0
    for expert in range(num_experts):
        begin, end = int(offsets[expert]), int(offsets[expert + 1])
        count = int(counts[expert])
        sorted_ids[begin:begin + count] = routes[source_offset:source_offset + count]
        expert_ids[begin // block_size:end // block_size] = expert
        source_offset += count
    return sorted_ids, expert_ids, offsets[-1:].to(torch.int32)


def mhc_gemm_rms_scale(x, weight, bias):
    width = x.shape[1] // 16
    linear, squares = [], []
    for part in range(16):
        values = x[:, part * width:(part + 1) * width].float()
        linear.append(values @ weight[part * width:(part + 1) * width].float())
        squares.append(values.square().sum(dim=1))
    total = torch.stack(linear).sum(dim=0)
    rstd = (torch.stack(squares).sum(dim=0) / x.shape[1]).rsqrt()
    mixed = total * rstd[:, None] + bias.float()
    mixed[:, :4] = mixed[:, :4].sigmoid()
    mixed[:, 4:8] = 2 * mixed[:, 4:8].sigmoid()
    return mixed.bfloat16(), rstd.reciprocal()[:, None]


def mhc_pre(residual, weight, scale, base):
    values = residual.flatten(1, 2).float()
    rstd = (values.square().sum(dim=1) / weight.shape[1] + 1e-6).rsqrt()
    mixes = (values @ weight.bfloat16().float().T) * rstd[:, None]
    factors = torch.cat((scale[0].expand(4), scale[1].expand(4), scale[2].expand(16)))
    mixes = mixes * factors + base
    pre = mixes[:, :4].sigmoid() + 1e-6
    post = mixes[:, 4:8].sigmoid()
    matrix = mixes[:, 8:].reshape(-1, 4, 4).softmax(dim=-1) + 1e-6
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-6)
    for _ in range(9):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-6)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-6)
    layer_input = (residual.float() * pre[:, :, None]).sum(dim=1).bfloat16()
    return post, matrix, layer_input


def mhc_sinkhorn(values):
    matrix = values.exp()
    for _ in range(20):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True)
    values.copy_(matrix)
    return values


def mhc_apply_residual(residual, layer_output, post_mix, residual_mix):
    mixed = torch.bmm(residual_mix, residual.float())
    return (mixed + layer_output.float()[:, None, :] * post_mix[:, :, None]).to(residual.dtype)


def swiglu_float_intermediate(gate, up):
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)


def rotary_embedding_flat(values, cosine, sine):
    rows, dimension = values.shape
    half = dimension // 2
    dimensions = torch.arange(dimension)
    paired = (dimensions - half) % dimension
    phase = dimensions % half
    sign = torch.where(dimensions < half, -1.0, 1.0).to(values.dtype)
    tokens = torch.arange(rows) // (rows // cosine.shape[0])
    rotated = values * cosine[tokens][:, phase] + values[:, paired] * sine[tokens][:, phase] * sign
    return rotated.reshape(rows, 2, half)


def mamba_chunk_state_fwd(state_basis, x, dt, cumulative_decay):
    batch, sequence, heads, dimension = x.shape
    chunks, chunk = dt.shape[-2:]
    basis = state_basis.repeat_interleave(heads // state_basis.shape[2], dim=2)
    basis = basis.reshape(batch, chunks, chunk, heads, state_basis.shape[-1])
    values = x.reshape(batch, chunks, chunk, heads, dimension)
    decay = torch.exp(cumulative_decay[..., -1:] - cumulative_decay)
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", basis, decay.to(x.dtype), dt, values)


def linear_attention_forward(q, k, v, scale):
    batch, sequence, heads, dimension = q.shape
    chunk = 64
    def chunks(tensor):
        return tensor.float().reshape(batch, sequence // chunk, chunk, heads, tensor.shape[-1]).permute(0, 3, 1, 2, 4)

    query, key, value = chunks(q) * scale, chunks(k), chunks(v)
    prefix = (key.transpose(-1, -2) @ value).cumsum(2)
    final = prefix[:, :, -1]
    previous = torch.cat((torch.zeros_like(prefix[:, :, :1]), prefix[:, :, :-1]), dim=2)
    future = torch.ones((chunk, chunk), dtype=torch.bool).triu(1)
    intra = (query @ key.transpose(-1, -2)).masked_fill(future, 0.0) @ value
    output = query @ previous + intra
    return output.permute(0, 2, 3, 1, 4).reshape(batch, sequence, heads, value.shape[-1]), final


def moe_expert_ffn(x, route_offsets, member_routes, route_token, route_weights, w1, w2, y):
    for expert in range(w1.shape[0]):
        routes = member_routes[route_offsets[expert]:route_offsets[expert + 1]].long()
        tokens = route_token[routes].long()
        hidden = torch.relu(x[tokens].float() @ w1[expert].float())
        values = hidden @ w2[expert].float()
        y.index_add_(0, tokens, route_weights[routes, None] * values)
    return y


def make_sparse_2to4_inputs(rows, depth):
    # The original TileLang input generator zeros each group's two largest values.
    dense = torch.randn((rows, depth), dtype=torch.float32).view(rows, -1, 4)
    dense.scatter_(-1, dense.topk(2, dim=-1).indices, 0)
    groups = dense.to(torch.float16)
    m0, m1, _, m3 = (groups != 0).unbind(-1)
    expr0, expr1, expr2 = m0 & m1, ~m0 & m1, ~m0 & ~m1
    first = expr1.long() | (expr2.long() << 1)
    second = (expr0 | expr2 | m3).long() | ((expr1 | ~m1).long() << 1)
    compressed = torch.stack((groups.gather(-1, first[..., None]),
                              groups.gather(-1, second[..., None])), dim=-1).view(rows, depth // 2)
    nibbles = (first | (second << 2)).view(rows, depth // 16, 4).to(torch.int16)
    metadata = nibbles[:, :, 0]
    for lane in range(1, 4):
        metadata = metadata | (nibbles[:, :, lane] << (4 * lane))
    return compressed, metadata


def sparse_2to4_gemm(compressed, metadata, rhs):
    rows = compressed.shape[0]
    depth = rhs.shape[0]
    shifts = torch.arange(4, dtype=torch.int16) * 4
    nibbles = ((metadata[..., None] >> shifts) & 15).reshape(rows, depth // 4)
    positions = torch.stack((nibbles & 3, (nibbles >> 2) & 3), dim=-1).long()
    dense = torch.zeros((rows, depth // 4, 4), dtype=torch.float32)
    dense.scatter_(2, positions, compressed.float().reshape(rows, depth // 4, 2))
    return dense.reshape(rows, depth) @ rhs.float()


def reshape_and_cache(key, value, slots, key_cache, value_cache):
    flat_shape = (-1, *key.shape[1:])
    key_cache.view(flat_shape)[slots.long()] = key
    value_cache.view(flat_shape)[slots.long()] = value
    return key_cache, value_cache


def nvfp4_quantize(x, global_scale, packed, scales):
    rows, columns = x.shape
    groups = x.float().reshape(rows, columns // 16, 16)
    scale = ((groups.abs().amax(-1) / 6.0) * global_scale[0]).clamp_min(1.5258789e-5).to(torch.float8_e4m3fn)
    values = groups * (global_scale[0] / scale.float())[..., None]
    magnitude = values.abs()
    encoded = (magnitude > 0.25).to(torch.uint8)
    for code, threshold in enumerate((0.75, 1.25, 1.75, 2.5, 3.5, 5.0), start=2):
        selected = magnitude >= threshold if code % 2 == 0 else magnitude > threshold
        encoded = torch.where(selected, code, encoded)
    encoded |= torch.signbit(values).to(torch.uint8) << 3
    pairs = encoded.reshape(rows, columns // 2, 2)
    packed.copy_(pairs[..., 0] | (pairs[..., 1] << 4))
    scale_bytes = scale.view(torch.uint8).reshape(rows // 128, 4, 32, columns // 64, 4)
    scales.copy_(scale_bytes.permute(0, 3, 2, 1, 4))
    return packed, scales


def fp8_mqa_logits(q, kv, kv_scale, head_weight, key_start, key_end):
    logits = (q.float() @ kv.float().T).relu()
    output = (logits * head_weight[..., None]).sum(dim=1) * kv_scale[None, :]
    keys = torch.arange(kv.shape[0])[None, :]
    return output.masked_fill((keys < key_start[:, None]) | (keys >= key_end[:, None]), -float("inf"))


def _grouped_decode(query, key, value, scale, soft_cap=0.0):
    heads, dimension = query.shape
    groups = key.shape[1]
    q = query.float().reshape(groups, heads // groups, dimension)
    scores = (q @ key.float().permute(1, 2, 0)) * scale
    if soft_cap > 0.0:
        scores = soft_cap * torch.tanh(scores / soft_cap)
    output = torch.softmax(scores, dim=-1) @ value.float().permute(1, 0, 2)
    return output.reshape(heads, value.shape[-1]).to(query.dtype)


def gemma_decode(q, k, v, scale, *, window_size, soft_cap, kv_len_per_split):
    begin = max(0, k.shape[2] - 1 - window_size) if window_size > 0 else 0
    return torch.stack([
        _grouped_decode(q[batch, :, 0], k[batch, :, begin:].transpose(0, 1),
                        v[batch, :, begin:].transpose(0, 1), scale, soft_cap)
        for batch in range(q.shape[0])
    ]).unsqueeze(2)


def paged_gqa_decode(q, key_cache, value_cache, page_offsets, page_indices,
                     lengths, split_offsets, scale, *, page_size):
    outputs = []
    for batch in range(q.shape[0]):
        pages = page_indices[page_offsets[batch]:page_offsets[batch + 1]].long()
        key = key_cache[pages].reshape(-1, *key_cache.shape[2:])[:lengths[batch]]
        value = value_cache[pages].reshape(-1, *value_cache.shape[2:])[:lengths[batch]]
        outputs.append(_grouped_decode(q[batch], key, value, scale))
    return torch.stack(outputs)


def block_sparse_gqa_decode(q, k, v, selected, lengths, split_offsets, scale, *, block_size):
    outputs = []
    group = q.shape[1] // k.shape[2]
    for batch in range(q.shape[0]):
        heads = []
        for head in range(k.shape[2]):
            positions = (selected[batch, head].long()[:, None] * block_size + torch.arange(block_size)).flatten()
            positions = positions[(positions >= 0) & (positions < lengths[batch])]
            heads.append(_grouped_decode(q[batch, head * group:(head + 1) * group],
                                         k[batch, positions, head][:, None],
                                         v[batch, positions, head][:, None], scale))
        outputs.append(torch.cat(heads))
    return torch.stack(outputs)


def paged_mla_decode(q_latent, q_rope, latent_cache, rope_cache,
                     page_offsets, page_indices, lengths, scale):
    outputs = []
    groups = latent_cache.shape[2]
    heads_per_group = q_latent.shape[1] // groups
    for batch in range(q_latent.shape[0]):
        pages = page_indices[page_offsets[batch]:page_offsets[batch + 1]].long()
        latent = latent_cache[pages].reshape(-1, groups, latent_cache.shape[-1])[:lengths[batch]].float()
        rope = rope_cache[pages].reshape(-1, groups, rope_cache.shape[-1])[:lengths[batch]].float()
        query = q_latent[batch].float().reshape(groups, heads_per_group, -1)
        query_rope = q_rope[batch].float().reshape(groups, heads_per_group, -1)
        scores = (query @ latent.permute(1, 2, 0) + query_rope @ rope.permute(1, 2, 0)) * scale
        output = torch.softmax(scores, dim=-1) @ latent.permute(1, 0, 2)
        outputs.append(output.reshape(q_latent.shape[1], latent.shape[-1]).to(q_latent.dtype))
    return torch.stack(outputs)


def prepare_sparse_mla_backward(query, key_value, grad_output, selected_indices, scale):
    batch, sequence, heads, dimension = query.shape
    groups = key_value.shape[2]
    heads_per_group = heads // groups
    value_dimension = grad_output.shape[-1]
    output = torch.empty_like(grad_output)
    lse = torch.empty((batch, sequence, heads), dtype=torch.float32)

    def blocks():
        for b in range(batch):
            for group in range(groups):
                head_slice = slice(group * heads_per_group, (group + 1) * heads_per_group)
                for begin in range(0, sequence, 32):
                    end = min(begin + 32, sequence)
                    rows = slice(begin, end)
                    raw = selected_indices[b, rows, group].long()
                    positions = torch.arange(begin, end)[:, None]
                    valid = (raw >= 0) & (raw <= positions) & (raw < key_value.shape[1])
                    safe = torch.where(valid, raw, 0)
                    selected = key_value[b, safe, group].float()
                    q = query[b, rows, head_slice].float()
                    yield b, group, rows, head_slice, raw, valid, selected, q

    for b, group, rows, head_slice, raw, valid, selected, q in blocks():
        scores = (q @ selected.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~valid[:, None, :], -float("inf"))
        probability = torch.softmax(scores, dim=-1)
        output[b, rows, head_slice] = (probability @ selected[..., :value_dimension]).to(output.dtype)
        lse[b, rows, head_slice] = torch.logsumexp(scores, dim=-1) * np.log2(np.e)

    def backward():
        delta = (output.float() * grad_output.float()).sum(dim=-1)
        grad_query = torch.empty_like(query)
        grad_kv = torch.zeros_like(key_value, dtype=torch.float32)
        for b, group, rows, head_slice, raw, valid, selected, q in blocks():
            do = grad_output[b, rows, head_slice].float()
            probability = torch.exp2((q @ selected.transpose(-2, -1)) * (scale * np.log2(np.e))
                                     - lse[b, rows, head_slice, None])
            probability = torch.where(valid[:, None, :], probability, 0.0)
            dp = do @ selected[..., :value_dimension].transpose(-2, -1)
            ds = (probability * (dp - delta[b, rows, head_slice, None]) * scale).to(query.dtype).float()
            grad_query[b, rows, head_slice] = (ds @ selected).to(query.dtype)
            update = ds.transpose(-2, -1) @ q
            update[..., :value_dimension] += probability.to(query.dtype).float().transpose(-2, -1) @ do
            grad_kv[b, :, group].index_add_(0, raw[valid], update[valid])
        return grad_query, grad_kv.to(key_value.dtype)

    return output, lse, backward
