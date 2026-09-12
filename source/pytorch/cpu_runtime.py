import torch
import torch.nn.functional as F


def gelu(x):
    return F.gelu(x, approximate="tanh")


def relu(x):
    return torch.maximum(x, x.new_zeros(()))


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
