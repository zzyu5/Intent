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


def softmax_backward(probabilities, upstream):
    projection = (probabilities * upstream).sum(dim=-1, keepdim=True)
    return probabilities * (upstream - projection)


def transpose(x):
    return x.T.contiguous()


def index_select(source, indices):
    return torch.index_select(source, 0, indices)


def matmul(a, b):
    return torch.matmul(a.float(), b.float()).to(a.dtype)


def conv1d_same(x, weight):
    width = weight.numel()
    patches = F.pad(x.float(), (width // 2, width // 2)).unfold(-1, width, 1)
    return (patches * weight.float()).sum(dim=-1).to(x.dtype)


def triangular_solve(lower, solution):
    for row in range(solution.shape[-1]):
        residual = solution[:, row].clone()
        for column in range(row):
            residual = residual - lower[:, row, column] * solution[:, column]
        solution[:, row].copy_(residual / lower[:, row, row])
    return solution


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
