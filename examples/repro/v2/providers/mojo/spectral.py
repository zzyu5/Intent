import math

import torch

from kernels.spectral.fft import radix2_fft
from ...model import Tolerance
from .common import prepare_host_comparison


def fft(context):
    real = torch.randn((1024, 1024), dtype=torch.float32)
    imag = torch.randn_like(real)
    lanes = torch.arange(512, dtype=torch.float32)
    twiddle_real = torch.empty((10, 512), dtype=torch.float32)
    twiddle_imag = torch.empty_like(twiddle_real)
    for stage in range(10):
        angle = -2.0 * math.pi * lanes / (1 << (stage + 1))
        twiddle_real[stage] = torch.cos(angle)
        twiddle_imag[stage] = torch.sin(angle)
    return prepare_host_comparison(context, radix2_fft, (real, imag, twiddle_real, twiddle_imag),
                                   "radix2_fft", (Tolerance(atol=3e-4), Tolerance(atol=3e-4)))


CASES = {"radix2_fft": fft}
