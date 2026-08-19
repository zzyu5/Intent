import importlib.util
from pathlib import Path

import torch


SOURCE = Path(__file__).with_name("example_convolution.py")


def load_source():
    spec = importlib.util.spec_from_file_location("local_tilelang_convolution", SOURCE)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


def main():
    source = load_source()
    batch, height, width, channels, filters = 32, 128, 128, 256, 512
    kernel_size, stride, dilation, padding = 3, 1, 1, 1
    data = torch.randn(
        batch, height, width, channels, device="cuda", dtype=torch.float16
    )
    weight = torch.randn(
        kernel_size, kernel_size, channels, filters, device="cuda", dtype=torch.float16
    )

    output = source.convolution(
        data, weight, stride, dilation, padding, 64, 128, 32, 3, 256
    )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = source.convolution(
        data, weight, stride, dilation, padding, 64, 128, 32, 3, 256
    )
    end.record()
    torch.cuda.synchronize()

    print(f"data={tuple(data.shape)} weight={tuple(weight.shape)} layout=NHWC/HWCF")
    print(f"output={tuple(output.shape)} mean={output.float().mean().item():.6f}")
    print(f"latency_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
