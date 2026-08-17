import importlib.util
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[2] / "support"))

source_path = Path(__file__).with_name("cross_entropy.py")
spec = importlib.util.spec_from_file_location("liger_kernel.ops.cross_entropy", source_path)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)
cross_entropy_forward = source.cross_entropy_forward
STATE = {}


def upstream(arguments):
    logits, target = arguments
    rows, vocabulary = logits.shape
    key = (
        tuple(logits.shape),
        logits.dtype,
        logits.device,
        target.data_ptr(),
        target._version,
    )
    if key not in STATE:
        non_ignored = int((target != -100).sum().item())
        STATE[key] = (
            torch.empty((rows,), device=logits.device, dtype=logits.dtype),
            torch.empty((rows,), device=logits.device, dtype=torch.int64),
            non_ignored,
        )
    loss, predicted, non_ignored = STATE[key]
    block = min(source.MAX_FUSED_SIZE, source.triton.next_power_of_2(vocabulary))
    source.liger_cross_entropy_kernel[(rows,)](
        X_ptr=logits,
        X_stride=logits.stride(0),
        Y_ptr=target,
        Y_stride=target.stride(0),
        weight_ptr=None,
        loss_ptr=loss,
        z_loss_ptr=None,
        loss_stride=loss.stride(0),
        token_accuracy_ptr=None,
        token_accuracy_stride=0,
        predicted_tokens_ptr=predicted,
        predicted_tokens_stride=predicted.stride(0),
        n_cols=vocabulary,
        n_non_ignore=non_ignored,
        sum_non_ignore_weight=non_ignored,
        ignore_index=-100,
        weight_sum=0.0,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="none",
        softcap=None,
        RETURN_Z_LOSS=False,
        RETURN_TOKEN_ACCURACY=False,
        RETURN_PREDICTED_TOKENS=True,
        BLOCK_SIZE=block,
        HAS_WEIGHT=False,
        HAS_SOFTCAPPING=False,
        HAS_GRADIENTS=True,
        num_warps=32 if not source.is_hip() else 16,
    )
    return loss, predicted, logits


def main():
    tokens, vocabulary = 8192, 32768
    logits = torch.randn(
        (tokens, vocabulary), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    target = torch.randint(0, vocabulary, (tokens,), device="cuda", dtype=torch.int64)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss, _, accuracy, _, _ = cross_entropy_forward(
        logits,
        target,
        None,
        -100,
        0.0,
        0.0,
        "mean",
        None,
        False,
        True,
        False,
    )
    end.record()
    torch.cuda.synchronize()

    print(f"logits: shape={tuple(logits.shape)}, dtype={logits.dtype}")
    print(f"target: shape={tuple(target.shape)}, vocabulary={vocabulary}")
    print(f"loss={loss.float().item():.6f}, token_accuracy={accuracy.item():.6f}")
    print(f"end_to_end_ms={start.elapsed_time(end):.3f}")


if __name__ == "__main__":
    main()
