import importlib.util
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name("_fused_adam.py")
SPEC = importlib.util.spec_from_file_location("intent_flaggems_fused_adam", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def upstream(arguments):
    (
        gradient,
        parameter,
        first_moment,
        second_moment,
        learning_rate,
        beta1,
        beta2,
        inverse_bias1,
        inverse_bias2,
        epsilon,
        weight_decay,
    ) = arguments
    elements = parameter.numel()
    block_size = min(max(MODULE.triton.next_power_of_2(elements), 128), 4096)
    MODULE._fused_adam_kernel[(MODULE.triton.cdiv(elements, block_size),)](
        parameter,
        gradient,
        first_moment,
        second_moment,
        second_moment,
        elements,
        block_size,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        epsilon,
        1.0 / inverse_bias1,
        1.0 / inverse_bias2,
        False,
        False,
    )
    return parameter, first_moment, second_moment
