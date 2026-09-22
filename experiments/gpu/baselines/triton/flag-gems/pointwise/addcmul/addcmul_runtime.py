import importlib.util
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name("addcmul.py")
SPEC = importlib.util.spec_from_file_location("intent_flaggems_addcmul", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
STATE = {}


def upstream(arguments):
    x, scale, bias = arguments
    if "output" not in STATE:
        STATE["output"] = x.new_empty(x.shape)
    return MODULE.addcmul_out(
        bias[:, :, None],
        x,
        scale[:, :, None],
        value=1.0,
        out=STATE["output"],
    )
