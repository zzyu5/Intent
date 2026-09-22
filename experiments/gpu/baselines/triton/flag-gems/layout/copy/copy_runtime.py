import importlib.util
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name("copy.py")
SPEC = importlib.util.spec_from_file_location("intent_flaggems_copy", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
STATE = {}


def upstream(arguments):
    x = arguments[0]
    if "output" not in STATE:
        STATE["output"] = x.new_empty((x.shape[1], x.shape[0]))
    MODULE.copy_(STATE["output"], x.transpose(0, 1))
    return STATE["output"]
