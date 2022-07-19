from contextvars import ContextVar

from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import *

ImperativeLoopNestCount = ContextVar("ImperativeLoopNestCount", default=1)
ImperativeLoopDepth = ContextVar("ImperativeLoopDepth", default=0)
StageName = ContextVar("StageName", default="")
NestedCompute = ContextVar("NestedCompute", default=0)
BreakFlag = ContextVar("BreakFlag", default=False)

class UniqueName(object):
    scalar_idx = 0
    loop_idx = 0
    tensor_idx = 0
    stage_idx = 0
    schedule_idx = 0
    reduction_axis_idx = 0
    instance_idx = 0

    def __init__(self):
        pass

    @classmethod
    def get(cls, case="stage"):
        if case == "stage":
            # Imperative computing stage
            name = "stage_" + str(cls.stage_idx)
            cls.stage_idx += 1
        elif case == "loop":
            name = "loop_" + str(cls.loop_idx)
            cls.loop_idx += 1
        elif case == "scalar":
            name = "scalar_" + str(cls.scalar_idx)
            cls.scalar_idx += 1
        elif case == "tensor":
            name = "compute_" + str(cls.tensor_idx)
            cls.tensor_idx += 1
        elif case == "schedule":
            name = "schedule_" + str(cls.schedule_idx)
            cls.schedule_idx += 1
        elif case == "reduction_axis":
            name = "rx_" + str(cls.reduction_axis_idx)
            cls.reduction_axis_idx += 1
        elif case == "instance":
            name = "instance_" + str(cls.instance_idx)
            cls.instance_idx += 1
        else:
            raise RuntimeError(f"Unrecognized case in get_unique_name: {case}")
        return name


class GlobalContext(object):
    def __init__(self):
        self.ctx = None
        self.loc = None

    def get_context(self):
        return self.ctx

    def set_context(self):
        self.ctx = Context()
        hcl_d.register_dialect(self.ctx)
        self.loc = Location.unknown(self.ctx)

    def get_location(self):
        return self.loc


global_ctx = GlobalContext()
get_context = global_ctx.get_context
set_context = global_ctx.set_context
get_location = global_ctx.get_location
