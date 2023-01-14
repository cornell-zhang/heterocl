from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import *


class UniqueName(object):
    scalar_idx = 0
    loop_idx = 0
    tensor_idx = 0
    stage_idx = 0
    schedule_idx = 0
    reduction_axis_idx = 0
    instance_idx = 0
    op_idx = 0

    def __init__(self):
        pass

    @classmethod
    def reset(cls):
        cls.scalar_idx = 0
        cls.loop_idx = 0
        cls.tensor_idx = 0
        cls.stage_idx = 0
        cls.schedule_idx = 0
        cls.reduction_axis_idx = 0
        cls.instance_idx = 0
        cls.op_idx = 0

    @classmethod
    def get(cls, case="stage"):
        if case == "stage":
            # Imperative computing stage
            name = "stage_" + str(cls.stage_idx)
            cls.stage_idx += 1
        elif case == "loop":
            name = "i" + str(cls.loop_idx)
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
        elif case == "op":
            name = "op_" + str(cls.op_idx)
            cls.op_idx += 1
        else:
            raise RuntimeError(f"Unrecognized case in get_unique_name: {case}")
        return name


class GlobalContext(object):
    in_context = False

    def __init__(self):
        self.ctx = None
        self.loc = None
        GlobalContext.in_context = True

    def get_context(self):
        return self.ctx

    def set_context(self):
        self.ctx = Context()
        hcl_d.register_dialect(self.ctx)
        self.loc = Location.unknown(self.ctx)

    def get_location(self):
        return self.loc

    def exit_context(self):
        GlobalContext.in_context = False


global_ctx = GlobalContext()
get_context = global_ctx.get_context
set_context = global_ctx.set_context
get_location = global_ctx.get_location
exit_context = global_ctx.exit_context
