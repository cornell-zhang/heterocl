# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2023 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import *
from hcl_mlir.exceptions import *


class UniqueName(object):

    sets = {
        "scalar": set(),
        "loop": set(),
        "tensor": set(),
        "stage": set(),
        "schedule": set(),
        "reduction_axis": set(),
        "instance": set(),
        "op": set(),
    }

    def __init__(self):
        pass

    @classmethod
    def reset(cls):
        for _, v in cls.sets.items():
            v.clear()

    @classmethod
    def get(cls, name, case):
        if case not in cls.sets.keys():
            raise APIError(f"Unrecognized case in UniqueName.get(): {case}")

        if name is None or name == "":
            # generate a name if name is not given
            case_set = cls.sets[case]
            set_size = len(case_set)
            name = case + "_" + str(set_size)
            cls.sets[case].add(name)
            return name

        if name in cls.sets[case]:
            # name is not unique
            # generate a unique name
            case_set = cls.sets[case]
            set_size = len(case_set)
            name = name + "_" + str(set_size)
            cls.sets[case].add(name)
            return name
        else:
            # name is unique
            cls.sets[case].add(name)
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
