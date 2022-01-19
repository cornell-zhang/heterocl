import io
import os
import subprocess

import hcl_mlir
from hcl_mlir import GlobalInsertionPoint, get_context, get_location

from mlir import passmanager
from mlir.execution_engine import *
from mlir.ir import *

from .module import HCLModule
from .runtime import copy_build_files


def lower(sch,
          name="top",
          binds=None,
          simple_mode=False,
          kernel_only=False,
          stmt=None):
    """Lowering step before build into target
    """
    func = sch.get_top_function()

    # apply optimization passes
    hcl_mlir.loop_transformation(func.operation)
    sch.get_module().dump()

    return sch.get_module()


def build(schedule, target=None, name="top", stmt=None):
    """Build the executable according to the schedule and target.
    """
    lowered_module = lower(schedule)

    if target != None:
        return build_fpga_kernel(schedule, target, name, stmt)
    else:
        return build_llvm(schedule, target, name, stmt)


def build_fpga_kernel(schedule, target=None, name="top", stmt=None):
    # make the project folder and copy files
    copy_build_files(target)

    # generate code
    buf = io.StringIO()
    hcl_mlir.emit_hlscpp(schedule.get_module(), buf)
    buf.seek(0)
    hls_code = buf.read()

    # write HLS code to file
    with open("{}/kernel.cpp".format(target.project), "w") as outfile:
        outfile.write(hls_code)
    # TODO: generate host code
    with open("{}/host.cpp".format(target.project), "w") as outfile:
        outfile.write("")

    hcl_module = HCLModule(name, hls_code, target)

    return hcl_module

# TODO(Niansong): not useful for now, consider removal
def reconcile_unrealized_casts(module):
    import mlir.conversions
    pm = passmanager.PassManager.parse(
        "reconcile-unrealized-casts")
    pm.run(module)
    return module


def build_llvm(schedule, target=None, name="top", stmt=None):
    with get_context() as ctx, get_location():
        func = schedule.get_top_function()
        func.attributes['llvm.emit_c_interface'] = UnitAttr.get()
        # print("\n\nBefore Lowering: ")
        # schedule.get_module().dump()
        hcl_mlir.lower_hcl_to_llvm(schedule.get_module(), ctx)
        # print("lowered.")
        print("\n\nAfter Lowering: ")
        schedule.get_module().dump()
        # execution_engine = ExecutionEngine(schedule.get_module(), opt_level=0, shared_libs=["/work/shared/users/phd/nz264/llvm-13.0/build/lib/libmlir_c_runner_utils.so.13"])
        execution_engine = ExecutionEngine(schedule.get_module())
        hcl_module = HCLModule(name, execution_engine, "llvm", ctx)
        return hcl_module
