import io
import os
import subprocess

import hcl_mlir
from hcl_mlir import GlobalInsertionPoint, get_context, get_location

from mlir import passmanager
from mlir.execution_engine import *
from mlir.ir import *

from .module import HCLModule
from .runtime import copy_build_files, execute_fpga_backend


def lower(schedule,
          name="top",
          binds=None,
          simple_mode=False,
          kernel_only=False,
          stmt=None):
    """Lowering step before build into target
    """

    # apply optimization passes
    hcl_mlir.loop_transformation(schedule.get_module())
    schedule.get_module().dump()

    return schedule.get_module()


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


def lowerToLLVM(module):
    import mlir.conversions
    pm = passmanager.PassManager.parse(
        "reconcile-unrealized-casts")
    pm.run(module)
    return module


def build_llvm(schedule, target=None, name="top", stmt=None):
    with get_context() as ctx, get_location():
        # mod = schedule.get_module()
        func = schedule.get_top_function()
        func.attributes['llvm.emit_c_interface'] = UnitAttr.get()
        print("\n\nBefore Lowering: ")
        schedule.get_module().dump()
        hcl_mlir.lower_hcl_to_llvm(get_module(), ctx)
        # lowerToLLVM(schedule.get_module())
        print("lowered.")
        print("\n\nAfter Lowering: ")
        schedule.get_module().dump()
        execution_engine = ExecutionEngine(schedule.get_module())
        execution_engine.invoke(name)
        print("Execution success")
