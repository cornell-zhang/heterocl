import io
import os
import subprocess

import hcl_mlir
from hcl_mlir import (get_context, get_insertion_point, get_location,
                      set_insertion_point)

from mlir import passmanager
from mlir.execution_engine import *
from mlir.ir import *

from .base import get_module, get_top_function


def lower(sch,
          name="top",
          binds=None,
          simple_mode=False,
          kernel_only=False,
          stmt=None):
    """Lowering step before build into target
    """
    func = get_top_function()

    # apply optimization passes
    hcl_mlir.loop_transformation(func.operation)
    get_module().dump()

    return get_module()


def build(schedule, target=None, name="top", stmt=None):
    """Build the executable according to the schedule and target.
    """
    lowered_module = lower(schedule)

    if target.mode in ["csyn", "vhls"]:
        return build_fpga_kernel(schedule, target, name, stmt)
    else:
        return build_llvm(schedule, target, name, stmt)


def build_fpga_kernel(schedule, target=None, name="top", stmt=None):
    # make the project folder and copy files
    os.makedirs(target.project, exist_ok=True)
    path = os.path.dirname(__file__)
    path = os.path.join(path, "../harness")
    cp_cmd = "cp {}/vivado/Makefile {}; ".format(path, target.project)
    cp_cmd += "cp {}/vivado/run.tcl {}".format(path, target.project)
    subprocess.Popen(cp_cmd, shell=True, stdout=subprocess.PIPE)

    # generate code
    buf = io.StringIO()
    hcl_mlir.emit_hlscpp(get_module(), buf)
    buf.seek(0)
    hls_code = buf.read()

    # write hls_code to file
    with open("{}/kernel.cpp".format(target.project), "w") as outfile:
        outfile.write(hls_code)
    # host code
    with open("{}/host.cpp".format(target.project), "w") as outfile:
        outfile.write("")

    return hls_code


def lowerToLLVM(module):
    import mlir.conversions
    pm = passmanager.PassManager.parse(
        "reconcile-unrealized-casts")
    pm.run(module)
    return module


def build_llvm(schedule, target=None, name="top", stmt=None):
    with get_context() as ctx, get_location():
        # mod = get_module()
        func = get_top_function()
        func.attributes['llvm.emit_c_interface'] = UnitAttr.get()
        print("\n\nBefore Lowering: ")
        get_module().dump()
        hcl_mlir.lower_hcl_to_llvm(get_module(), ctx)
        # lowerToLLVM(get_module())
        print("lowered.")
        print("\n\nAfter Lowering: ")
        get_module().dump()
        execution_engine = ExecutionEngine(get_module())
        execution_engine.invoke(name)
        print("Execution success")
