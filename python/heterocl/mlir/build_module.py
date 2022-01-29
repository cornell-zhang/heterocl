import io

import hcl_mlir
from hcl_mlir import GlobalInsertionPoint, get_context, get_location
import hcl_mlir.affine as affine

from mlir import passmanager
from mlir.execution_engine import *
from mlir.ir import *
from mlir.dialects import std

from .module import HCLModule
from .operation import placeholder
from .runtime import copy_build_files
from .schedule import Schedule


def lower(schedule,
          name="top",
          binds=None,
          simple_mode=False,
          kernel_only=False,
          stmt=None):
    """Lowering step before build into target
       by applying optimization pass
    """
    hcl_mlir.loop_transformation(schedule.device_module)
    return schedule.device_module


def build(schedule, target=None, name="top", stmt=None):
    """Build the executable according to the schedule and target.
    """
    lowered_module = lower(schedule)

    if target != None:
        return build_fpga_kernel(schedule, target, name, stmt)
    else:
        return build_llvm(schedule, target, name, stmt)


def separate_host_device(schedule):
    xcel_module = schedule.create_xcel_module()
    host_module = schedule.create_host_module()

    # create basic components
    hcl_mlir.enable_build_inplace()
    with get_context(), get_location():
        host_tensors = []
        host_nodes = Schedule._DataflowGraph.roots + \
            Schedule._DataflowGraph.subgraph["outputs"]
        for node in host_nodes:
            tensor = node.tensor
            shape = tensor.shape
            loop_names = ["i{}".format(i) for i in range(len(shape))]
            # create new tensors for host
            host_tensor = placeholder(
                shape, name=tensor.name+"_host", dtype=tensor.dtype)
            host_tensor.build()
            if node in Schedule._DataflowGraph.subgraph["outputs"]:
                host_tensors.append(host_tensor.result)
            # create initialization loops
            loops = []
            body_ip = GlobalInsertionPoint.get()
            for i, (ub, loop_name) in enumerate(zip(shape, loop_names)):
                loop = hcl_mlir.make_affine_for(
                    0,
                    ub,
                    step=1,
                    name=loop_name,
                    stage=tensor.name+"_host" if i == 0 else "",
                    ip=body_ip,
                )
                if i != 0:  # manually add terminator!
                    affine.AffineYieldOp([], ip=body_ip)
                loops.append(loop)
                body_ip = InsertionPoint(loop.body)
            GlobalInsertionPoint.save(body_ip)
            cst = hcl_mlir.ConstantOp(tensor.dtype, 0)
            store = hcl_mlir.StoreOp(
                cst, host_tensor, [hcl_mlir.IterVar(loop.induction_variable) for loop in loops])
            GlobalInsertionPoint.restore()
        # call device function
        host_tensors = [
            node.tensor.result for node in Schedule._DataflowGraph.subgraph["inputs"]] + host_tensors
        call_op = hcl_mlir.CallOp(None, "top", host_tensors)
        call_op.built_op.attributes["inputs"] = StringAttr.get(
            ",".join([node.tensor.name for node in Schedule._DataflowGraph.subgraph["inputs"]]))
        call_op.built_op.attributes["outputs"] = StringAttr.get(
            ",".join([node.tensor.name for node in Schedule._DataflowGraph.subgraph["outputs"]]))
        # fix device top function signature
        func_op = schedule.xcel_top
        function_type = FunctionType.get(
            inputs=[node.tensor.get_memref_type()
                    for node in Schedule._DataflowGraph.subgraph["inputs"]],
            results=[node.tensor.get_memref_type() for node in Schedule._DataflowGraph.subgraph["outputs"]])
        func_op.attributes["type"] = TypeAttr.get(function_type)
        func_op.attributes["inputs"] = StringAttr.get(
            ",".join([node.tensor.name+"_device" for node in Schedule._DataflowGraph.subgraph["inputs"]]))
        func_op.attributes["outputs"] = StringAttr.get(
            ",".join([node.tensor.name+"_device" for node in Schedule._DataflowGraph.subgraph["outputs"]]))
    hcl_mlir.disable_build_inplace()

    # call C++ pass to further fix the references
    device_map = Schedule._DataflowGraph.device_map
    subgraph_name = {}
    subgraph_name["inputs"] = [
        node.name for node in Schedule._DataflowGraph.subgraph["inputs"]]
    subgraph_name["outputs"] = [
        node.name for node in Schedule._DataflowGraph.subgraph["outputs"]]
    roots = [node.name for node in Schedule._DataflowGraph.roots]
    hcl_mlir.host_device_separation(
        host_module, xcel_module, device_map, roots, subgraph_name)
    host_module.dump()
    xcel_module.dump()

def generate_kernel_header(schedule):
    header = """#ifndef KERNEL_H
#define KERNEL_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

void top("""
    all_inputs_outputs = Schedule._DataflowGraph.subgraph["inputs"] + \
        Schedule._DataflowGraph.subgraph["outputs"]
    args = []
    for node in all_inputs_outputs:
        tensor = node.tensor
        arg = hcl_mlir.print_mlir_type(tensor.dtype) + " " + tensor.name
        for index in tensor.shape:
            arg += "[{}]".format(index)
        args.append(arg)
    header += ", ".join(args)
    header += ");\n\n#endif // KERNEL_H"
    return header


def build_fpga_kernel(schedule, target=None, name="top", stmt=None):
    # make the project folder and copy files
    copy_build_files(target)

    # data placement
    Schedule._DataflowGraph.graph_partition()
    separate_host_device(schedule)

    # generate device code
    buf = io.StringIO()
    hcl_mlir.emit_hlscpp(schedule.device_module, buf)
    buf.seek(0)
    hls_code = buf.read()

    # write HLS code to file
    with open("{}/kernel.cpp".format(target.project), "w") as outfile:
        outfile.write(hls_code)

    # generate host code
    host_module = schedule.host_module
    host_buf = io.StringIO()
    hcl_mlir.emit_hlscpp(host_module, host_buf)
    host_buf.seek(0)
    host_code = host_buf.read()

    with open("{}/host.cpp".format(target.project), "w") as outfile:
        outfile.write(host_code)

    # generate header
    header = generate_kernel_header(schedule)
    with open("{}/kernel.h".format(target.project), "w") as outfile:
        outfile.write(header)

    hcl_module = HCLModule(name, hls_code, target, host_src=host_code)
    return hcl_module


def reconcile_unrealized_casts(module):
    # TODO(Niansong): not useful for now, consider removal
    import mlir.conversions
    pm = passmanager.PassManager.parse(
        "reconcile-unrealized-casts")
    pm.run(module)
    return module


def build_llvm(schedule, target=None, name="top", stmt=None):
    with get_context() as ctx, get_location():
        func = schedule.device_top
        func.attributes['llvm.emit_c_interface'] = UnitAttr.get()
        # print("\n\nBefore Lowering: ")
        # schedule.device_module.dump()
        hcl_mlir.lower_hcl_to_llvm(schedule.device_module, ctx)
        # print("lowered.")
        # print("\n\nAfter Lowering: ")
        # schedule.device_module.dump()
        execution_engine = ExecutionEngine(schedule.device_module)
        hcl_module = HCLModule(name, execution_engine, "llvm", ctx)
        return hcl_module
