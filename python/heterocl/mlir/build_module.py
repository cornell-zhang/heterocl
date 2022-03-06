import io

import hcl_mlir
from hcl_mlir import GlobalInsertionPoint
from hcl_mlir.dialects import affine
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.execution_engine import *
from hcl_mlir.ir import *

from ..devices import Platform
from .context import get_context, get_location
from .module import HCLModule
from .operation import placeholder
from .runtime import copy_build_files
from .schedule import Schedule
from .utils import get_extra_type_hints


def lower(schedule,
          name="top",
          binds=None,
          simple_mode=False,
          kernel_only=False,
          stmt=None):
    """Lowering step before build into target
       by applying optimization pass
    """
    hcl_d.loop_transformation(schedule.device_module)
    return schedule.device_module


def build(schedule, target=None, name="top", stmt=None):
    """Build the executable according to the schedule and target.
    """
    try: 
        lowered_module = lower(schedule)
        if target != None:
            return build_fpga_kernel(schedule, target, name, stmt)
        else:
            return build_llvm(schedule, target, name, stmt)
    except Exception as e:
        raise e
    finally:
        hcl_mlir.reset_build_inplace()


def separate_host_device(schedule):
    xcel_module = schedule.create_xcel_module()
    host_module = schedule.create_host_module()
    extern_module = schedule.create_extern_module()

    # create basic components
    hcl_mlir.enable_build_inplace()
    with get_context(), get_location():
        host_tensors = []
        host_nodes = schedule.DataflowGraph.roots + \
            schedule.DataflowGraph.subgraph["outputs"]
        for node in host_nodes:
            tensor = node.tensor
            shape = tensor.shape
            loop_names = ["i{}".format(i) for i in range(len(shape))]
            # create new tensors for host
            host_tensor = placeholder(
                shape, name=tensor.op.name+"_host", dtype=tensor.dtype)
            host_tensor.build()
            if node in schedule.DataflowGraph.subgraph["outputs"]:
                host_tensors.append(host_tensor.op.result)
            # create initialization loops
            loops = []
            body_ip = GlobalInsertionPoint.get()
            for i, (ub, loop_name) in enumerate(zip(shape, loop_names)):
                loop = hcl_mlir.make_affine_for(
                    0,
                    ub,
                    step=1,
                    name=loop_name,
                    stage=tensor.op.name+"_host" if i == 0 else "",
                    ip=body_ip,
                )
                if i != 0:  # manually add terminator!
                    affine.AffineYieldOp([], ip=body_ip)
                loops.append(loop)
                body_ip = InsertionPoint(loop.body)
            GlobalInsertionPoint.save(body_ip)
            cst = hcl_mlir.ConstantOp(tensor.op.dtype, 0)
            store = hcl_mlir.StoreOp(
                cst, host_tensor.op, [hcl_mlir.IterVar(loop.induction_variable) for loop in loops])
            GlobalInsertionPoint.restore()
        # call device function
        host_tensors = [
            node.tensor.result for node in schedule.DataflowGraph.subgraph["inputs"]] + host_tensors
        call_op = hcl_mlir.CallOp(None, "top", host_tensors)
        call_op.built_op.attributes["inputs"] = StringAttr.get(
            ",".join([node.tensor.name for node in schedule.DataflowGraph.subgraph["inputs"]]))
        call_op.built_op.attributes["outputs"] = StringAttr.get(
            ",".join([node.tensor.name for node in schedule.DataflowGraph.subgraph["outputs"]]))
        # fix device top function signature
        func_op = schedule.xcel_top
        function_type = FunctionType.get(
            inputs=[node.tensor.memref_type
                    for node in schedule.DataflowGraph.subgraph["inputs"]],
            results=[node.tensor.memref_type for node in schedule.DataflowGraph.subgraph["outputs"]])
        func_op.attributes["type"] = TypeAttr.get(function_type)
        func_op.attributes["inputs"] = StringAttr.get(
            ",".join([node.tensor.name+"_xcel" for node in schedule.DataflowGraph.subgraph["inputs"]]))
        extra_itypes = "".join([get_extra_type_hints(
            node.tensor.op.dtype) for node in schedule.DataflowGraph.subgraph["inputs"]])
        func_op.attributes["extra_itypes"] = StringAttr.get(extra_itypes)
        func_op.attributes["outputs"] = StringAttr.get(
            ",".join([node.tensor.name+"_xcel" for node in schedule.DataflowGraph.subgraph["outputs"]]))
        extra_otypes = "".join([get_extra_type_hints(
            node.tensor.op.dtype) for node in schedule.DataflowGraph.subgraph["outputs"]])
        func_op.attributes["extra_itypes"] = StringAttr.get(extra_otypes)
    hcl_mlir.disable_build_inplace()

    # call C++ pass to further fix the references
    device_map = schedule.DataflowGraph.device_map
    subgraph_name = {}
    subgraph_name["inputs"] = [
        node.name for node in schedule.DataflowGraph.subgraph["inputs"]]
    subgraph_name["outputs"] = [
        node.name for node in schedule.DataflowGraph.subgraph["outputs"]]
    roots = [node.name for node in schedule.DataflowGraph.roots]
    hcl_d.host_device_separation(
        host_module, xcel_module, extern_module, device_map, roots, subgraph_name)
    host_module.dump()
    xcel_module.dump()
    extern_module.dump()


def generate_kernel_header(schedule):
    header = """#ifndef KERNEL_H
#define KERNEL_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

void top("""
    all_inputs_outputs = schedule.DataflowGraph.subgraph["inputs"] + \
        schedule.DataflowGraph.subgraph["outputs"]
    args = []
    for node in all_inputs_outputs:
        tensor = node.tensor.op
        with get_context():
            arg = hcl_mlir.print_mlir_type(
                hcl_mlir.get_mlir_type(tensor.dtype)) + " " + tensor.name
        for index in tensor.shape:
            arg += "[{}]".format(index)
        args.append(arg)
    header += ", ".join(args)
    header += ");\n\n#endif // KERNEL_H"
    return header


def build_fpga_kernel(schedule, target=None, name="top", stmt=None):
    if target == "vhls":
        buf = io.StringIO()
        hcl_d.emit_hlscpp(schedule.device_module, buf)
        buf.seek(0)
        hls_code = buf.read()
        return hls_code
    elif not isinstance(target, Platform):
        raise RuntimeError("Not supported target")

    # make the project folder and copy files
    copy_build_files(target)

    # data placement
    if not hcl_mlir.EXTRACT_FUNCTION:
        raise RuntimeError("Should set hcl_mlir.EXTRACT_FUNCTION to True!")
    schedule.DataflowGraph.graph_partition()
    separate_host_device(schedule)

    # generate xcel code
    buf = io.StringIO()
    hcl_d.emit_hlscpp(schedule.xcel_module, buf)
    buf.seek(0)
    hls_code = buf.read()
    with open("{}/kernel.cpp".format(target.project), "w") as outfile:
        outfile.write(hls_code)

    # generate host code
    host_buf = io.StringIO()
    hcl_d.emit_hlscpp(schedule.host_module, host_buf)
    host_buf.seek(0)
    host_code = host_buf.read()
    with open("{}/host.cpp".format(target.project), "w") as outfile:
        outfile.write(host_code)

    # generate extern code
    extern_buf = io.StringIO()
    hcl_d.emit_hlscpp(schedule.extern_module, extern_buf)
    extern_buf.seek(0)
    extern_code = extern_buf.read()
    with open("{}/extern.cpp".format(target.project), "w") as outfile:
        outfile.write(extern_code)

    # generate header
    header = generate_kernel_header(schedule)
    with open("{}/kernel.h".format(target.project), "w") as outfile:
        outfile.write(header)

    hcl_module = HCLModule(name, hls_code, target, host_src=host_code)
    return hcl_module


def build_llvm(schedule, target=None, name="top", stmt=None):
    with get_context() as ctx, get_location():
        func = schedule.device_top
        func.attributes['llvm.emit_c_interface'] = UnitAttr.get()
        # module = schedule.device_module
        module = Module.parse(str(schedule.device_module), ctx)
        # module.dump()
        hcl_d.loop_transformation(module)
        hcl_d.lower_fixed_to_int(module)
        hcl_d.lower_hcl_to_llvm(module, ctx)
        num_results = len(func.type.results)
        # print("lowered.")
        # print("\n\nAfter Lowering: ")
        # module.dump()
        execution_engine = ExecutionEngine(module)
        hcl_module = HCLModule(name, execution_engine,
                               "llvm", ctx, return_num=num_results)
        return hcl_module
