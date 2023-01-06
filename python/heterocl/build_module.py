import io
import os
import copy

import hcl_mlir
from hcl_mlir import GlobalInsertionPoint
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import memref
from hcl_mlir.dialects import func as func_d
from hcl_mlir.execution_engine import *
from hcl_mlir.exceptions import *
from hcl_mlir.ir import *
from hcl_mlir.passmanager import PassManager as mlir_pass_manager

from .devices import Platform
from .context import NestedStageLevel, get_context, get_location, set_context, exit_context
from .module import HCLModule, HCLSuperModule
from .operation import placeholder
from .runtime import copy_build_files
from .schedule import Schedule, Stage
from .utils import get_extra_type_hints, hcl_dtype_to_mlir
from .passes.pass_manager import PassManager as ast_pass_manager
from .passes.nest_if import NestElseIf
from .passes.promote_func import PromoteFunc
from .ast.ir_builder import IRBuilder
from .ast import ast


def _mlir_lower_pipeline(module):
    hcl_d.loop_transformation(module)
    pipeline = (
        f"func.func"
        f"(affine-loop-normalize, cse, affine-simplify-structures)"
    )
    try:
        with get_context():
            mlir_pass_manager.parse(pipeline).run(module)
        return module
    except:
        print("Error: failed to run MLIR lower pipeline, printing module...")
        print(module)


def lower(schedule,
          name="top",
          binds=None,
          simple_mode=False,
          kernel_only=False,
          stmt=None):
    """Lowering step before build into target
       by applying optimization pass
    """
    if schedule.is_lowered():
        raise APIError(
                "The module has been lowered. Please apply schedule primitives before the lowering process."
            )
    # HeteroCL Transformation Pipeline
    ast_pm = ast_pass_manager()
    ast_pm.add_pass(NestElseIf)
    ast_pm.add_pass(PromoteFunc)
    device_agnostic_ast = ast_pm.run(schedule.ast)
    schedule._ast = device_agnostic_ast

    # Build MLIR IR
    set_context()
    agnostic_ir_builder = IRBuilder(device_agnostic_ast)
    agnostic_ir_builder.build()
    agnostic_module = agnostic_ir_builder.module
    schedule._module = _mlir_lower_pipeline(agnostic_module)
    schedule._top_func = agnostic_ir_builder.top_func
    exit_context()

    schedule.set_lowered()
    return schedule.module


def build(schedule, target=None, stmt=None, top=None):
    """Build the executable according to the schedule and target.
    """
    try:
        # if isinstance(target, Platform) and str(target.tool.mode) != "debug":
        #     for _, stage in Stage._mapping:
        #         stage.outline()
        if not schedule.is_lowered():
            lower(schedule)
        if top is not None:
            if not isinstance(top, list):
                top = [top]
            modules = []
            for func in top:
                func_mod = func.build(schedule)
                if target is not None:
                    target.top = func.name
                    original_name = target.project
                    target.project = "{}/{}.prj".format(
                        original_name, func.name)
                    modules.append(build_fpga_kernel(func_mod, target, stmt))
                    target.project = original_name
                else:
                    modules.append(build_llvm(func_mod, target, stmt))
            return HCLSuperModule(modules)
        if target is not None:
            return build_fpga_kernel(schedule, target, stmt)
        else:
            return build_llvm(schedule)
    except Exception as e:
        raise e
    finally:
        # TODO: no longer necessary
        hcl_mlir.reset_build_inplace()
        NestedStageLevel.set(0)


def separate_host_xcel(schedule, device_agnostic_ast):
    dfg = schedule._dfg

    if not dfg.has_host_xcel_place():
        # if there is no host-xcel data placement
        # the whole design is offloaded to the device
        return None, device_agnostic_ast

    dfg.create_device_map()
    dfg.graph_partition()
    
    # outline the device function
    dev_func_body = list()
    top_func = device_agnostic_ast.top_func
    for body_op in top_func.body:
        if isinstance(body_op, ast.ComputeOp):
            op_name = body_op.name
            if op_name not in dfg.device_map:
                raise APIError("Cannot find the device map for op {}".format(op_name))
            if dfg.device_map[op_name] in ["FPGA", "device"]:
                dev_func_body.append(body_op)
        elif body_op.is_customize_op:
            dev_func_body.append(body_op)
    
    # create device function
    args = list()
    return_tensors = list()
    for node in dfg.subgraph["inputs"]:
        if node.base is not None:
            args.append(node.base.tensor)
        else:
            args.append(node.tensor)
    for node in dfg.subgraph["outputs"]:
        if node.base is not None:
            return_tensors.append(node.base.tensor)
        else:
            return_tensors.append(node.tensor)
    device_func = ast.FuncOp("top", args, dev_func_body, top_func.loc)
    device_func.level = 0
    device_func.return_tensors = return_tensors

    # create host function
    host_func_body = list()
    call_inserted = False
    new_rets = list()
    for body_op in top_func.body:
        if body_op in dev_func_body:
            if not call_inserted:
                # allocate return tensors
                for t in return_tensors:
                    alloc = ast.AllocOp(t.name + "_host", t.shape, t.dtype, t.loc)
                    alloc.level = body_op.level
                    host_func_body.append(alloc)
                    new_rets.append(alloc)
                # insert a call to device function
                call = ast.CallOp(device_func.name, args + new_rets, [], body_op.loc)
                call.level = body_op.level
                host_func_body.append(call)
                call_inserted = True
        else:
            #TODO: this should be a deep copy
            # because the ast.replace_all_uses_with will affect the original ast
            host_func_body.append(copy.copy(body_op))
    host_func = ast.FuncOp("main", top_func.args, host_func_body, top_func.loc)
    host_func.level = 0
    host_func.return_tensors = top_func.return_tensors

    for old, new in zip(return_tensors, new_rets):
        ast.replace_all_uses_with(host_func, old, new)
    
    # create device function prototype
    device_func_proto = ast.FuncOp(device_func.name, args + return_tensors, [], top_func.loc)
    device_func_proto.level = 0
    device_func_proto.prototype = True

    host_ast = ast.AST(host_func)
    host_ast.region.insert(0, device_func_proto)
    device_ast = ast.AST(device_func)
    return host_ast, device_ast

def separate_host_device_old(schedule):
    xcel_module = schedule.create_xcel_module()
    host_module = schedule.create_host_module()

    # create basic components
    hcl_mlir.enable_build_inplace()
    set_context()
    with get_context(), get_location():
        host_tensors = []
        host_nodes = schedule.DataflowGraph.roots + \
            schedule.DataflowGraph.subgraph["outputs"]
        op_map = {}
        # initialization: create host tensors
        for node in host_nodes:
            tensor = node.tensor
            shape = tensor.shape
            loop_names = ["i{}".format(i) for i in range(len(shape))]
            # create new tensors for host
            host_tensor = placeholder(
                shape, name=tensor.op.name+"_host", dtype=tensor.dtype)
            op_map[tensor.op.name] = {"alloc": host_tensor.op}
            if node in schedule.DataflowGraph.subgraph["inputs"] or node in schedule.DataflowGraph.subgraph["outputs"]:
                host_tensors.append(host_tensor.op.result)
            # create initialization loops
            loops = []
            body_ip = GlobalInsertionPoint.get()
            for i, (ub, loop_name) in enumerate(zip(shape, loop_names)):
                loop = hcl_mlir.make_for(
                    0,
                    ub,
                    step=1,
                    name=loop_name,
                    stage=tensor.op.name+"_host" if i == 0 else "",
                    ip=body_ip,
                )
                loops.append(loop)
                body_ip = InsertionPoint(loop.body.operations[0])
            GlobalInsertionPoint.save(body_ip)
            cst = hcl_mlir.ConstantOp(tensor.op.dtype, 0)
            store = hcl_mlir.StoreOp(
                cst, host_tensor.op, [hcl_mlir.IterVar(loop.induction_variable, name=loop_name) for loop, loop_name in zip(loops, loop_names)])
            GlobalInsertionPoint.restore()
        # fix device top function signature
        func_op = schedule.xcel_top
        function_type = FunctionType.get(
            inputs=[node.tensor.memref_type
                    for node in schedule.DataflowGraph.subgraph["inputs"]],
            results=[node.tensor.memref_type for node in schedule.DataflowGraph.subgraph["outputs"]])
        func_op.attributes["function_type"] = TypeAttr.get(function_type)
        func_op.attributes["inputs"] = StringAttr.get(
            ",".join([node.tensor.name+"_xcel" for node in schedule.DataflowGraph.subgraph["inputs"]]))
        itypes = "".join([get_extra_type_hints(
            node.tensor.op.dtype) for node in schedule.DataflowGraph.subgraph["inputs"]])
        func_op.attributes["itypes"] = StringAttr.get(itypes)
        func_op.attributes["outputs"] = StringAttr.get(
            ",".join([node.tensor.name+"_xcel" for node in schedule.DataflowGraph.subgraph["outputs"]]))
        otypes = "".join([get_extra_type_hints(
            node.tensor.op.dtype) for node in schedule.DataflowGraph.subgraph["outputs"]])
        func_op.attributes["itypes"] = StringAttr.get(otypes)
        # preparation: create operation mapping
        for op in schedule.xcel_module.body.operations:
            if "Stage_" in str(op.name):
                name = str(op.name)[1:-1].split("_")[1]  # omit quotation mark
                if name not in op_map:
                    op_map[name] = {"func": op}
        for op in schedule.xcel_top.entry_block.operations:
            if isinstance(op, memref.AllocOp):
                name = str(op.attributes["name"])[1:-1]
                if "alloc" not in op_map[name]:
                    op_map[name]["alloc"] = op
                else:
                    op_map[name]["xcel"] = op
            elif isinstance(op, func_d.CallOp):
                name = str(op.attributes["callee"]).split("_")[1]
                op_map[name]["call"] = op
        for i, param in enumerate(func_op.arguments):
            name = schedule.DataflowGraph.subgraph["inputs"][i].name
            op_map[name]["xcel"] = param
        # traverse the dfg (BFS) and move ops to host based on device_map
        working_set = [node for node in schedule.DataflowGraph.roots]
        flag = False
        while len(working_set) > 0:
            working_node = working_set.pop(0)
            name = working_node.name
            if working_node not in host_nodes and schedule.DataflowGraph.device_map[name] == "CPU":
                op_map[name]["func"].move_before(schedule._host_top)
                if "alloc" in op_map[name]:
                    op_map[name]["alloc"].move_before(schedule._host_ret)
                op_map[name]["call"].move_before(schedule._host_ret)
                # update reference
                for i, parent in enumerate(working_node.parents):
                    if "alloc" in op_map[parent.name]:
                        op_map[name]["call"].operands[i] = op_map[parent.name]["alloc"].result
            elif schedule.DataflowGraph.device_map[name] == "FPGA":
                if not flag:
                    flag = True
                    # call device function
                    for node in schedule.DataflowGraph.subgraph["inputs"]:
                        if node not in schedule.DataflowGraph.roots:
                            host_tensors.insert(
                                0, op_map[node.name]["alloc"].result)
                    call_op = hcl_mlir.CallOp(None, "top", host_tensors)
                    call_op.built_op.attributes["inputs"] = StringAttr.get(
                        ",".join([node.tensor.name for node in schedule.DataflowGraph.subgraph["inputs"]]))
                    call_op.built_op.attributes["outputs"] = StringAttr.get(
                        ",".join([node.tensor.name for node in schedule.DataflowGraph.subgraph["outputs"]]))
                # update reference
                for i, parent in enumerate(working_node.parents):
                    if parent.base is not None:
                        op_dict = op_map[parent.base.name]
                    else:
                        op_dict = op_map[parent.name]
                    if "xcel" in op_dict:
                        if isinstance(op_dict["xcel"], hcl_mlir.BlockArgument):
                            op_map[name]["call"].operands[i] = op_dict["xcel"]
                        else:
                            op_map[name]["call"].operands[i] = op_dict["xcel"].result
                    else:
                        op_map[name]["call"].operands[i] = op_dict["alloc"].result
                if working_node in schedule.DataflowGraph.subgraph["outputs"]:
                    if working_node.base is not None:
                        op_dict = op_map[working_node.base.name]
                    else:
                        op_dict = op_map[working_node.name]
                    if "xcel" in op_dict:
                        if isinstance(op_dict["xcel"], hcl_mlir.BlockArgument):
                            schedule._xcel_ret.operands[0] = op_dict["xcel"]
                        else:
                            schedule._xcel_ret.operands[0] = op_dict["xcel"].result
                    else:
                        schedule._xcel_ret.operands[0] = op_dict["alloc"].result

            for child in working_node.children:
                working_set.append(child)

    hcl_mlir.disable_build_inplace()


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
        tensor = node.tensor
        with get_context():
            arg = hcl_mlir.print_mlir_type(
                hcl_dtype_to_mlir(tensor.dtype)) + " " + tensor.name
        for index in tensor.shape:
            arg += "[{}]".format(index)
        args.append(arg)
    header += ", ".join(args)
    header += ");\n\n#endif // KERNEL_H"
    return header


def build_fpga_kernel(schedule, target=None, stmt=None):
    if isinstance(schedule, Schedule):
        module = schedule.module
    else:
        module = schedule
    if target == "vhls":
        buf = io.StringIO()
        hcl_d.emit_vhls(module, buf)
        buf.seek(0)
        hls_code = buf.read()
        return hls_code
    elif target == "ihls":
        buf = io.StringIO()
        hcl_d.emit_ihls(module, buf)
        buf.seek(0)
        hls_code = buf.read()
        return hls_code
    elif not isinstance(target, Platform):
        raise RuntimeError("Not supported target")

    if str(target.tool.mode) == "debug":
        # debug mode: full code without host-xcel partition
        # is generated and written to kernel.cpp
        # host.cpp is kept empty
        # make the project folder and copy files
        copy_build_files(target)
        buf = io.StringIO()
        hcl_d.emit_vhls(module, buf)
        buf.seek(0)
        hls_code = buf.read()
        with open("{}/kernel.cpp".format(target.project), "w") as outfile:
            outfile.write(hls_code)
        host_code = None
        with open("{}/host.cpp".format(target.project), "w") as outfile:
            outfile.write("")

    else:
        # release mode: host-xcel partition is generated
        # and written to kernel.cpp and host.cpp
        device_agnostic_ast = schedule.ast
        # Separate host and device
        host_ast, xcel_ast = separate_host_xcel(schedule, device_agnostic_ast)
        
        set_context()
        xcel_ir_builder = IRBuilder(xcel_ast)
        xcel_ir_builder.build()
        xcel_module = xcel_ir_builder.module
        schedule._xcel_module = _mlir_lower_pipeline(xcel_module)
        exit_context()

        set_context()
        host_ir_builder = IRBuilder(host_ast)
        host_ir_builder.build()
        host_module = host_ir_builder.module
        schedule._host_module = host_module
        exit_context()

        # make the project folder and copy files
        copy_build_files(target)

        # generate xcel code
        buf = io.StringIO()
        hcl_d.emit_vhls(schedule.xcel_module, buf)
        buf.seek(0)
        hls_code = buf.read()
        with open("{}/kernel.cpp".format(target.project), "w") as outfile:
            outfile.write(hls_code)

        # generate host code
        host_buf = io.StringIO()
        hcl_d.emit_vhls(schedule.host_module, host_buf)
        host_buf.seek(0)
        host_code = host_buf.read()
        with open("{}/host.cpp".format(target.project), "w") as outfile:
            outfile.write(host_code)

        # generate header
        header = generate_kernel_header(schedule)
        with open("{}/kernel.h".format(target.project), "w") as outfile:
            outfile.write(header)

    hcl_module = HCLModule(target.top, hls_code, target, host_src=host_code)
    return hcl_module


def build_llvm(schedule, top_func_name="top"):

    def attach_llvm_attrs(module):
        # find top func op
        func = None
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp) and op.name.value == top_func_name:
                func = op
                break
        if func is None:
            raise APIError("No top-level function found in the built MLIR module")
        func.attributes['llvm.emit_c_interface'] = UnitAttr.get()
        func.attributes[top_func_name] = UnitAttr.get()
        func.attributes['sym_name'] = StringAttr.get("top")

    with get_context() as ctx, get_location():
        if isinstance(schedule, Schedule):
            attach_llvm_attrs(schedule.module)
            module = Module.parse(str(schedule.module), ctx)
        else:
            module = Module.parse(str(schedule), ctx)
            attach_llvm_attrs(module)

        host_src = Module.parse(str(module))

        # memref dce should precede lower_composite_type
        hcl_d.memref_dce(module) 
        hcl_d.lower_composite_type(module)
        hcl_d.lower_fixed_to_int(module)
        hcl_d.lower_print_ops(module)
        hcl_d.lower_anywidth_int(module)
        # Note: lower_any_width_int should precede
        # move_return_to_input, because it uses input/output
        # type hints.
        hcl_d.move_return_to_input(module)
        hcl_d.lower_bit_ops(module)
        hcl_d.legalize_cast(module)
        hcl_d.remove_stride_map(module)
        hcl_d.lower_hcl_to_llvm(module, ctx)
        
        # Add shared library
        if os.getenv("LLVM_BUILD_DIR") is not None:
            shared_libs = [
                os.path.join(os.getenv("LLVM_BUILD_DIR"),
                            'lib', 'libmlir_runner_utils.so'),
                os.path.join(os.getenv("LLVM_BUILD_DIR"),
                            'lib', 'libmlir_c_runner_utils.so')
            ]
        else:
            APIWarning("LLVM_BUILD_DIR is not set, print memref feature is not available.").warn()
            shared_libs = None

        if shared_libs is not None:
            execution_engine = ExecutionEngine(module, opt_level=0, shared_libs=shared_libs)
        else:
            execution_engine = ExecutionEngine(module, opt_level=0)
        hcl_module = HCLModule(top_func_name, execution_engine,
                               "llvm", host_src=host_src, return_num=0)
        return hcl_module
