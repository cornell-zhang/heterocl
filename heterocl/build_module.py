# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, unused-argument

import io
import os
import copy

import hcl_mlir
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import func as func_d
from hcl_mlir.execution_engine import ExecutionEngine
from hcl_mlir.exceptions import APIError, PassWarning
from hcl_mlir.ir import (
    Module,
    StringAttr,
    UnitAttr,
)
from hcl_mlir.passmanager import PassManager as mlir_pass_manager

from .devices import Platform
from .context import get_context, get_location, set_context, exit_context
from .module import HCLModule, HCLSuperModule
from .runtime import copy_build_files
from .schedule import Schedule
from .utils import hcl_dtype_to_mlir
from .ast.ir_builder import IRBuilder
from .ast.build_cleaner import ASTCleaner
from .ast import ast


def _mlir_lower_pipeline(module):
    hcl_d.loop_transformation(module)
    pipeline = "func.func(affine-loop-normalize, cse, affine-simplify-structures)"
    try:
        with get_context():
            mlir_pass_manager.parse(pipeline).run(module)
        return module
    except Exception as e:
        print("Error: failed to run MLIR lower pipeline, printing module...")
        print(module)
        raise e


def lower(
    schedule, name="top", binds=None, simple_mode=False, kernel_only=False, stmt=None
):
    """Lowering step before build into target
    by applying optimization pass
    """
    schedule._module = _mlir_lower_pipeline(schedule._module)
    schedule.set_lowered()
    return schedule.module


def build(schedule, target=None, stmt=None, top=None):
    """Build the executable according to the schedule and target."""
    # pylint: disable=too-many-try-statements
    try:
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
                    target.project = f"{original_name}/{func.name}.prj"
                    modules.append(build_fpga_kernel(func_mod, target, stmt))
                    target.project = original_name
                else:
                    raise RuntimeError("Untested code path.")
                    # modules.append(build_llvm(func_mod, target, stmt))
            return HCLSuperModule(modules)
        if target is not None:
            return build_fpga_kernel(schedule, target, stmt)
        return build_llvm(schedule)
    except Exception as e:
        raise e


def separate_host_xcel(schedule, device_agnostic_ast):
    dfg = schedule._dfg

    if not dfg.has_host_xcel_place():
        # if there is no host-xcel data placement
        # the whole design is offloaded to the device
        return None, device_agnostic_ast

    dfg.create_device_map()
    dfg.graph_partition()

    # outline the device function
    dev_func_body = []
    top_func = device_agnostic_ast.top_func
    for body_op in top_func.body:
        if isinstance(body_op, ast.ComputeOp):
            op_name = body_op.name
            if op_name not in dfg.device_map:
                raise APIError(f"Cannot find the device map for op {op_name}")
            if dfg.device_map[op_name] in {"FPGA", "device"}:
                dev_func_body.append(body_op)
        elif body_op.is_customize_op:
            dev_func_body.append(body_op)

    # create device function
    args = []
    return_tensors = []
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
    host_func_body = []
    call_inserted = False
    new_rets = []
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
            # TODO: this should be a deep copy
            # because the ast.replace_all_uses_with will affect the original ast
            host_func_body.append(copy.copy(body_op))
    host_func = ast.FuncOp("main", top_func.args, host_func_body, top_func.loc)
    host_func.level = 0
    host_func.return_tensors = top_func.return_tensors

    for old, new in zip(return_tensors, new_rets):
        ast.replace_all_uses_with(host_func, old, new)

    # create device function prototype
    device_func_proto = ast.FuncOp(
        device_func.name, args + return_tensors, [], top_func.loc
    )
    device_func_proto.level = 0
    device_func_proto.prototype = True

    host_ast = ast.AST(host_func)
    host_ast.region.insert(0, device_func_proto)
    device_ast = ast.AST(device_func)
    return host_ast, device_ast


def generate_kernel_header(schedule):
    header = """#ifndef KERNEL_H
#define KERNEL_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

void top("""
    all_inputs_outputs = (
        schedule.DataflowGraph.subgraph["inputs"]
        + schedule.DataflowGraph.subgraph["outputs"]
    )
    args = []
    for node in all_inputs_outputs:
        tensor = node.tensor
        with get_context():
            arg = (
                hcl_mlir.print_mlir_type(hcl_dtype_to_mlir(tensor.dtype))
                + " "
                + tensor.name
            )
        for index in tensor.shape:
            arg += f"[{index}]"
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
    if target == "ihls":
        buf = io.StringIO()
        hcl_d.emit_ihls(module, buf)
        buf.seek(0)
        hls_code = buf.read()
        return hls_code
    if not isinstance(target, Platform):
        raise RuntimeError("Not supported target")

    # pylint: disable=no-else-return
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
        with open(f"{target.project}/kernel.cpp", "w", encoding="utf-8") as outfile:
            outfile.write(hls_code)
        host_code = None
        with open(f"{target.project}/host.cpp", "w", encoding="utf-8") as outfile:
            outfile.write("")

        return hls_code

    else:
        # release mode: host-xcel partition is generated
        # and written to kernel.cpp and host.cpp
        device_agnostic_ast = schedule.ast
        # Clean device_agnostic_ast build results
        ast_cleaner = ASTCleaner()
        ast_cleaner.visit(device_agnostic_ast)
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
        with open(f"{target.project}/kernel.cpp", "w", encoding="utf-8") as outfile:
            outfile.write(hls_code)

        # generate host code
        host_buf = io.StringIO()
        hcl_d.emit_vhls(schedule.host_module, host_buf)
        host_buf.seek(0)
        host_code = host_buf.read()
        with open(f"{target.project}/host.cpp", "w", encoding="utf-8") as outfile:
            outfile.write(host_code)

        # generate header
        header = generate_kernel_header(schedule)
        with open(f"{target.project}/kernel.h", "w", encoding="utf-8") as outfile:
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
        func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        func.attributes[top_func_name] = UnitAttr.get()
        func.attributes["sym_name"] = StringAttr.get("top")

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
        # print(module)
        hcl_d.legalize_cast(module)
        hcl_d.remove_stride_map(module)
        pipeline = "lower-affine,func.func(buffer-loop-hoisting)"
        try:
            with get_context():
                mlir_pass_manager.parse(pipeline).run(module)
        except Exception as e:  # pylint: disable=broad-exception-caught
            PassWarning(str(e)).warn()
            print(module)

        hcl_d.lower_hcl_to_llvm(module, ctx)

        # Add shared library
        if os.system("which llvm-config >> /dev/null") != 0:
            raise APIError(
                "llvm-config is not found in PATH, llvm is not installed or not in PATH."
            )

        lib_path = os.popen("llvm-config --libdir").read().strip()
        shared_libs = [
            os.path.join(lib_path, "libmlir_runner_utils.so"),
            os.path.join(lib_path, "libmlir_c_runner_utils.so"),
        ]
        opt_level = 3
        if shared_libs is not None:
            execution_engine = ExecutionEngine(
                module, opt_level=opt_level, shared_libs=shared_libs
            )
        else:
            execution_engine = ExecutionEngine(module, opt_level=opt_level)
        hcl_module = HCLModule(
            top_func_name, execution_engine, "llvm", host_src=host_src, return_num=0
        )
        return hcl_module
