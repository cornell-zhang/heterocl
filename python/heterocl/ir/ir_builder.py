# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

import inspect
from typing import List, Callable

from . import intermediate as itmd
from ..context import *
from ..tensor import Tensor
import hcl_mlir
from ..utils import hcl_dtype_to_mlir
from .. import types as htypes
from ..type_infer import TypeInfer
# Import MLIR dialects
# Naming rule: import dialect as dialect_d 
from hcl_mlir.dialects import \
    func as func_d, hcl as hcl_d, \
    scf as scf_d, memref as memref_d, \
    affine as affine_d
from hcl_mlir.exceptions import *

class IRBuilder(object):
    """IRBuilder class to build MLIR
    operations from intermediate layer
    """
    def __init__(self, intermediate):
        self._intermediate = intermediate
        self.module = Module.create(get_location())
        self.iv = [] # TODO(Niansong): what is this list of iv for?
        self.tinf_engine = TypeInfer()

    def build(self):
        """Builder entry point
        
        Build MLIR module with a top-level function
        """
        top_func = self._intermediate.top_func
        with get_context(), get_location():
            input_types = []
            for tensor in top_func.args:
                ele_type = hcl_dtype_to_mlir(tensor.dtype)
                memref_type = MemRefType.get(tensor.shape, ele_type)
                input_types.append(memref_type)
            ip = InsertionPoint(self.module.body)
            func = func_d.FuncOp(name=top_func.name, type=FunctionType.get(
                inputs=input_types, results=[]), ip=ip)
            func.add_entry_block()

            # Set alloc op's result as function block arg
            for alloc_op, arg in zip(top_func.args, func.entry_block.arguments):
                alloc_op.result = arg

            # build build's body op
            ip = InsertionPoint(func.entry_block)
            for op in top_func.body:
                self.build_visitor(op, ip)
            func_d.ReturnOp([], ip=ip)


    def build_visitor(self, op, ip):
        """Build dispatcher
        
        Build MLIR operation from intermediate layer
        """
        if isinstance(op, itmd.ComputeOp):
            self.build_compute(op, ip)
        elif isinstance(op, itmd.AllocOp):
            self.build_alloc_op(op, ip)
        elif isinstance(op, itmd.BinaryOp):
            self.build_binary_op(op, ip)
        elif isinstance(op, itmd.LoadOp):
            self.build_load_op(op, ip)
        elif isinstance(op, itmd.StoreOp):
            self.build_store_op(op, ip)
        elif isinstance(op, itmd.ConstantOp):
            self.build_constant_op(op, ip)
        else:
            raise HCLNotImplementedError(f"{type(op)}'s build visitor is not implemented yet.")


    def build_compute(self, op, ip):
        # TODO(Niansong): use real loop names
        arg_names = ["i%d" % i for i in range(len(op.shape))]
        with get_context(), get_location():
            # build output tensor
            allocOp = itmd.AllocOp(op.name, op.shape, op.dtype, op.loc)
            self.build_visitor(allocOp, ip)
            
            loops = list()
            for i, (ub, loop_name) in enumerate(zip(op.shape, arg_names)):
                loop = hcl_mlir.make_for(
                    0,
                    ub,
                    step=1,
                    name=loop_name,
                    stage=(op.name if i == 0 else ""),
                    ip=ip,
                )
                loops.append(loop)
                ip = InsertionPoint(loop.body.operations[0])

            iter_var = [hcl_mlir.IterVar(loop.induction_variable, name=loop_name)
                for loop, loop_name in zip(loops, arg_names)]
            result_expr = op.body(*iter_var)
            # visit the result expression
            # here ip is the innermost loop
            self.build_visitor(result_expr, ip)
            store_op = itmd.StoreOp(allocOp, iter_var, result_expr, op.loc)
            self.build_visitor(store_op, ip)

    def build_alloc_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        ele_type = hcl_dtype_to_mlir(op.dtype)
        memref_type = MemRefType.get(op.shape, ele_type)
        op.result = memref_d.AllocOp(memref_type, [], [], ip=ip, loc=loc).result

    def build_binary_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)

        # Step 1: build lhs and rhs
        self.build_visitor(op.lhs, ip)
        self.build_visitor(op.rhs, ip)

        # Step 2: cast lhs and rhs to the same type
        t = self.tinf_engine.infer(op)
        lhs = itmd.CastOp(op.lhs, t, op.loc)
        rhs = itmd.CastOp(op.rhs, t, op.loc)
        self.build_visitor(lhs, ip)
        self.build_visitor(rhs, ip)

        # Step 3: build binary op

        # Step 4: attach necessary attributes

    def build_load_op(self, op : itmd.LoadOp, ip):
        index_exprs = []
        flag = True
        load_op = None
        for index in op.index:
            try:
                affine_expr = self.build_affine_expr(index)
                index_exprs.append(affine_expr)
            except:
                flag = False
                break
        if flag:
            affine_map = AffineMap.get(
                dim_count=len(op.index), symbol_count=0, exprs=index_exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            load_op = affine_d.AffineLoadOp(
                op.tensor.result,
                [idx.result for idx in op.index],
                affine_attr,
                ip=ip
            )
            op.result = load_op.result
        else:
            new_indices = []
            for index in op.index:
                new_indices.append(index.result)
            load_op = memref_d.LoadOp(op.tensor.result, new_indices, ip=ip)
            op.result = load_op.result

        load_op.attributes["from"] = StringAttr.get(op.tensor.name)
        if isinstance(op.dtype, htypes.UInt):
            load_op.attributes["unsigned"] = UnitAttr.get()

    def build_store_op(self, op : itmd.StoreOp, ip):
        index_exprs = []
        flag = True
        store_op = None
        for index in op.index:
            try:
                affine_expr = self.build_affine_expr(index)
                index_exprs.append(affine_expr)
            except:
                flag = False
                break
        if flag:
            affine_map = AffineMap.get(
                dim_count=len(op.index), symbol_count=0, exprs=index_exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            store_op = affine_d.AffineStoreOp(
                op.value.result,
                op.tensor.result,
                [idx.result for idx in op.index],
                affine_attr,
                ip=ip
            )
        else:
            new_indices = []
            for index in op.index:
                new_indices.append(index.result)
            store_op = memref_d.StoreOp(op.value.result, op.tensor.result, new_indices, ip=ip)
        # we don't need to set the result of store op
        # because store op doesn't have a result
        store_op.attributes["to"] = StringAttr.get(op.tensor.name)
        if isinstance(op.tensor.dtype, htypes.UInt):
            store_op.attributes["unsigned"] = UnitAttr.get()        

    def build_constant_op(self, op, ip):
        pass


    def build_affine_expr(self, expr):
        """Build affine expression.
        * Should all be binary op
        * AffineExpr can be automatically simplied
        """
        if not isinstance(expr, (hcl_mlir.IterVar, itmd.IterVar, itmd.ConstantOp, itmd.CastOp, itmd.BinaryOp)):
            raise HCLValueError("Not an affine index!")
        if isinstance(expr, hcl_mlir.IterVar):
            if isinstance(expr.op.owner.owner, scf_d.ForOp):
                raise HCLValueError("Outer loop is not affine!")
            if expr.op not in self.iv:
                self.iv.append(expr.op)  # BlockArgument
                return AffineExpr.get_dim(len(self.iv) - 1)
            else:
                return AffineExpr.get_dim(self.iv.index(expr.op))
        elif isinstance(expr, itmd.ConstantOp):
            return AffineExpr.get_constant(expr.val)
        elif isinstance(expr, itmd.CastOp):
            return self.build_affine_expr(expr.val)
        lhs = self.build_affine_expr(expr.lhs)
        rhs = self.build_affine_expr(expr.rhs)
        if isinstance(expr, itmd.AddOp):
            return lhs + rhs
        elif isinstance(expr, itmd.SubOp):
            return lhs - rhs
        elif isinstance(expr, itmd.MulOp):
            return lhs * rhs
        elif isinstance(expr, itmd.DivOp):
            return AffineExpr.get_floor_div(lhs, rhs)  # or get_ceil_div
        elif isinstance(expr, itmd.RemOp):
            return lhs % rhs
        else:
            raise HCLValueError("Not an affine index!")