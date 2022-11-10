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
    affine as affine_d, arith as arith_d
from hcl_mlir.exceptions import *


def get_op_class(op, typ):
    """Get the class of the given op"""
    if isinstance(op, itmd.Add):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.AddIOp
        elif isinstance(typ, htypes.Float):
            return arith_d.AddFOp
        elif isinstance(typ, (htypes.Fixed, htypes.UFixed)):
            return hcl_d.AddFixedOp
        else:
            raise APIError("Unsupported type for AddOp: {}".format(typ))
    elif isinstance(op, itmd.SubOp):
        pass
    else:
        raise APIError("Unsupported op in get_op_class: {}".format(op))

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
        elif isinstance(op, itmd.CastOp):
            self.build_cast_op(op, ip)
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
        lhs = itmd.CastOp(op.lhs, t, loc)
        rhs = itmd.CastOp(op.rhs, t, loc)
        self.build_visitor(lhs, ip)
        self.build_visitor(rhs, ip)

        # Step 3: build binary op
        OpClass = get_op_class(op, t)
        binary_op = OpClass(lhs.result, rhs.result, ip=ip, loc=loc)
        op.result = binary_op.result

        # Step 4: attach necessary attributes
        if isinstance(t, (htypes.UInt, htypes.UFixed)):
            binary_op.attributes["unsigned"] = UnitAttr.get()

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


    def build_cast_op(self, op, ip):
        res_type = op.dtype
        src_type = self.tinf_engine.infer(op.expr)
        # determine cast op
        CastOpClass = None
        if res_type == src_type:
            op.result = op.expr.result
            return
        elif isinstance(src_type, (htypes.Int, htypes.UInt)) and isinstance(res_type, htypes.Index):
            CastOpClass = arith_d.IndexCastOp
        elif isinstance(src_type, htypes.Index) and isinstance(res_type, (htypes.Int, htypes.UInt)):
            CastOpClass = arith_d.IndexCastOp
        elif isinstance(src_type, htypes.Int) and isinstance(res_type, htypes.Float):
            CastOpClass = arith_d.SIToFPOp
        elif isinstance(src_type, htypes.UInt) and isinstance(res_type, htypes.Float):
            CastOpClass = arith_d.UIToFPOp
        elif isinstance(src_type, htypes.Float) and isinstance(res_type, htypes.Int):
            CastOpClass = arith_d.FPToSIOp
        elif isinstance(src_type, htypes.Float) and isinstance(res_type, htypes.UInt):
            CastOpClass = arith_d.FPToUIOp
        elif isinstance(src_type, (htypes.Int, htypes.UInt)) and isinstance(res_type, (htypes.Int, htypes.UInt)):
            if src_type.width > res_type.width:
                CastOpClass = arith_d.TruncIOp
            elif src_type.width == res_type.width:
                op.result = op.expr.result
                return
            else: # src_type.width < res_type.width
                if (
                    isinstance(op.expr, (itmd.GetBitOp, itmd.GetSliceOp, itmd.LeftShiftOp))
                    or src_type.bits == 1
                ):
                    CastOpClass = arith_d.ExtUIOp
                elif isinstance(src_type, htypes.UInt):
                    CastOpClass = arith_d.ExtUIOp
                else:
                    CastOpClass = arith_d.ExtSIOp
        elif isinstance(src_type, htypes.Float) and isinstance(res_type, htypes.Float):
            if res_type.bits < src_type.bits:
                CastOpClass = arith_d.TruncFOp
            elif res_type.bits > src_type.bits:
                CastOpClass = arith_d.ExtFOp
            else:
                op.result = op.expr.result
                return
        elif isinstance(src_type, htypes.Float) and isinstance(res_type, (htypes.Fixed, htypes.UFixed)):
            CastOpClass = hcl_d.FloatToFixedOp
        elif isinstance(src_type, (htypes.Fixed, htypes.UFixed)) and isinstance(res_type, htypes.Float):
            CastOpClass = hcl_d.FixedToFloatOp
        elif isinstance(src_type, (htypes.Fixed, htypes.UFixed)) and isinstance(res_type, (htypes.Int, htypes.UInt)):
            CastOpClass = hcl_d.FixedToIntOp
        elif isinstance(src_type, (htypes.Int, htypes.UInt)) and isinstance(res_type, (htypes.Fixed, htypes.UFixed)):
            CastOpClass = hcl_d.IntToFixedOp
        elif isinstance(src_type, (htypes.Fixed, htypes.UFixed)) and isinstance(res_type, (htypes.Fixed, htypes.UFixed)):
            if src_type == res_type:
                op.result = op.expr.result
                return
            else:
                CastOpClass = hcl_d.FixedToFixedOp
        elif isinstance(src_type, htypes.Struct) and isinstance(res_type, htypes.Struct):
            # We don't actually cast between struct types,
            # here we check if two structs are identical when all
            # integer fields are signless.
            if len(src_type.dtype_dict) != len(res_type.dtype_dict):
                raise DTypeError(
                    "Casting between structs with different number of fields. " +
                    f"src type: {src_type}, dst type: {res_type}"
                )
            for res_ftype, src_ftype in zip(res_type.dtype_dict.values(), src_type.dtype_dict.values()):
                if isinstance(src_ftype, (htypes.Int, htypes.UInt)) and isinstance(res_ftype, (htypes.Int, htypes.UInt)):
                    if src_ftype.width != res_ftype.width:
                        raise DTypeError(
                            "Casting between structs with different field width. " +
                            f"src type: {src_type}, dst type: {res_type}"
                        )
                else:
                    raise DTypeError(
                        "Casting between structs with different field types. " +
                        f"src type: {src_type}, dst type: {res_type}"
                    )
            op.result = op.expr.result
            return
        elif isinstance(src_type, htypes.Struct) and isinstance(res_type, (htypes.Int, htypes.UInt)):
            all_field_int = True
            total_width = 0
            for ftype in src_type.dtype_dict.values():
                if not isinstance(ftype, (htypes.Int, htypes.UInt)):
                    all_field_int = False
                    break
                total_width += ftype.bits
            if not all_field_int:
                raise DTypeError(
                        "Casting from integer to struct with non-integer fields. " +
                        f"src type: {src_type}, dst type: {res_type}"
                    )
            if total_width != res_type.bits:
                raise DTypeError(
                        "Casting from integer to struct with different width. " +
                        f"src type: {src_type}, dst type: {res_type}"
                    )
            CastOpClass = hcl_d.IntToStructOp
        else:
            raise DTypeError(
                "Casting between unsupported types. " +
                f"src type: {src_type}, dst type: {res_type}"
            )

        # build the cast op
        if isinstance(res_type, (htypes.Int, htypes.UInt, htypes.Struct)):
            mlir_type = hcl_dtype_to_mlir(res_type, signless=True)
            cast_op = CastOpClass(mlir_type, op.expr.result, ip=ip)
            if isinstance(res_type, (htypes.UInt, htypes.Struct)):
                cast_op.attributes["unsigned"] = UnitAttr.get()
        else:
            mlir_type = hcl_dtype_to_mlir(res_type)
            cast_op = CastOpClass(mlir_type, op.expr.result, ip=ip)
        op.result = cast_op.result

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
        if isinstance(expr, itmd.Add):
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