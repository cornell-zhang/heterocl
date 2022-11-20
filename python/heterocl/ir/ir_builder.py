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
from ..utils import hcl_dtype_to_mlir, get_extra_type_hints
from .. import types as htypes
from ..type_infer import TypeInfer
# Import MLIR dialects
# Naming rule: import dialect as dialect_d 
from hcl_mlir.dialects import \
    func as func_d, hcl as hcl_d, \
    scf as scf_d, memref as memref_d, \
    affine as affine_d, arith as arith_d
from hcl_mlir.exceptions import *

""" IRBuilder Assumptions
- All Python immediate should be converted to ConstantOp
"""

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
    elif isinstance(op, itmd.Sub):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.SubIOp
        elif isinstance(typ, htypes.Float):
            return arith_d.SubFOp
        elif isinstance(typ, (htypes.Fixed, htypes.UFixed)):
            return hcl_d.SubFixedOp
        else:
            raise APIError("Unsupported type for SubOp: {}".format(typ))
    elif isinstance(op, itmd.Mul):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.MulIOp
        elif isinstance(typ, htypes.Float):
            return arith_d.MulFOp
        elif isinstance(typ, (htypes.Fixed, htypes.UFixed)):
            return hcl_d.MulFixedOp
        else:
            raise APIError("Unsupported type for MulOp: {}".format(typ))
    elif isinstance(op, itmd.Mod):
        if isinstance(typ, htypes.Int):
            return arith_d.RemSIOp
        elif isinstance(typ, htypes.UInt):
            return arith_d.RemUIOp
        elif isinstance(typ, htypes.Float):
            return arith_d.RemFOp
        else:
            raise APIError("Unsupported type for ModOp: {}".format(typ))
    elif isinstance(op, itmd.And):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.AndIOp
        else:
            raise APIError("Unsupported type for AndOp: {}".format(typ))
    elif isinstance(op, itmd.Or):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.OrIOp
        else:
            raise APIError("Unsupported type for OrOp: {}".format(typ))
    elif isinstance(op, itmd.XOr):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.XOrIOp
        else:
            raise APIError("Unsupported type for XOrOp: {}".format(typ))
    elif isinstance(op, itmd.Mod):
        if isinstance(typ, htypes.Int):
            return arith_d.RemSIOp
        elif isinstance(typ, htypes.UInt):
            return arith_d.RemUIOp
        elif isinstance(typ, htypes.Float):
            return arith_d.RemFOp
        else:
            raise APIError("Unsupported type for ModOp: {}".format(typ))
    else:
        raise APIError("Unsupported op in get_op_class: {}".format(op))

class IRBuilder(object):
    """IRBuilder class to build MLIR
    operations from intermediate layer
    """
    def __init__(self, intermediate):
        self._intermediate = intermediate
        self.module = Module.create(get_location())
        self.iv = [] # a list to keep track of affine expression's induction variables
        self.tinf_engine = TypeInfer()
        self.tensor_dict = dict() # tensor name -> memref.allocOp

    def build(self):
        """Builder entry point
        
        Build MLIR module with a top-level function
        """
        top_func = self._intermediate.top_func
        with get_context(), get_location():
            input_types = []
            input_typehints = []
            for tensor in top_func.args:
                self.tensor_dict[tensor.name] = tensor
                ele_type = hcl_dtype_to_mlir(tensor.dtype, signless=True)
                input_typehints.append(get_extra_type_hints(tensor.dtype))
                memref_type = MemRefType.get(tensor.shape, ele_type)
                input_types.append(memref_type)
            return_types = []
            output_typehints = []
            for tensor in top_func.return_tensors:
                ele_type = hcl_dtype_to_mlir(tensor.dtype, signless=True)
                output_typehints.append(get_extra_type_hints(tensor.dtype))
                memref_type = MemRefType.get(tensor.shape, ele_type)
                return_types.append(memref_type)
            ip = InsertionPoint(self.module.body)
            func_type = FunctionType.get(input_types, return_types)
            func = func_d.FuncOp(name=top_func.name, type=func_type, ip=ip)
            top_func.ir_op = func
            func.add_entry_block()

            # Set alloc op's result as function block arg
            for alloc_op, arg in zip(top_func.args, func.entry_block.arguments):
                alloc_op.result = arg

            # build build's body op
            ip = InsertionPoint(func.entry_block)
            for op in top_func.body:
                self.build_visitor(op, ip)
            return_names = [tensor.name for tensor in top_func.return_tensors]
            returns = [self.tensor_dict[name].result for name in return_names]
            func_d.ReturnOp(returns, ip=ip)
            
            # attach attributes
            # if program has bit operations
            # func.attributes["bit"] = UnitAttr.get()
            func.attributes["function_type"] = TypeAttr.get(func_type)
            # attach type hints
            otypes = "".join(output_typehints)
            itypes = "".join(input_typehints)
            func.attributes["otypes"] = StringAttr.get(otypes)
            func.attributes["itypes"] = StringAttr.get(itypes)


    def build_visitor(self, op, ip):
        """Build dispatcher
        
        Build MLIR operation from intermediate layer
        """
        if op.result is not None:
            return
        if isinstance(op, itmd.ComputeOp):
            self.build_compute(op, ip)
        elif isinstance(op, itmd.IterVar):
            self.build_iter_var(op, ip)
        elif isinstance(op, itmd.ReduceOp):
            self.build_reduce(op, ip)
        elif isinstance(op, itmd.AllocOp):
            self.build_alloc_op(op, ip)
        elif isinstance(op, itmd.Cmp):
            self.build_cmp_op(op, ip)
        elif isinstance(op, itmd.BinaryOp):
            self.build_binary_op(op, ip)
        elif isinstance(op, itmd.BitCastOp):
            self.build_bitcast_op(op, ip)
        elif isinstance(op, itmd.LoadOp):
            self.build_load_op(op, ip)
        elif isinstance(op, itmd.StoreOp):
            self.build_store_op(op, ip)
        elif isinstance(op, itmd.ConstantOp):
            self.build_constant_op(op, ip)
        elif isinstance(op, itmd.CastOp):
            self.build_cast_op(op, ip)
        elif isinstance(op, itmd.IfOp):
            self.build_if_op(op, ip)
        elif isinstance(op, itmd.SelectOp):
            self.build_select_op(op, ip)
        elif isinstance(op, itmd.PrintOp):
            self.build_print_op(op, ip)
        elif isinstance(op, itmd.PrintTensorOp):
            self.build_print_tensor_op(op, ip)
        else:
            raise HCLNotImplementedError(f"{type(op)}'s build visitor is not implemented yet.")

    def build_iter_var(self, iv, ip):
        """Build IterVar"""
        if iv.parent_loop is None:
            raise APIError("IterVar {} parent loop has not been set".format(iv))
        iv.result = iv.parent_loop.induction_variable

    def build_compute(self, op, ip):
        iv_names = [iv.name for iv in op.iter_vars]
        with get_context(), get_location():
            # build output tensor
            alloc_op = op.tensor
            if alloc_op is not None:
                self.build_visitor(alloc_op, ip)
                op.result = alloc_op.result
                op.ir_op = alloc_op
            
            loops = list()
            for i, (ub, loop_name) in enumerate(zip(op.shape, iv_names)):
                # TODO(Niansong): merge make_for with build_for?
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
            for iter_var, loop in zip(op.iter_vars, loops):
                iter_var.parent_loop = loop
            for body_op in op.body:
                self.build_visitor(body_op, ip)

    def build_alloc_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        ele_type = hcl_dtype_to_mlir(op.dtype, signless=True)
        memref_type = MemRefType.get(op.shape, ele_type)
        alloc_op = memref_d.AllocOp(memref_type, [], [], ip=ip, loc=loc)
        alloc_op.attributes["name"] = StringAttr.get(op.name)
        op.result = alloc_op.result
        op.ir_op = alloc_op
        # assume no name conflict
        if op.name in self.tensor_dict:
            raise APIError("Tensor name conflict: {}".format(op.name))
        self.tensor_dict[op.name] = alloc_op

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
        op.ir_op = binary_op

        # Step 4: attach necessary attributes
        if isinstance(t, (htypes.UInt, htypes.UFixed)):
            binary_op.attributes["unsigned"] = UnitAttr.get()

    def build_cmp_op(self, op : itmd.Cmp, ip):
        """
        # Check mlir/Dialect/Arithmetic/IR/ArithmeticBase.td
        # s/u: signed/unsigned
        # o/u: ordered/unordered
        #      ordered means only one of < = > cases is true
        #      unordered happens for floating points with NaN
        // Opcode              U L G E    Intuitive operation
        FCMP_FALSE =  0,  ///< 0 0 0 0    Always false (always folded)
        FCMP_OEQ   =  1,  ///< 0 0 0 1    True if ordered and equal
        FCMP_OGT   =  2,  ///< 0 0 1 0    True if ordered and greater than
        FCMP_OGE   =  3,  ///< 0 0 1 1    True if ordered and greater than or equal
        FCMP_OLT   =  4,  ///< 0 1 0 0    True if ordered and less than
        FCMP_OLE   =  5,  ///< 0 1 0 1    True if ordered and less than or equal
        FCMP_ONE   =  6,  ///< 0 1 1 0    True if ordered and operands are unequal
        FCMP_ORD   =  7,  ///< 0 1 1 1    True if ordered (no nans)
        FCMP_UNO   =  8,  ///< 1 0 0 0    True if unordered: isnan(X) | isnan(Y)
        FCMP_UEQ   =  9,  ///< 1 0 0 1    True if unordered or equal
        FCMP_UGT   = 10,  ///< 1 0 1 0    True if unordered or greater than
        FCMP_UGE   = 11,  ///< 1 0 1 1    True if unordered, greater than, or equal
        FCMP_ULT   = 12,  ///< 1 1 0 0    True if unordered or less than
        FCMP_ULE   = 13,  ///< 1 1 0 1    True if unordered, less than, or equal
        FCMP_UNE   = 14,  ///< 1 1 1 0    True if unordered or not equal
        FCMP_TRUE  = 15,  ///< 1 1 1 1    Always true (always folded)
        """
        ATTR_MAP = {
            "int": {
                "eq": 0,
                "ne": 1,
                "slt": 2,
                "sle": 3,
                "sgt": 4,
                "sge": 5,
                "ult": 6,
                "ule": 7,
                "ugt": 8,
                "uge": 9,
            },
            "float": {
                "false": 0,
                "oeq": 1,
                "ogt": 2,
                "oge": 3,
                "olt": 4,
                "ole": 5,
                "one": 6,
                "ord": 7,
                "ueq": 8,
                "ugt": 9,
                "uge": 10,
                "ult": 11,
                "ule": 12,
                "une": 13,
                "uno": 14,
                "true": 15,
            },
            "fixed": {
                "eq": 0,
                "ne": 1,
                "slt": 2,
                "sle": 3,
                "sgt": 4,
                "sge": 5,
                "ult": 6,
                "ule": 7,
                "ugt": 8,
                "uge": 9,
            },
        }
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
        if isinstance(t, (htypes.Int, htypes.Index)):
            OpClass = arith_d.CmpIOp
            attr = ATTR_MAP["int"][
                "s" + op.name if op.name not in ["eq", "ne"] else op.name
            ]
        elif isinstance(t, htypes.UInt):
            OpClass = arith_d.CmpIOp
            attr = ATTR_MAP["int"][
                "u" + op.name if op.name not in ["eq", "ne"] else op.name
            ]
        elif isinstance(t, htypes.Float):
            OpClass = arith_d.CmpFOp
            attr = ATTR_MAP["float"]['o' + op.name]
        elif isinstance(t, htypes.Fixed):
            OpClass = hcl_d.CmpFixedOp
            attr = ATTR_MAP["fixed"][
                's' + op.name if op.name not in ["eq", "ne"] else op.name
            ]
        elif isinstance(t, htypes.UFixed):
            OpClass = hcl_d.CmpFixedOp
            attr = ATTR_MAP["fixed"][
                'u' + op.name if op.name not in ["eq", "ne"] else op.name
            ]
        else:
            raise NotImplementedError(f"Unsupported type for CmpOp build: {t}")

        dtype = IntegerType.get_signless(1)
        cmp_attr = IntegerAttr.get(IntegerType.get_signless(64), attr)
        cmp_op = OpClass(dtype, cmp_attr, lhs.result, rhs.result, ip=ip, loc=loc)
        op.result = cmp_op.result
        op.ir_op = cmp_op

        # Step 4: attach necessary attributes
        if isinstance(t, (htypes.UInt, htypes.UFixed)):
            cmp_op.attributes["unsigned"] = UnitAttr.get()

    def build_load_op(self, op : itmd.LoadOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        index_exprs = []
        flag = True
        load_op = None
        self.iv.clear() # clear iv
        for index in op.index:
            try:
                affine_expr = self.build_affine_expr(index)
                index_exprs.append(affine_expr)
            except:
                flag = False
                break
        if flag:
            dim_count = len(self.iv)
            affine_map = AffineMap.get(
                dim_count=dim_count, symbol_count=0, exprs=index_exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            load_op = affine_d.AffineLoadOp(
                op.tensor.result,
                self.iv,
                affine_attr,
                ip=ip,
                loc=loc
            )
            op.result = load_op.result
            op.ir_op = load_op
        else:
            new_indices = []
            for index in op.index:
                self.build_visitor(index, ip)
                # cast to index type
                index = itmd.CastOp(index, htypes.Index(), loc)
                self.build_visitor(index, ip)
                new_indices.append(index.result)
            load_op = memref_d.LoadOp(op.tensor.result, new_indices, ip=ip, loc=loc)
            op.result = load_op.result
            op.ir_op = load_op

        load_op.attributes["from"] = StringAttr.get(op.tensor.name)
        if isinstance(op.dtype, htypes.UInt):
            load_op.attributes["unsigned"] = UnitAttr.get()

    def build_store_op(self, op : itmd.StoreOp, ip):
        index_exprs = []
        flag = True
        store_op = None
        if op.value.result is None:
            self.build_visitor(op.value, ip)
        casted_expr = itmd.CastOp(op.value, op.tensor.dtype, op.loc)
        self.build_visitor(casted_expr, ip)
        self.iv.clear() # clear iv
        for index in op.index:
            try:
                affine_expr = self.build_affine_expr(index)
                index_exprs.append(affine_expr)
            except:
                flag = False
                break
        if flag:
            dim_count = len(self.iv)
            affine_map = AffineMap.get(
                dim_count=dim_count, symbol_count=0, exprs=index_exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            store_op = affine_d.AffineStoreOp(
                casted_expr.result,
                op.tensor.result,
                self.iv,
                affine_attr,
                ip=ip
            )
        else:
            new_indices = []
            for index in op.index:
                self.build_visitor(index, ip)
                index = itmd.CastOp(index, htypes.Index(), op.loc)
                self.build_visitor(index, ip)
                new_indices.append(index.result)
            store_op = memref_d.StoreOp(casted_expr.result, op.tensor.result, new_indices, ip=ip)
        # we don't need to set the result of store op
        # because store op doesn't have a result
        store_op.attributes["to"] = StringAttr.get(op.tensor.name)
        if isinstance(op.tensor.dtype, htypes.UInt):
            store_op.attributes["unsigned"] = UnitAttr.get()        

    def build_constant_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        dtype = hcl_dtype_to_mlir(op.dtype)
        if isinstance(op.dtype, (htypes.Int, htypes.UInt)):
            if isinstance(op.dtype, htypes.Index):
                value_attr = IntegerAttr.get(IndexType.get(), op.value)
            elif op.dtype.bits == 1:
                value_attr = BoolAttr.get(op.value)
            else:
                attr_type = IntegerType.get_signless(op.dtype.bits)
                value_attr = IntegerAttr.get(attr_type, op.value)
            const_op = arith_d.ConstantOp(dtype, value_attr, ip=ip, loc=loc)
        elif isinstance(op.dtype, htypes.Float):
            if op.dtype.bits == 16:
                value_attr = FloatAttr.get(F16Type.get(), op.value)
            elif op.dtype.bits == 32:
                value_attr = FloatAttr.get(F32Type.get(), op.value)
            elif op.dtype.bits == 64:
                value_attr = FloatAttr.get(F64Type.get(), op.value)
            else:
                raise DTypeError("Unsupported float type: {}".format(op.dtype))
            const_op = arith_d.ConstantOp(dtype, value_attr, ip=ip, loc=loc)
        elif isinstance(op.dtype, (htypes.Fixed, htypes.UFixed)):
            # assume the value is converted to integer base
            if not isinstance(op.value, int):
                raise DTypeError("Fixed point value must be converted to integer base")
            attr_type = IntegerType.get_signless(op.dtype.bits)
            value_attr = IntegerAttr.get(attr_type, op.value)
            const_op = arith_d.ConstantOp(dtype, value_attr, ip=ip, loc=loc)
        else:
            raise DTypeError("Unsupported type: {}".format(op.dtype))
        
        op.result = const_op.result
        op.ir_op = const_op

        # attach necessary attributes
        if isinstance(op.dtype, (htypes.UInt, htypes.UFixed)):
            const_op.attributes["unsigned"] = UnitAttr.get()


    def build_cast_op(self, op, ip):
        if op.expr.result is None:
            self.build_visitor(op.expr, ip)
        res_type = op.dtype
        src_type = self.tinf_engine.infer(op.expr)
        # determine cast op
        CastOpClass = None
        if type(res_type) == type(src_type) and res_type == src_type:
            op.result = op.expr.result
            op.ir_op = op.expr.ir_op
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
            if src_type.bits > res_type.bits:
                CastOpClass = arith_d.TruncIOp
            elif src_type.bits == res_type.bits:
                op.result = op.expr.result
                op.ir_op = op.expr.ir_op
                return
            else: # src_type.bits < res_type.bits
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
                op.ir_op = op.expr.ir_op
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
                op.ir_op = op.expr.ir_op
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
            op.ir_op = op.expr.ir_op
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
        op.ir_op = cast_op

    def build_affine_expr(self, expr):
        """Build affine expression.
        * Should all be binary op
        * AffineExpr can be automatically simplied
        """
        if not isinstance(expr, (itmd.IterVar, itmd.ConstantOp, itmd.CastOp, itmd.BinaryOp)):
            raise HCLValueError(f"{expr} is not an affine index")
        if isinstance(expr, itmd.IterVar):
            if isinstance(expr.parent_loop, scf_d.ForOp):
                raise HCLValueError(f"loop {expr.parent_loop} is not affine")
            if expr.parent_loop.induction_variable not in self.iv:
                self.iv.append(expr.parent_loop.induction_variable)  # BlockArgument
                return AffineExpr.get_dim(len(self.iv) - 1)
            else:
                return AffineExpr.get_dim(self.iv.index(expr.parent_loop.induction_variable))
        elif isinstance(expr, itmd.ConstantOp):
            return AffineExpr.get_constant(expr.value)
        elif isinstance(expr, itmd.CastOp):
            return self.build_affine_expr(expr.expr)
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
        elif isinstance(expr, itmd.ModOp):
            return lhs % rhs
        else:
            raise HCLValueError(f"{expr} is not an affine index!")

    def build_if_op(self, op : itmd.IfOp, ip):
        """Build IfOp"""
        # TODO: support affine if
        # build condition
        self.build_visitor(op.cond, ip)
        has_else = op.else_branch_valid
        if_op = scf_d.IfOp(op.cond.result, hasElse=has_else, results_=[], ip=ip)
        # build then body
        ip = InsertionPoint(if_op.then_block)
        for body_op in op.body:
            self.build_visitor(body_op, ip)
        scf_d.YieldOp([], ip=ip)
        # build else body
        if has_else:
            ip = InsertionPoint(if_op.else_block)
            for body_op in op.else_body:
                self.build_visitor(body_op, ip)
            scf_d.YieldOp([], ip=ip)


    def build_reduce(self, op: itmd.ReduceOp, ip):
        """Build ReduceOp"""
        # scalar to hold the reduction result
        self.build_visitor(op.scalar, ip)
        # store the init value to the scalar
        init = itmd.immediate_to_constant(op.init, op.loc)
        self.build_visitor(init, ip)
        zero_idx = itmd.ConstantOp(0, htypes.Index(), op.loc)
        self.build_visitor(zero_idx, ip)
        store_op = itmd.StoreOp(op.scalar, (zero_idx,), init, op.loc)
        self.build_visitor(store_op, ip)
        body_ip = ip

        # build loop nest
        loops = list()
        for axis in op.axis:
            lb, ub = axis.bound
            loop = hcl_mlir.make_for(
                lb, ub, step=1,
                reduction=True,
                name=axis.name,
                ip=body_ip
            )
            axis.parent_loop = loop
            loops.append(loop)
            body_ip = InsertionPoint(loop.body.operations[0])

        # load from the input tensor
        self.build_visitor(op.expr, body_ip)
        # load from scalar
        load_scalar = itmd.LoadOp(op.scalar, (zero_idx,), op.loc)
        # build the reduction op
        if op.reduce_op == "sum":
            reduce_op = itmd.Add(op.expr, load_scalar, op.loc)
        elif op.reduce_op == "max":
            pass
        elif op.reduce_op == "min":
            pass
        else:
            raise HCLValueError(f"Unsupported reduction op {op.reduce_op}")

        self.build_visitor(reduce_op, body_ip)
        # store to the scalar
        store_res = itmd.StoreOp(op.scalar, (zero_idx,), reduce_op, op.loc)
        self.build_visitor(store_res, body_ip)

        # load from the scalar
        load_op = itmd.LoadOp(op.scalar, (zero_idx,), op.loc)
        self.build_visitor(load_op, ip)
        op.ir_op = load_op
        op.result = load_op.result

    def build_select_op(self, op : itmd.SelectOp, ip):
        # Step 1: get condition, true, and false value
        # if any of them is an immediate, convert it to a constant
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        cond = itmd.immediate_to_constant(op.cond, op.loc)
        self.build_visitor(cond, ip)
        true_value = itmd.immediate_to_constant(op.true_value, op.loc)
        self.build_visitor(true_value, ip)
        false_value = itmd.immediate_to_constant(op.false_value, op.loc)
        self.build_visitor(false_value, ip)
        
        # Step 2: type inference and cast
        res_type = self.tinf_engine.infer(op)
        # cast condition to uint1
        if not isinstance(cond, itmd.Cmp):
            cond = itmd.CastOp(cond, htypes.UInt(1), op.loc)
            self.build_visitor(cond, ip)
        # cast true and false value to the same type
        true_value = itmd.CastOp(true_value, res_type, op.loc)
        self.build_visitor(true_value, ip)
        false_value = itmd.CastOp(false_value, res_type, op.loc)
        self.build_visitor(false_value, ip)
        
        # Step 3: build select op
        select_op = arith_d.SelectOp(cond.result, true_value.result, false_value.result, ip=ip, loc=loc)
        op.ir_op = select_op
        op.result = select_op.result

    def build_bitcast_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.expr, ip)
        src_dtype = self.tinf_engine.infer(op.expr)
        dst_dtype = op.dtype
        if src_dtype.bits != dst_dtype.bits:
            raise APIError(
                "Destination datatype bitwidth does not match source bitwidth:"
                + f"source bitwidth: {src_dtype.bits} , destination bitwidth {op.dtype}."
            )
        dst_dtype = hcl_dtype_to_mlir(dst_dtype, signless=True) 
        bitcast_op = arith_d.BitcastOp(dst_dtype, op.expr.result, ip=ip, loc=loc)
        op.ir_op = bitcast_op
        op.result = bitcast_op.result

    def build_print_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        # print op assumes all inputs are built
        for arg in op.args:
            self.build_visitor(arg, ip)
        default_fmt = ""
        signedness_str = ""
        for arg in op.args:
            dtype = self.tinf_engine.infer(arg)
            if isinstance(dtype, (htypes.UInt, htypes.Int)):
                default_fmt += "%d "
                if isinstance(dtype, htypes.UInt):
                    signedness_str += "u"
                else:
                    signedness_str += "_"
            else:
                default_fmt += "%.3f "
                signedness_str += "_"
            default_fmt += "\n"
    
        if op.fmt == "":
            op.fmt = default_fmt

        # build print op
        operands = [v.result for v in op.args]
        print_op = hcl_d.PrintOp(operands, ip=ip, loc=loc)
        fmt_str = StringAttr.get(op.fmt)
        signedness_str = StringAttr.get(signedness_str)
        print_op.attributes["signedness"] = signedness_str
        print_op.attributes["format"] = fmt_str
        op.ir_op = print_op

    def build_print_tensor_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.tensor, ip)
        print_op = hcl_d.PrintMemRefOp(op.tensor.result, ip=ip, loc=loc)
        op.ir_op = print_op