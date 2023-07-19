# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, too-many-return-statements, too-many-branches, unused-argument, too-many-public-methods, no-else-return, else-if-used, too-many-function-args
""" IRBuilder Assumptions
    - All Python immediate should be converted to ConstantOp
"""

# Import MLIR dialects
# Naming rule: import dialect as dialect_d
import numpy as np

from hcl_mlir.dialects import (
    func as func_d,
    hcl as hcl_d,
    scf as scf_d,
    memref as memref_d,
    affine as affine_d,
    arith as arith_d,
    math as math_d,
)
from hcl_mlir.ir import (
    Module,
    InsertionPoint,
    Location,
    MemRefType,
    FunctionType,
    TypeAttr,
    StringAttr,
    UnitAttr,
    FlatSymbolRefAttr,
    AffineConstantExpr,
    AffineMap,
    AffineMapAttr,
    IntegerType,
    IntegerAttr,
    BoolAttr,
    IndexType,
    FloatAttr,
    F16Type,
    F32Type,
    F64Type,
    AffineExpr,
    DenseElementsAttr,
)
from hcl_mlir.exceptions import (
    DTypeError,
    APIError,
    HCLNotImplementedError,
    MLIRLimitationError,
    HCLValueError,
)

from . import ast
from ..context import get_context, get_location
from ..utils import hcl_dtype_to_mlir, get_extra_type_hints
from .. import types as htypes
from . import build_cleaner


def get_op_class(op, typ):
    """Get the class of the given op
    TODO: Consider using dictionary to store the mapping
    """
    if isinstance(op, ast.Add):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.AddIOp
        if isinstance(typ, htypes.Float):
            return arith_d.AddFOp
        if isinstance(typ, (htypes.Fixed, htypes.UFixed)):
            return hcl_d.AddFixedOp
        raise APIError(f"Unsupported type for AddOp: {typ}")
    if isinstance(op, ast.Sub):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.SubIOp
        if isinstance(typ, htypes.Float):
            return arith_d.SubFOp
        if isinstance(typ, (htypes.Fixed, htypes.UFixed)):
            return hcl_d.SubFixedOp
        raise APIError(f"Unsupported type for SubOp: {typ}")
    if isinstance(op, ast.Mul):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.MulIOp
        if isinstance(typ, htypes.Float):
            return arith_d.MulFOp
        if isinstance(typ, (htypes.Fixed, htypes.UFixed)):
            return hcl_d.MulFixedOp
        raise APIError(f"Unsupported type for MulOp: {typ}")
    if isinstance(op, ast.Div):
        if isinstance(typ, htypes.Int):
            return arith_d.DivSIOp
        if isinstance(typ, htypes.UInt):
            return arith_d.DivUIOp
        if isinstance(typ, htypes.Float):
            return arith_d.DivFOp
        if isinstance(typ, (htypes.Fixed, htypes.UFixed)):
            return hcl_d.DivFixedOp
        raise APIError(f"Unsupported type for DivOp: {typ}")
    if isinstance(op, ast.Max):
        if isinstance(typ, htypes.Int):
            return arith_d.MaxSIOp
        if isinstance(typ, htypes.UInt):
            return arith_d.MaxUIOp
        if isinstance(typ, htypes.Float):
            return arith_d.MaxFOp
        if isinstance(typ, (htypes.Fixed, htypes.UFixed)):
            return hcl_d.MaxFixedOp
        raise APIError(f"Unsupported type for MaxOp: {typ}")
    if isinstance(op, ast.Min):
        if isinstance(typ, htypes.Int):
            return arith_d.MinSIOp
        if isinstance(typ, htypes.UInt):
            return arith_d.MinUIOp
        if isinstance(typ, htypes.Float):
            return arith_d.MinFOp
        if isinstance(typ, (htypes.Fixed, htypes.UFixed)):
            return hcl_d.MinFixedOp
        raise APIError(f"Unsupported type for MinOp: {typ}")
    if isinstance(op, ast.FloorDiv):
        if isinstance(typ, htypes.Int):
            return arith_d.FloorDivSIOp
        raise APIError(f"Unsupported type for FloorDivOp: {typ}")
    if isinstance(op, ast.Mod):
        if isinstance(typ, htypes.Int):
            return arith_d.RemSIOp
        if isinstance(typ, htypes.UInt):
            return arith_d.RemUIOp
        if isinstance(typ, htypes.Float):
            return arith_d.RemFOp
        raise APIError(f"Unsupported type for ModOp: {typ}")
    if isinstance(op, ast.And):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.AndIOp
        raise APIError(f"Unsupported type for AndOp: {typ}")
    if isinstance(op, ast.Or):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.OrIOp
        raise APIError(f"Unsupported type for OrOp: {typ}")
    if isinstance(op, ast.XOr):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.XOrIOp
        raise APIError(f"Unsupported type for XOrOp: {typ}")
    if isinstance(op, ast.Mod):
        if isinstance(typ, htypes.Int):
            return arith_d.RemSIOp
        if isinstance(typ, htypes.UInt):
            return arith_d.RemUIOp
        if isinstance(typ, htypes.Float):
            return arith_d.RemFOp
        raise APIError(f"Unsupported type for ModOp: {typ}")
    if isinstance(op, ast.LogicalAnd):
        if isinstance(typ, (htypes.Int, htypes.UInt)) and typ.bits == 1:
            return arith_d.AndIOp
        raise APIError(f"Unsupported type for LogicalAndOp: {typ}")
    if isinstance(op, ast.LogicalOr):
        if isinstance(typ, (htypes.Int, htypes.UInt)) and typ.bits == 1:
            return arith_d.OrIOp
        raise APIError(f"Unsupported type for LogicalOrOp: {typ}")
    if isinstance(op, ast.LogicalXOr):
        if isinstance(typ, (htypes.Int, htypes.UInt)) and typ.bits == 1:
            return arith_d.XOrIOp
        raise APIError(f"Unsupported type for LogicalXOrOp: {typ}")
    if isinstance(op, ast.MathPowOp):
        if isinstance(typ, htypes.Float):
            return math_d.PowFOp
        raise APIError(f"Unsupported type for MathPowOp: {typ}")
    if isinstance(op, ast.LeftShiftOp):
        if isinstance(typ, (htypes.Int, htypes.UInt)):
            return arith_d.ShLIOp
        raise APIError(f"Unsupported type for LeftShiftOp: {typ}")
    if isinstance(op, ast.RightShiftOp):
        if isinstance(typ, htypes.Int):
            return arith_d.ShRSIOp
        if isinstance(typ, htypes.UInt):
            return arith_d.ShRUIOp
        raise APIError(f"Unsupported type for RightShiftOp: {typ}")
    if isinstance(op, ast.MathExpOp):
        if isinstance(typ, htypes.Float):
            return math_d.ExpOp
        raise APIError(f"Unsupported type for MathExpOp: {typ}")
    if isinstance(op, ast.MathPowOp):
        if isinstance(typ, htypes.Float):
            return math_d.PowFOp
        raise APIError(f"Unsupported type for MathPowOp: {typ}")
    if isinstance(op, ast.MathLogOp):
        if isinstance(typ, htypes.Float):
            return math_d.LogOp
        raise APIError(f"Unsupported type for MathLogOp: {typ}")
    if isinstance(op, ast.MathLog2Op):
        if isinstance(typ, htypes.Float):
            return math_d.Log2Op
        raise APIError(f"Unsupported type for MathLog2Op: {typ}")
    if isinstance(op, ast.MathLog10Op):
        if isinstance(typ, htypes.Float):
            return math_d.Log10Op
        raise APIError(f"Unsupported type for MathLog10Op: {typ}")
    if isinstance(op, ast.MathSqrtOp):
        if isinstance(typ, htypes.Float):
            return math_d.SqrtOp
        raise APIError(f"Unsupported type for MathSqrtOp: {typ}")
    if isinstance(op, ast.MathSinOp):
        if isinstance(typ, htypes.Float):
            return math_d.SinOp
        raise APIError(f"Unsupported type for MathSinOp: {typ}")
    if isinstance(op, ast.MathCosOp):
        if isinstance(typ, htypes.Float):
            return math_d.CosOp
        raise APIError(f"Unsupported type for MathCosOp: {typ}")
    if isinstance(op, ast.MathTanOp):
        if isinstance(typ, htypes.Float):
            return math_d.TanOp
        raise APIError(f"Unsupported type for MathTanOp: {typ}")
    if isinstance(op, ast.MathTanhOp):
        if isinstance(typ, htypes.Float):
            return math_d.TanhOp
        raise APIError(f"Unsupported type for MathTanhOp: {typ}")
    raise APIError(f"Unsupported op in get_op_class: {op}")


def is_all_field_int(dtype):
    """Check if a struct type has all integer fields
    When it has nested struct field, recursively check
    the nested struct field.
    """
    if not isinstance(dtype, htypes.Struct):
        return False
    for field_type in dtype.dtype_dict.values():
        if isinstance(field_type, htypes.Struct):
            if not is_all_field_int(field_type):
                return False
        elif not isinstance(field_type, (htypes.Int, htypes.UInt)):
            return False
    return True


def get_struct_bitwidth(dtype):
    bitwidth = 0
    for field in dtype.dtype_dict.values():
        if isinstance(field, htypes.Struct):
            bitwidth += get_struct_bitwidth(field)
        else:
            bitwidth += field.bits
    return bitwidth


class IRBuilder:
    """IRBuilder class to build MLIR
    operations from intermediate layer
    """

    def __init__(self, _ast):
        self._ast = _ast
        self.module = Module.create(get_location())
        self.top_func = None
        self.iv = []  # a list to keep track of affine expression's induction variables
        self.tinf_engine = ast.TypeInference()
        self.cleaner = build_cleaner.ASTCleaner()
        self.tensor_dict = {}  # tensor name -> memref.allocOp
        self.BIT_OPS = False

    def build(self):
        if self._ast is None:
            # if ast is None, we just return an empty module
            return

        # build each operation in the ast
        with get_context(), get_location():
            for op in self._ast.region:
                ip = InsertionPoint.at_block_begin(self.module.body)
                self.build_visitor(op, ip)

        self.top_func = self._ast.top_func.ir_op

    def build_visitor(self, op, ip):
        """Build dispatcher

        Build MLIR operation from intermediate layer
        """
        if hasattr(op, "result") and op.result is not None:
            if op.reusable:
                # if operation as result and is reusable
                # return without building new operation
                return
        if isinstance(op, ast.ComputeOp):
            self.build_compute(op, ip)
        elif isinstance(op, ast.IterVar):
            self.build_iter_var(op, ip)
        elif isinstance(op, ast.ReduceOp):
            self.build_reduce(op, ip)
        elif isinstance(op, ast.AllocOp):
            self.build_alloc_op(op, ip)
        elif isinstance(op, ast.Cmp):
            self.build_cmp_op(op, ip)
        elif isinstance(op, ast.BinaryOp):
            self.build_binary_op(op, ip)
        elif isinstance(
            op,
            (
                ast.MathExpOp,
                ast.MathPowOp,
                ast.MathLogOp,
                ast.MathLog2Op,
                ast.MathLog10Op,
                ast.MathSqrtOp,
                ast.MathSinOp,
                ast.MathCosOp,
                ast.MathTanOp,
                ast.MathTanhOp,
                # ast.PowOp is covered by build_binary_op
            ),
        ):
            self.build_math_op(op, ip)
        elif isinstance(op, ast.BitCastOp):
            self.build_bitcast_op(op, ip)
        elif isinstance(op, ast.LoadOp):
            self.build_load_op(op, ip)
        elif isinstance(op, ast.StoreOp):
            self.build_store_op(op, ip)
        elif isinstance(op, ast.ConstantOp):
            self.build_constant_op(op, ip)
        elif isinstance(op, ast.CastOp):
            self.build_cast_op(op, ip)
        elif isinstance(op, ast.IfOp):
            self.build_if_op(op, ip)
        elif isinstance(op, ast.ForOp):
            self.build_for_op(op, ip)
        elif isinstance(op, ast.WhileOp):
            self.build_while_op(op, ip)
        elif isinstance(op, ast.SelectOp):
            self.build_select_op(op, ip)
        elif isinstance(op, ast.PrintOp):
            self.build_print_op(op, ip)
        elif isinstance(op, ast.PrintTensorOp):
            self.build_print_tensor_op(op, ip)
        elif isinstance(op, ast.GetBitOp):
            self.BIT_OPS = True
            self.build_get_bit_op(op, ip)
        elif isinstance(op, ast.GetSliceOp):
            self.BIT_OPS = True
            self.build_get_slice_op(op, ip)
        elif isinstance(op, ast.SetBitOp):
            self.BIT_OPS = True
            self.build_set_bit_op(op, ip)
        elif isinstance(op, ast.SetSliceOp):
            self.BIT_OPS = True
            self.build_set_slice_op(op, ip)
        elif isinstance(op, ast.BitReverseOp):
            self.BIT_OPS = True
            self.build_bit_reverse_op(op, ip)
        elif isinstance(op, ast.ConstantTensorOp):
            self.build_constant_tensor_op(op, ip)
        elif isinstance(op, ast.StructConstructOp):
            self.build_struct_construct_op(op, ip)
        elif isinstance(op, ast.StructGetOp):
            self.build_struct_get_op(op, ip)
        elif isinstance(op, ast.FuncOp):
            self.build_func_op(op, ip)
        elif isinstance(op, ast.CallOp):
            self.build_call_op(op, ip)
        elif isinstance(op, ast.Neg):
            self.build_neg_op(op, ip)
        elif isinstance(op, ast.OpHandle):
            self.build_op_handle(op, ip)
        elif isinstance(op, ast.LoopHandle):
            self.build_loop_handle(op, ip)
        elif isinstance(op, ast.InterKernelToOp):
            self.build_inter_kernel_to_op(op, ip)
        elif isinstance(op, ast.OutlineOp):
            self.build_outline_op(op, ip)
        else:
            raise HCLNotImplementedError(
                f"{type(op)}'s build visitor is not implemented yet."
            )

    def build_func_op(self, op: ast.FuncOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        # use global insetion point instead
        ip = InsertionPoint(self.module.body)
        input_types = []
        input_typehints = []
        for arg in op.args:
            if isinstance(arg, ast.AllocOp):
                ele_type = hcl_dtype_to_mlir(arg.dtype, signless=True)
                input_typehints.append(get_extra_type_hints(arg.dtype))
                memref_type = MemRefType.get(arg.shape, ele_type)
                input_types.append(memref_type)
            else:
                dtype = self.tinf_engine.infer(arg)
                input_typehints.append(get_extra_type_hints(dtype))
                dtype = hcl_dtype_to_mlir(dtype, signless=True)
                input_types.append(dtype)
        output_types = []
        output_typehints = []
        for ret in op.return_tensors:
            if isinstance(ret, ast.AllocOp):
                ele_type = hcl_dtype_to_mlir(ret.dtype, signless=True)
                output_typehints.append(get_extra_type_hints(ret.dtype))
                memref_type = MemRefType.get(ret.shape, ele_type)
                output_types.append(memref_type)
            else:
                dtype = self.tinf_engine.infer(ret)
                output_typehints.append(get_extra_type_hints(dtype))
                dtype = hcl_dtype_to_mlir(dtype, signless=True)
                output_types.append(dtype)
        func_type = FunctionType.get(input_types, output_types)
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        func_op = func_d.FuncOp(name=op.name, type=func_type, ip=ip, loc=loc)
        op.ir_op = func_op

        if op.prototype:
            # function prototype, function visibility is private
            func_op.attributes["sym_visibility"] = StringAttr.get("private")
            return

        func_op.add_entry_block()
        for arg, block_arg in zip(op.args, func_op.entry_block.arguments):
            arg.result = block_arg

        # build body
        ip = InsertionPoint(func_op.entry_block)
        for body_op in op.body:
            self.build_visitor(body_op, ip)
        for ret in op.return_tensors:
            self.build_visitor(ret, ip)
        returns = [ret.result for ret in op.return_tensors]
        func_d.ReturnOp(returns, ip=ip, loc=loc)
        func_op.attributes["function_type"] = TypeAttr.get(func_type)
        # attach type hints
        otypes = "".join(output_typehints)
        itypes = "".join(input_typehints)
        func_op.attributes["otypes"] = StringAttr.get(otypes)
        func_op.attributes["itypes"] = StringAttr.get(itypes)
        if self.BIT_OPS:
            # if function body has bit operations
            # add bit attribute to let VHLS know
            # that it should use ap_int type for integers
            func_op.attributes["bit"] = UnitAttr.get()

        # It is necessary to clear the result of each argument
        # as the same argument object may be refered in multiple functions
        # we need to make sure that the result is not reused
        for arg in op.args:
            arg.prev_result = arg.result
            arg.result = None

    def build_call_op(self, op: ast.CallOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        func = FlatSymbolRefAttr.get(op.name)
        # build arguments
        args = []
        for arg in op.args:
            self.build_visitor(arg, ip)
            args.append(arg.result)
        return_types = []
        for ret in op.rets:
            if isinstance(ret, ast.AllocOp):
                ele_type = hcl_dtype_to_mlir(ret.dtype, signless=True)
                memref_type = MemRefType.get(ret.shape, ele_type)
                return_types.append(memref_type)
            else:
                dtype = self.tinf_engine.infer(ret)
                dtype = hcl_dtype_to_mlir(dtype, signless=True)
                return_types.append(dtype)
        call_op = func_d.CallOp(return_types, func, args, ip=ip, loc=loc)
        op.ir_op = call_op
        if len(op.rets) > 0:
            if len(op.rets) == 1:
                op.result = call_op.results[0]
            else:
                raise HCLNotImplementedError(
                    "Multiple return values are not supported by @def_."
                )

    def build_iter_var(self, iv, ip):
        """Build IterVar"""
        if iv.parent_loop is None:
            raise APIError(f"IterVar {iv} parent loop has not been set")
        iv.result = iv.parent_loop.induction_variable

    def build_for_loop(
        self, lb, ub, step=1, name="", stage="", reduction=False, ip=None, loc=None
    ):
        """Build Affine or SCF for loop.
        If the upper and lower bounds are constant, build an AffineForOp.
        Otherwise, build an SCFForOp.

        Parameters
        ----------
        lb : int or Expr
            Lower bound of the loop.
        ub : int or Expr
            Upper bound of the loop.
        step : int
            Step of the loop.
        name : str
            Name of the loop.
        stage : str
            Name of the stage.
        reduction : bool
            Whether the loop is a reduction loop.
        ip : InsertPoint
            Insert point of the loop.
        loc : Location
            Instance of MLIR Location. Represents location in source code.
        """
        if not isinstance(step, int):
            raise HCLNotImplementedError("Non-constant step size is not supported yet")
        if step < 0:  # swap lb and ub
            lb, ub = ub + 1, lb + 1
            step = -step

        if isinstance(lb, int) and isinstance(ub, int):  # build affine for loop
            lbCst = AffineConstantExpr.get(lb)
            lbMap = AffineMap.get(dim_count=0, symbol_count=0, exprs=[lbCst])
            lbMapAttr = AffineMapAttr.get(lbMap)
            lb_expr = None
            ubCst = AffineConstantExpr.get(ub)
            ubMap = AffineMap.get(dim_count=0, symbol_count=0, exprs=[ubCst])
            ubMapAttr = AffineMapAttr.get(ubMap)
            ub_expr = None
            step = IntegerAttr.get(IntegerType.get_signless(32), step)
            for_op = affine_d.AffineForOp(
                lb_expr,
                ub_expr,
                step,
                lbMapAttr,
                ubMapAttr,
                name=(
                    StringAttr.get("") if name in {"", None} else StringAttr.get(name)
                ),
                stage=("" if stage == "" else StringAttr.get(stage)),
                reduction=(UnitAttr.get() if reduction else None),
                ip=ip,
                loc=loc,
            )
            affine_d.AffineYieldOp([], ip=InsertionPoint(for_op.body))
        else:  # build scf for loop
            # cast lb and up to index type
            itmd_loc = ast.Location("unknown", 0)
            lb = ast.immediate_to_constant(lb, itmd_loc, htypes.Index())
            ub = ast.immediate_to_constant(ub, itmd_loc, htypes.Index())
            self.build_visitor(lb, ip)
            self.build_visitor(ub, ip)
            lb_cast = ast.CastOp(lb, htypes.Index(), loc=itmd_loc)
            ub_cast = ast.CastOp(ub, htypes.Index(), loc=itmd_loc)
            self.build_visitor(lb_cast, ip)
            self.build_visitor(ub_cast, ip)
            step_cast = ast.immediate_to_constant(step, itmd_loc, htypes.Index())
            self.build_visitor(step_cast, ip)
            for_op = scf_d.ForOp(
                lb_cast.result,
                ub_cast.result,
                step_cast.result,
                name=(
                    StringAttr.get("") if name in {"", None} else StringAttr.get(name)
                ),
                stage=("" if stage == "" else StringAttr.get(stage)),
                reduction=(UnitAttr.get() if reduction else None),
                ip=ip,
                loc=loc,
            )
            scf_d.YieldOp([], ip=InsertionPoint(for_op.body))
        return for_op

    def build_compute(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        iv_names = [iv.name for iv in op.iter_vars]
        with get_context(), loc:
            # build output tensor
            alloc_op = op.tensor
            if alloc_op is not None:
                self.build_visitor(alloc_op, ip)
                op.result = alloc_op.result
                op.ir_op = alloc_op

            loops = []
            for i, (ub, loop_name) in enumerate(zip(op.shape, iv_names)):
                loop = self.build_for_loop(
                    0,
                    ub,
                    step=1,
                    name=loop_name,
                    stage=(op.name if i == 0 else ""),
                    ip=ip,
                    loc=loc,
                )
                loops.append(loop)
                ip = InsertionPoint(loop.body.operations[0])
            for iter_var, loop in zip(op.iter_vars, loops):
                iter_var.parent_loop = loop
            for body_op in op.body:
                self.build_visitor(body_op, ip)

    def build_for_op(self, op: ast.ForOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        with get_context(), loc:
            stage = "" if op.tag is None else op.tag
            loop = self.build_for_loop(
                op.low, op.high, op.step, op.name, stage=stage, ip=ip, loc=loc
            )
            ip = InsertionPoint(loop.body.operations[0])
            op.iter_var.parent_loop = loop
            for body_op in op.body:
                self.build_visitor(body_op, ip)

    def build_while_op(self, op: ast.WhileOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        with get_context(), loc:
            # bulid empty while loop
            while_op = scf_d.WhileOp([], [], ip=ip, loc=loc)
            while_op.before.blocks.append(*[])
            while_op.after.blocks.append(*[])
            # build condition
            cond_ip = InsertionPoint(while_op.before.blocks[0])
            self.build_visitor(op.cond, cond_ip)
            scf_d.ConditionOp(op.cond.result, [], ip=cond_ip, loc=loc)
            body_ip = InsertionPoint(while_op.after.blocks[0])
            # build yield
            scf_d.YieldOp([], ip=body_ip, loc=loc)
            # build body
            body_ip = InsertionPoint(while_op.after.blocks[0].operations[0])
            for body_op in op.body:
                self.build_visitor(body_op, body_ip)
        op.ir_op = while_op

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
            raise APIError(f"Tensor name conflict: {op.name}")
        self.tensor_dict[op.name] = alloc_op

    def build_binary_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)

        # Step 1: build lhs and rhs
        self.build_visitor(op.lhs, ip)
        self.build_visitor(op.rhs, ip)

        # Step 2: cast lhs and rhs to the same type
        t = self.tinf_engine.infer(op)
        lhs = ast.CastOp(op.lhs, t, op.loc)
        rhs = ast.CastOp(op.rhs, t, op.loc)
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

        # Bypass for left shift an integer by its bitwidth
        if isinstance(op, ast.LeftShiftOp) and isinstance(t, (htypes.Int, htypes.UInt)):
            i1 = IntegerType.get_signless(1)
            i64 = IntegerType.get_signless(64)
            sge_attr = IntegerAttr.get(i64, 5)  # signed greater or equal
            uge_attr = IntegerAttr.get(i64, 9)  # unsigned greater or equal
            shift_type = hcl_dtype_to_mlir(t, signless=True)
            bitwidth = arith_d.ConstantOp(
                shift_type, IntegerAttr.get(shift_type, t.bits), ip=ip, loc=loc
            )
            op_attr = sge_attr if isinstance(t, htypes.Int) else uge_attr
            cond = arith_d.CmpIOp(
                i1, op_attr, rhs.result, bitwidth.result, ip=ip, loc=loc
            )
            zero = arith_d.ConstantOp(
                shift_type, IntegerAttr.get(shift_type, 0), ip=ip, loc=loc
            )
            select = arith_d.SelectOp(
                cond.result, zero.result, binary_op.result, ip=ip, loc=loc
            )
            if isinstance(t, htypes.UInt):
                select.attributes["unsigned"] = UnitAttr.get()
            op.result = select.result
            op.ir_op = select

    def build_math_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.expr, ip)
        casted = ast.CastOp(op.expr, op.dtype, loc)
        self.build_visitor(casted, ip)
        op_class = get_op_class(op, op.dtype)
        math_op = op_class(casted.result, ip=ip, loc=loc)
        op.result = math_op.result
        op.ir_op = math_op

    def build_neg_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.expr, ip)
        t = self.tinf_engine.infer(op.expr)

        if isinstance(t, htypes.Float):
            neg_op = arith_d.NegFOp(op.expr.result, ip=ip, loc=loc)
        elif isinstance(t, htypes.Int):
            # A bypass to avoid type casting for type safety.
            # Since we know this is a negation on integer,
            # we don't have to worry about overflow.
            neg_one = ast.ConstantOp(-1, t, op.loc)
            self.build_visitor(neg_one, ip)
            neg_op = arith_d.MulIOp(op.expr.result, neg_one.result, ip=ip, loc=loc)
        else:
            mul_neg_one = ast.Mul(op.expr, -1, op.loc)
            casted = ast.CastOp(mul_neg_one, t, op.loc)
            self.build_visitor(casted, ip)
            neg_op = casted.ir_op

        op.result = neg_op.result
        op.ir_op = neg_op

        if isinstance(t, (htypes.UInt, htypes.UFixed)):
            neg_op.attributes["unsigned"] = UnitAttr.get()

    def build_cmp_op(self, op: ast.Cmp, ip):
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
        if isinstance(t, tuple):
            t = t[0]  # index 0 is src type, index 1 is res type
        lhs = ast.CastOp(op.lhs, t, loc)
        rhs = ast.CastOp(op.rhs, t, loc)
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
            attr = ATTR_MAP["float"]["o" + op.name]
        elif isinstance(t, htypes.Fixed):
            OpClass = hcl_d.CmpFixedOp
            attr = ATTR_MAP["fixed"][
                "s" + op.name if op.name not in ["eq", "ne"] else op.name
            ]
        elif isinstance(t, htypes.UFixed):
            OpClass = hcl_d.CmpFixedOp
            attr = ATTR_MAP["fixed"][
                "u" + op.name if op.name not in ["eq", "ne"] else op.name
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

    def build_load_op(self, op: ast.LoadOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        index_exprs = []
        flag = True
        load_op = None
        self.iv.clear()  # clear iv
        for index in op.index:
            try:
                affine_expr = self.build_affine_expr(index)
                index_exprs.append(affine_expr)
            # pylint: disable=broad-exception-caught
            except Exception:
                flag = False
                break
        if flag:
            dim_count = len(self.iv)
            affine_map = AffineMap.get(
                dim_count=dim_count, symbol_count=0, exprs=index_exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            load_op = affine_d.AffineLoadOp(
                op.tensor.result, self.iv, affine_attr, ip=ip, loc=loc
            )
            op.result = load_op.result
            op.ir_op = load_op
        else:
            new_indices = []
            for index in op.index:
                self.build_visitor(index, ip)
                # cast to index type
                index = ast.CastOp(index, htypes.Index(), loc)
                self.build_visitor(index, ip)
                new_indices.append(index.result)
            # pylint: disable=no-value-for-parameter
            load_op = memref_d.LoadOp(op.tensor.result, new_indices, ip=ip, loc=loc)
            op.result = load_op.result
            op.ir_op = load_op

        load_op.attributes["from"] = StringAttr.get(op.tensor.name)
        if isinstance(op.dtype, htypes.UInt):
            load_op.attributes["unsigned"] = UnitAttr.get()

    def build_store_op(self, op: ast.StoreOp, ip):
        index_exprs = []
        flag = True
        store_op = None
        if op.value.result is None:
            self.build_visitor(op.value, ip)
        casted_expr = ast.CastOp(op.value, op.tensor.dtype, op.loc)
        self.build_visitor(casted_expr, ip)
        self.iv.clear()  # clear iv
        for index in op.index:
            try:
                affine_expr = self.build_affine_expr(index)
                index_exprs.append(affine_expr)
            # pylint: disable=broad-exception-caught
            except Exception:
                flag = False
                break
        if flag:
            dim_count = len(self.iv)
            affine_map = AffineMap.get(
                dim_count=dim_count, symbol_count=0, exprs=index_exprs
            )
            affine_attr = AffineMapAttr.get(affine_map)
            store_op = affine_d.AffineStoreOp(
                casted_expr.result, op.tensor.result, self.iv, affine_attr, ip=ip
            )
        else:
            new_indices = []
            for index in op.index:
                self.build_visitor(index, ip)
                index = ast.CastOp(index, htypes.Index(), op.loc)
                self.build_visitor(index, ip)
                new_indices.append(index.result)
            store_op = memref_d.StoreOp(
                casted_expr.result, op.tensor.result, new_indices, ip=ip
            )
        # we don't need to set the result of store op
        # because store op doesn't have a result
        store_op.attributes["to"] = StringAttr.get(op.tensor.name)
        if isinstance(op.tensor.dtype, htypes.UInt):
            store_op.attributes["unsigned"] = UnitAttr.get()
        op.ir_op = store_op

    def build_constant_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        dtype = hcl_dtype_to_mlir(op.dtype)
        if isinstance(op.dtype, (htypes.Int, htypes.UInt)):
            if isinstance(op.dtype, htypes.Index):
                value_attr = IntegerAttr.get(IndexType.get(), op.value)
            elif op.dtype.bits == 1:
                value_attr = BoolAttr.get(op.value)
            # pylint: disable=else-if-used
            else:
                if op.dtype.bits < 64:
                    attr_type = IntegerType.get_signless(op.dtype.bits)
                    value_attr = IntegerAttr.get(attr_type, op.value)
                elif op.dtype.bits == 64:
                    value_attr = IntegerAttr.parse(str(op.value))
                else:
                    raise MLIRLimitationError(
                        f"Could not create constant op for value {op.value}, "
                        + "MLIR IntegerAttr only supports up to 64-bit integer values."
                        + f"This value requires {op.value.bit_length()} bits."
                    )
            const_op = arith_d.ConstantOp(dtype, value_attr, ip=ip, loc=loc)
        elif isinstance(op.dtype, htypes.Float):
            if op.dtype.bits == 16:
                value_attr = FloatAttr.get(F16Type.get(), op.value)
            elif op.dtype.bits == 32:
                value_attr = FloatAttr.get(F32Type.get(), op.value)
            elif op.dtype.bits == 64:
                value_attr = FloatAttr.get(F64Type.get(), op.value)
            else:
                raise DTypeError(f"Unsupported float type: {op.dtype}")
            const_op = arith_d.ConstantOp(dtype, value_attr, ip=ip, loc=loc)
        elif isinstance(op.dtype, (htypes.Fixed, htypes.UFixed)):
            # assume the value is converted to integer base
            if not isinstance(op.value, int):
                raise DTypeError("Fixed point value must be converted to integer base")
            attr_type = IntegerType.get_signless(op.dtype.bits)
            value_attr = IntegerAttr.get(attr_type, op.value)
            const_op = arith_d.ConstantOp(dtype, value_attr, ip=ip, loc=loc)
        else:
            raise DTypeError(f"Unsupported type: {op.dtype}")

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
        if isinstance(src_type, tuple):
            src_type = src_type[1]
        # determine cast op
        CastOpClass = None
        # pylint: disable=unidiomatic-typecheck
        if type(res_type) == type(src_type) and res_type == src_type:
            op.result = op.expr.result
            op.ir_op = op.expr.ir_op
            return
        elif isinstance(src_type, (htypes.Int, htypes.UInt)) and isinstance(
            res_type, htypes.Index
        ):
            CastOpClass = arith_d.IndexCastOp
        elif isinstance(src_type, htypes.Index) and isinstance(
            res_type, (htypes.Int, htypes.UInt)
        ):
            CastOpClass = arith_d.IndexCastOp
        elif isinstance(src_type, htypes.Int) and isinstance(res_type, htypes.Float):
            CastOpClass = arith_d.SIToFPOp
        elif isinstance(src_type, htypes.UInt) and isinstance(res_type, htypes.Float):
            CastOpClass = arith_d.UIToFPOp
        elif isinstance(src_type, htypes.Float) and isinstance(res_type, htypes.Int):
            CastOpClass = arith_d.FPToSIOp
        elif isinstance(src_type, htypes.Float) and isinstance(res_type, htypes.Index):
            # FP to Index is not supported in MLIR
            # we need to cast to UInt first, then cast to Index
            cast_to_uint = ast.CastOp(op.expr, htypes.UInt(res_type.bits), op.loc)
            self.build_cast_op(cast_to_uint, ip)  # build cast to uint
            op.expr = cast_to_uint  # replace expr with cast to uint
            CastOpClass = arith_d.IndexCastOp  # proceed to build cast to index
        elif isinstance(src_type, htypes.Float) and isinstance(res_type, htypes.UInt):
            CastOpClass = arith_d.FPToUIOp
        elif isinstance(src_type, (htypes.Int, htypes.UInt)) and isinstance(
            res_type, (htypes.Int, htypes.UInt)
        ):
            if src_type.bits > res_type.bits:
                CastOpClass = arith_d.TruncIOp
            elif src_type.bits == res_type.bits:
                op.result = op.expr.result
                op.ir_op = op.expr.ir_op
                return
            else:  # src_type.bits < res_type.bits
                if (
                    isinstance(op.expr, (ast.GetBitOp, ast.GetSliceOp, ast.LeftShiftOp))
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
        elif isinstance(src_type, htypes.Float) and isinstance(
            res_type, (htypes.Fixed, htypes.UFixed)
        ):
            CastOpClass = hcl_d.FloatToFixedOp
        elif isinstance(src_type, (htypes.Fixed, htypes.UFixed)) and isinstance(
            res_type, htypes.Float
        ):
            CastOpClass = hcl_d.FixedToFloatOp
        elif isinstance(src_type, (htypes.Fixed, htypes.UFixed)) and isinstance(
            res_type, (htypes.Int, htypes.UInt)
        ):
            CastOpClass = hcl_d.FixedToIntOp
        elif isinstance(src_type, (htypes.Int, htypes.UInt)) and isinstance(
            res_type, (htypes.Fixed, htypes.UFixed)
        ):
            CastOpClass = hcl_d.IntToFixedOp
        elif isinstance(src_type, (htypes.Fixed, htypes.UFixed)) and isinstance(
            res_type, (htypes.Fixed, htypes.UFixed)
        ):
            if src_type == res_type:
                op.result = op.expr.result
                op.ir_op = op.expr.ir_op
                return
            else:
                CastOpClass = hcl_d.FixedToFixedOp
        elif isinstance(src_type, htypes.Struct) and isinstance(
            res_type, htypes.Struct
        ):
            # We don't actually cast between struct types,
            # here we check if two structs are identical when all
            # integer fields are signless.
            if len(src_type.dtype_dict) != len(res_type.dtype_dict):
                raise DTypeError(
                    "Casting between structs with different number of fields. "
                    + f"src type: {src_type}, dst type: {res_type}"
                )
            for res_ftype, src_ftype in zip(
                res_type.dtype_dict.values(), src_type.dtype_dict.values()
            ):
                if isinstance(src_ftype, (htypes.Int, htypes.UInt)) and isinstance(
                    res_ftype, (htypes.Int, htypes.UInt)
                ):
                    if src_ftype.width != res_ftype.width:
                        raise DTypeError(
                            "Casting between structs with different field width. "
                            + f"src type: {src_type}, dst type: {res_type}"
                        )
                else:
                    raise DTypeError(
                        "Casting between structs with different field types. "
                        + f"src type: {src_type}, dst type: {res_type}"
                    )
            op.result = op.expr.result
            op.ir_op = op.expr.ir_op
            return
        elif isinstance(src_type, (htypes.Int, htypes.UInt)) and isinstance(
            res_type, htypes.Struct
        ):
            # Int -> Struct Cast
            if not is_all_field_int(res_type):
                raise DTypeError(
                    "Casting from integer to struct with non-integer fields. "
                    + f"src type: {src_type}, dst type: {res_type}"
                )
            total_width = get_struct_bitwidth(res_type)
            if total_width != src_type.bits:
                raise DTypeError(
                    "Casting from integer to struct with different width. "
                    + f"src type: {src_type}, dst type: {res_type}"
                )
            CastOpClass = hcl_d.IntToStructOp
        elif isinstance(src_type, htypes.Struct) and isinstance(
            res_type, (htypes.Int, htypes.UInt)
        ):
            # Struct -> Int Cast
            raise HCLNotImplementedError(
                "Struct -> Int Cast is not implemented yet. "
                + "We plan to add an as_int() API for struct values."
            )
        else:
            raise DTypeError(
                "Casting between unsupported types. "
                + f"src type: {src_type}, dst type: {res_type}"
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
        if not isinstance(
            expr, (ast.IterVar, ast.ConstantOp, ast.CastOp, ast.BinaryOp)
        ):
            raise HCLValueError(f"{expr} is not an affine index")
        if isinstance(expr, ast.IterVar):
            if expr.parent_loop is None:
                raise HCLValueError(f"{expr} does not have parent loop set")
            if isinstance(expr.parent_loop, scf_d.ForOp):
                raise HCLValueError(f"loop {expr.parent_loop} is not affine")
            if expr.parent_loop.induction_variable not in self.iv:
                self.iv.append(expr.parent_loop.induction_variable)  # BlockArgument
                return AffineExpr.get_dim(len(self.iv) - 1)
            else:
                return AffineExpr.get_dim(
                    self.iv.index(expr.parent_loop.induction_variable)
                )
        elif isinstance(expr, ast.ConstantOp):
            return AffineExpr.get_constant(expr.value)
        elif isinstance(expr, ast.CastOp):
            return self.build_affine_expr(expr.expr)
        lhs = self.build_affine_expr(expr.lhs)
        rhs = self.build_affine_expr(expr.rhs)
        if isinstance(expr, ast.Add):
            return lhs + rhs
        elif isinstance(expr, ast.Sub):
            return lhs - rhs
        elif isinstance(expr, ast.Mul):
            return lhs * rhs
        elif isinstance(expr, ast.Div):
            return AffineExpr.get_floor_div(lhs, rhs)  # or get_ceil_div
        elif isinstance(expr, ast.Mod):
            return lhs % rhs
        else:
            raise HCLValueError(f"{expr} is not an affine index!")

    def build_if_op(self, op: ast.IfOp, ip):
        """Build IfOp"""
        # TODO: support affine if
        # build condition
        # clear condition build result from previous build
        self.cleaner.visit(op.cond)
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
        op.ir_op = if_op

    def build_reduce(self, op: ast.ReduceOp, ip):
        """Build ReduceOp"""
        # scalar to hold the reduction result
        self.build_visitor(op.scalar, ip)
        # store the init value to the scalar
        init = ast.immediate_to_constant(op.init, op.loc)
        self.build_visitor(init, ip)
        zero_idx = ast.ConstantOp(0, htypes.Index(), op.loc)
        self.build_visitor(zero_idx, ip)
        store_op = ast.StoreOp(op.scalar, (zero_idx,), init, op.loc)
        self.build_visitor(store_op, ip)
        body_ip = ip

        # build loop nest
        loops = []
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        for axis in op.axis:
            lb, ub = axis.bound
            loop = self.build_for_loop(
                lb, ub, step=1, reduction=True, name=axis.name, ip=body_ip, loc=loc
            )
            axis.parent_loop = loop
            loops.append(loop)
            body_ip = InsertionPoint(loop.body.operations[0])

        # build body op
        for body_op in op.body:
            self.build_visitor(body_op, body_ip)

        # load from the scalar
        load_op = ast.LoadOp(op.scalar, (zero_idx,), op.loc)
        self.build_visitor(load_op, ip)
        op.ir_op = load_op
        op.result = load_op.result

    def build_select_op(self, op: ast.SelectOp, ip):
        # Step 1: get condition, true, and false value
        # if any of them is an immediate, convert it to a constant
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        cond = ast.immediate_to_constant(op.cond, op.loc)
        self.build_visitor(cond, ip)
        true_value = ast.immediate_to_constant(op.true_value, op.loc)
        self.build_visitor(true_value, ip)
        false_value = ast.immediate_to_constant(op.false_value, op.loc)
        self.build_visitor(false_value, ip)

        # Step 2: type inference and cast
        res_type = self.tinf_engine.infer(op)
        # cast condition to uint1
        # pylint: disable=redefined-variable-type
        if not isinstance(cond, ast.Cmp):
            cond = ast.CastOp(cond, htypes.UInt(1), op.loc)
            self.build_visitor(cond, ip)
        # cast true and false value to the same type
        true_value = ast.CastOp(true_value, res_type, op.loc)
        self.build_visitor(true_value, ip)
        false_value = ast.CastOp(false_value, res_type, op.loc)
        self.build_visitor(false_value, ip)

        # Step 3: build select op
        select_op = arith_d.SelectOp(
            cond.result, true_value.result, false_value.result, ip=ip, loc=loc
        )
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

        # add \00 llvm terminating character
        op.fmt += "\00"

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

    def build_get_bit_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.expr, ip)
        # check if expr is int type
        expr_dtype = self.tinf_engine.infer(op.expr)
        if not isinstance(expr_dtype, (htypes.Int, htypes.UInt)):
            raise APIError("Get bit operation only supports integer type")
        # cast index to index type
        index = ast.CastOp(op.index, htypes.Index(), op.loc)
        self.build_visitor(index, ip)

        # build get bit op
        res_dtype = hcl_dtype_to_mlir(htypes.UInt(1), signless=True)
        getbit_op = hcl_d.GetIntBitOp(
            res_dtype, op.expr.result, index.result, ip=ip, loc=loc
        )
        op.ir_op = getbit_op
        op.result = getbit_op.result

    def build_get_slice_op(self, op: ast.GetSliceOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.expr, ip)
        # check if expr is int type
        expr_dtype = self.tinf_engine.infer(op.expr)
        if not isinstance(expr_dtype, (htypes.Int, htypes.UInt)):
            raise APIError("Get bit operation only supports integer type")
        # cast start index to index type
        start = ast.CastOp(op.start, htypes.Index(), op.loc)
        self.build_visitor(start, ip)
        # cast end index to index type
        end = ast.CastOp(op.end, htypes.Index(), op.loc)
        self.build_visitor(end, ip)

        res_dtype = hcl_dtype_to_mlir(op.dtype, signless=True)
        getbit_op = hcl_d.GetIntSliceOp(
            res_dtype, op.expr.result, end.result, start.result, ip=ip, loc=loc
        )
        op.ir_op = getbit_op
        op.result = getbit_op.result

    def build_set_bit_op(self, op, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.expr, ip)
        self.build_visitor(op.value, ip)
        # check if expr is int type
        expr_dtype = self.tinf_engine.infer(op.expr)
        if not isinstance(expr_dtype, (htypes.Int, htypes.UInt)):
            raise APIError("Set bit operation only supports integer type")
        expr_dtype = hcl_dtype_to_mlir(expr_dtype, signless=True)
        # cast index to index type
        index = ast.CastOp(op.index, htypes.Index(), op.loc)
        self.build_visitor(index, ip)
        # cast value to uint1
        value = ast.CastOp(op.value, htypes.UInt(1), op.loc)
        self.build_visitor(value, ip)

        # build set bit op
        setbit_op = hcl_d.SetIntBitOp(
            expr_dtype, op.expr.result, index.result, value.result, ip=ip, loc=loc
        )
        op.ir_op = setbit_op
        op.result = setbit_op.result

        # if expr is a LoadOp, we need to update the value in the tensor
        if isinstance(op.expr, ast.LoadOp):
            # build store op
            load_op = op.expr
            store_op = ast.StoreOp(load_op.tensor, load_op.index, op, op.loc)
            self.build_visitor(store_op, ip)

    def build_set_slice_op(self, op: ast.SetSliceOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.expr, ip)
        self.build_visitor(op.value, ip)
        # check if expr is int type
        expr_dtype = self.tinf_engine.infer(op.expr)
        if not isinstance(expr_dtype, (htypes.Int, htypes.UInt)):
            raise APIError("Set bit operation only supports integer type expr")

        # cast start, end indices to index type
        start = ast.CastOp(op.start, htypes.Index(), op.loc)
        self.build_visitor(start, ip)
        end = ast.CastOp(op.end, htypes.Index(), op.loc)
        self.build_visitor(end, ip)

        expr_dtype = hcl_dtype_to_mlir(expr_dtype, signless=True)

        # build set bit op
        setbit_op = hcl_d.SetIntSliceOp(
            expr_dtype,
            op.expr.result,
            end.result,
            start.result,
            op.value.result,
            ip=ip,
            loc=loc,
        )
        op.ir_op = setbit_op
        op.result = setbit_op.result

        # if expr is a LoadOp, we need to update the value in the tensor
        if isinstance(op.expr, ast.LoadOp):
            # build store op
            load_op = op.expr
            store_op = ast.StoreOp(load_op.tensor, load_op.index, op, op.loc)
            self.build_visitor(store_op, ip)

    def build_bit_reverse_op(self, op: ast.BitReverseOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.expr, ip)
        # check if expr is int type
        expr_dtype = self.tinf_engine.infer(op.expr)
        if not isinstance(expr_dtype, (htypes.Int, htypes.UInt)):
            raise APIError("Bit reverse operation only supports integer type")
        bitreverse_op = hcl_d.BitReverseOp(op.expr.result, ip=ip, loc=loc)
        op.ir_op = bitreverse_op
        op.result = bitreverse_op.result

    def build_constant_tensor_op(self, op: ast.ConstantTensorOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        dtype = hcl_dtype_to_mlir(op.dtype, signless=True)
        shape = op.values.shape
        if isinstance(op.dtype, (htypes.Int, htypes.UInt)):
            # The following code has several steps to convert the numpy array to have
            # the correct data type in order to create an MLIR constant tensor.
            # Since MLIR-NumPy Python interface only supports byte-addressable data types,
            # we need to change the data type of the array to have the minimum number of bytes
            # that can represent the target bitwidth.
            # e.g., hcl.const_tensor(arr, dtype=hcl.Int(20)) (6*6 array)
            #       which requires 20 bits (3 bytes) to represent each element
            # declaration: 6*6*i20
            # numpy input: 6*6*i64
            # 1. Decompose the original i32 or i64 array into a structured array of uint8
            #  -> decompose: 6*6*8*i8
            if op.dtype.bits == 1:
                val = op.values
                array = np.packbits(val, axis=None, bitorder="little")
                value_attr = DenseElementsAttr.get(array, shape=val.shape, type=dtype)
            else:
                # Here we construct a customized NumPy dtype, "f0", "f1", "f2", etc.
                # are the field names, and the entire data type is `op.values.dtype`.
                # This can be viewed as a `union` type in C/C++.
                # Please refer to the documentation for more details:
                # https://numpy.org/doc/stable/reference/arrays.dtypes.html#specifying-and-constructing-data-types
                decomposed_np_dtype = np.dtype(
                    (
                        op.values.dtype,
                        {
                            f"f{i}": (np.uint8, i)
                            for i in range(op.values.dtype.itemsize)
                        },
                    )
                )
                val = op.values.view(decomposed_np_dtype)
                # 2. Compose the uint8 array into a structured array of target bitwidth
                # This is done by taking the first several bytes of the uint8 array
                # "u1" means one unsigned byte, and "i1" means one signed byte
                n_bytes = int(np.ceil(dtype.width / 8))
                new_dtype = np.dtype(
                    {
                        "names": [f"f{i}" for i in range(n_bytes)],
                        "formats": (["i1"] if isinstance(dtype, htypes.Int) else ["u1"])
                        + ["u1"] * (n_bytes - 1),
                        "offsets": list(range(n_bytes)),
                        "itemize": n_bytes,
                    }
                )
                # -> compose: 6*6*3*i8
                val = np.stack([val[f"f{i}"] for i in range(n_bytes)], axis=-1)
                # -> flatten: 108*i8
                val = val.flatten()
                # -> view: 36*i24
                val = val.view(np.dtype(new_dtype))
                # -> reshape: 6*6*i24
                val = val.reshape(shape)
                # Pass in the numpy array to get the MLIR attribute
                # -> result: 6*6*i20
                value_attr = DenseElementsAttr.get(val, shape=val.shape, type=dtype)
        else:
            val = op.values
            value_attr = DenseElementsAttr.get(val)
        sym_name = StringAttr.get(op.name)
        sym_visibility = StringAttr.get("private")
        if isinstance(op.dtype, (htypes.Fixed, htypes.UFixed)):
            memref_type = MemRefType.get(op.shape, IntegerType.get_signless(64))
        else:
            memref_type = MemRefType.get(op.shape, dtype)
        type_attr = TypeAttr.get(memref_type)
        const_tensor = memref_d.GlobalOp(
            sym_name,
            type_attr,
            sym_visibility=sym_visibility,
            initial_value=value_attr,
            constant=True,
            alignment=None,
            ip=InsertionPoint(self._ast.top_func.ir_op),
            loc=loc,
        )
        const_tensor.attributes["constant"] = UnitAttr.get()
        if isinstance(op.dtype, (htypes.UInt, htypes.UFixed)):
            const_tensor.attributes["unsigned"] = UnitAttr.get()

        if isinstance(op.dtype, (htypes.Fixed, htypes.UFixed)):
            fixed_memref_type = MemRefType.get(val.shape, dtype)
            get_global = hcl_d.GetGlobalFixedOp(
                fixed_memref_type, FlatSymbolRefAttr.get(op.name), ip=ip, loc=loc
            )
        else:
            get_global = memref_d.GetGlobalOp(
                memref_type, FlatSymbolRefAttr.get(op.name), ip=ip, loc=loc
            )
        op.ir_op = get_global
        op.result = get_global.result
        op.tensor.ir_op = get_global
        op.tensor.result = get_global.result

    def build_struct_construct_op(self, op: ast.StructConstructOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        # build fields
        field_results = []
        for idx, field in enumerate(op.args):
            field_key_list = list(op.dtype.dtype_dict.keys())
            type_key = field_key_list[idx]
            field = ast.CastOp(field, op.dtype.dtype_dict[type_key], op.loc)
            self.build_visitor(field, ip)
            field_results.append(field.result)
        field_types = [f.type for f in field_results]
        struct_type = hcl_d.StructType.get(field_types)
        struct_op = hcl_d.StructConstructOp(struct_type, field_results, ip=ip, loc=loc)
        op.ir_op = struct_op
        op.result = struct_op.result

    def build_struct_get_op(self, op: ast.StructGetOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.struct, ip)
        dtype = self.tinf_engine.infer(op)
        dtype = hcl_dtype_to_mlir(dtype, signless=True)
        assert isinstance(op.field, int)
        attr = IntegerAttr.get(IntegerType.get_signless(64), op.field)
        struct_get_op = hcl_d.StructGetOp(dtype, op.struct.result, attr, ip=ip, loc=loc)
        if isinstance(dtype, (htypes.UInt, htypes.UFixed)):
            struct_get_op.attr["unsigned"] = UnitAttr.get()
        op.ir_op = struct_get_op
        op.result = struct_get_op.result

    def build_op_handle(self, op: ast.OpHandle, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        hdl_op = hcl_d.CreateOpHandleOp(StringAttr.get(op.name), ip=ip, loc=loc)
        op.ir_op = hdl_op
        op.result = hdl_op.result

    def build_loop_handle(self, op: ast.LoopHandle, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.op_hdl, ip)
        hdl_op = hcl_d.CreateLoopHandleOp(
            op.op_hdl.result, StringAttr.get(op.name), ip=ip, loc=loc
        )
        op.ir_op = hdl_op
        op.result = hdl_op.result

    def build_inter_kernel_to_op(self, op: ast.InterKernelToOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        self.build_visitor(op.tensor, ip)
        self.build_visitor(op.stage, ip)
        i32 = IntegerType.get_signless(32)
        fifo_depth = IntegerAttr.get(i32, op.fifo_depth)
        top_func = self._ast.top_func.ir_op
        assert top_func is not None
        top_func.attributes["dataflow"] = UnitAttr.get()
        to_op = hcl_d.InterKernelToOp(
            op.tensor.result, op.stage.result, fifo_depth=fifo_depth, ip=ip, loc=loc
        )
        op.ir_op = to_op

    def build_outline_op(self, op: ast.OutlineOp, ip):
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        for stage_hdl in op.stage_hdls:
            self.build_visitor(stage_hdl, ip)
        hdl_results = [hdl.result for hdl in op.stage_hdls]
        outline_op = hcl_d.OutlineOp(hdl_results, ip=ip, loc=loc)
        if op.unify is not None:
            outline_op.attributes["unify"] = StringAttr.get(op.unify)
        if op.axis is not None:
            outline_op.attributes["axis"] = StringAttr.get(op.axis)
        op.ir_op = outline_op
