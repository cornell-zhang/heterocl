from ast import Global

from numpy import dtype
from .context import get_context, get_location
from .utils import hcl_dtype_to_mlir
from .types import Int, Type, UInt, Float, dtype_to_hcl

from typing import Callable, Sequence, Mapping, Union

from hcl_mlir import GlobalInsertionPoint
from hcl_mlir.dialects import pdl, transform
from hcl_mlir.ir import *

from hcl_mlir.dialects._ods_common import _cext as _ods_cext
from hcl_mlir.dialects._ods_common import extend_opview_class as _ods_extend_opview_class

_ods_ir = _ods_cext.ir

try:
    from . import _transform_ops_ext as _ods_ext_module
except ImportError:
    _ods_ext_module = None


@_ods_extend_opview_class(_ods_ext_module)
class HCLGetParentLoopOp(OpView):
    OPERATION_NAME = "transform.hcl.get_parent_loop"

    _ODS_REGIONS = (0, True)

    def __init__(self, target, num_loops, *, loc=None, ip=None):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.append(target)
        attributes["num_loops"] = num_loops
        results.extend([operands[0].type] * 1)
        _ods_successors = None
        super().__init__(self.build_generic(
            attributes=attributes, results=results, operands=operands,
            successors=_ods_successors, regions=regions, loc=loc, ip=ip))


@_ods_extend_opview_class(_ods_ext_module)
class HCLUnrollOp(OpView):
    OPERATION_NAME = "transform.hcl.unroll"

    _ODS_REGIONS = (0, True)

    def __init__(self, target, factor, *, loc=None, ip=None):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.append(target)
        attributes["factor"] = factor
        _ods_successors = None
        super().__init__(self.build_generic(
            attributes=attributes, results=results, operands=operands,
            successors=_ods_successors, regions=regions, loc=loc, ip=ip))


@_ods_extend_opview_class(_ods_ext_module)
class HCLSplitOp(OpView):
    OPERATION_NAME = "transform.hcl.split"

    _ODS_REGIONS = (0, True)

    def __init__(self, target, factor, *, loc=None, ip=None):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.append(target)
        attributes["factor"] = factor
        results.extend([operands[0].type] * 2)
        _ods_successors = None
        super().__init__(self.build_generic(
            attributes=attributes, results=results, operands=operands,
            successors=_ods_successors, regions=regions, loc=loc, ip=ip))


@_ods_extend_opview_class(_ods_ext_module)
class HCLPipelineOp(OpView):
    OPERATION_NAME = "transform.hcl.pipeline"

    _ODS_REGIONS = (0, True)

    def __init__(self, target, initial_interval, *, loc=None, ip=None):
        operands = []
        results = []
        attributes = {}
        regions = None
        operands.append(target)
        attributes["initialInterval"] = initial_interval
        results.extend([operands[0].type])
        _ods_successors = None
        super().__init__(self.build_generic(
            attributes=attributes, results=results, operands=operands,
            successors=_ods_successors, regions=regions, loc=loc, ip=ip))


class Handle(object):
    def __init__(self, ssa: Value):
        self.ssa = ssa


class OpHandle(Handle):
    def __init__(self, ssa: Value, dtypes: Sequence = []):
        super().__init__(ssa)
        self.dtypes = dtypes

    def get_op_ssa(self):
        return self.ssa

    def get_dtype(self, index=0):
        try:
            return self.dtypes[index]
        except:
            raise Exception("Result index out of bound")

    def get_result(self, index=0):
        with get_context(), get_location(), GlobalInsertionPoint.get():
            result_ssa = pdl.ResultOp(self.ssa, index).result
            return ValueHandle(self.get_dtype(index), result_ssa, self.ssa)


class TypeHandle(Handle):
    def __init__(self, ssa: Value, dtype):
        super().__init__(ssa)
        self.dtype = dtype


class AttrHandle(Handle):
    def __init__(self, ssa: Value, value, dtype):
        super().__init__(ssa)
        self.value = value
        self.dtype = dtype


class ValueHandle(Handle):
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
    }

    def __init__(self, dtype, ssa: Value, op_ssa: Value = None):
        super().__init__(ssa)
        self.op_ssa = op_ssa
        self.dtype = dtype

    def get_op_ssa(self):
        return self.op_ssa

    # FIXME: Only support single result and same type result and operand ops.
    def get_expr_result(self, name: str, operands: Sequence,
                        signedness: bool = False, compare_pred: str = ""):
        # FIXME: How to infer the type of result&constant?
        def get_or_create_value_ssa(operand):
            if isinstance(operand, ValueHandle):
                return operand.ssa
            elif isinstance(operand, (int, float, bool, Int, UInt, Float)):
                if isinstance(operand, (int, Int, UInt)):
                    type = hcl_dtype_to_mlir(
                        self.dtype) if self.dtype else IntegerType.get_signless(32)
                    attr = pdl.AttributeOp(
                        None, IntegerAttr.get(type, operand))
                elif isinstance(operand, (float, Float)):
                    type = hcl_dtype_to_mlir(
                        self.dtype) if self.dtype else F32Type.get()
                    attr = pdl.AttributeOp(None, FloatAttr.get(type, operand))
                elif isinstance(operand, bool):
                    type = IntegerType.get_signless(1)
                    attr = pdl.AttributeOp(None, BoolAttr.get())

                # type_ssa = pdl.TypeOp(type).result
                type_ssa = pdl.TypesOp()
                op = pdl.OperationOp(
                    "arith.constant", [], {"value": attr}, [type_ssa])
                return pdl.ResultOp(op, 0).result
            else:
                raise Exception("Unsupported operand type, such as HCL Fixed")

        if isinstance(self.dtype, Float):
            name += "f"
        elif isinstance(self.dtype, (Int, UInt)):
            if signedness:
                name += "s" if isinstance(self.dtype, Int) else "u"
            name += "i"
        else:
            raise Exception("Unsupported operand type, such as HCL Fixed")

        with get_context(), get_location(), GlobalInsertionPoint.get():
            result_type = hcl_dtype_to_mlir(self.dtype)
            pred_value = None
            if name == "arith.cmpf":
                # For now, we always use ordered comparison without considering NAN.
                compare_pred = "o" + compare_pred
                pred_value = self.ATTR_MAP["float"][compare_pred]
                result_type = IntegerType.get_signless(1)
            elif name == "arith.cmpi":
                if signedness:
                    if isinstance(self.dtype, Int):
                        compare_pred = "s" + compare_pred
                    else:
                        compare_pred = "u" + compare_pred
                pred_value = self.ATTR_MAP["int"][compare_pred]
                result_type = IntegerType.get_signless(1)

            attr = {}
            if pred_value is not None:
                attr_value = IntegerAttr.get(
                    IntegerType.get_signless(64), pred_value)
                attr = {"predicate": pdl.AttributeOp(None, attr_value).attr}

            operands_ssa = map(get_or_create_value_ssa, operands)
            # type_ssa = pdl.TypeOp(hcl_dtype_to_mlir(result_type)).result
            type_ssa = pdl.TypesOp()
            op_ssa = pdl.OperationOp(name, operands_ssa, attr, [type_ssa]).op
            result_ssa = pdl.ResultOp(op_ssa, 0).result
            return ValueHandle(self.dtype, result_ssa, op_ssa)

    def __add__(self, other):
        return self.get_expr_result("arith.add", [self, other])

    def __radd__(self, other):
        return self.get_expr_result("arith.add", [other, self])

    def __sub__(self, other):
        return self.get_expr_result("arith.sub", [self, other])

    def __rsub__(self, other):
        return self.get_expr_result("arith.sub", [other, self])

    def __mul__(self, other):
        return self.get_expr_result("arith.mul", [self, other])

    def __rmul__(self, other):
        return self.get_expr_result("arith.mul", [other, self])

    def __div__(self, other):
        return self.get_expr_result("arith.div", [self, other], True)

    def __rdiv__(self, other):
        return self.get_expr_result("arith.div", [other, self], True)

    def __truediv__(self, other):
        return self.get_expr_result("arith.ceildiv", [self, other], True)

    def __rtruediv__(self, other):
        return self.get_expr_result("arith.ceildiv", [other, self], True)

    def __floordiv__(self, other):
        return self.get_expr_result("arith.floordiv", [self, other], True)

    def __rfloordiv__(self, other):
        return self.get_expr_result("arith.floordiv", [other, self], True)

    def __mod__(self, other):
        return self.get_expr_result("arith.rem", [self, other], True)

    def __rmod__(self, other):
        return self.get_expr_result("arith.rem", [other, self], True)

    def __neg__(self):
        raise Exception("__neg__ is not implemented")

    def __lshift__(self, other):
        return self.get_expr_result("arith.shl", [self, other])

    def __rlshift__(self, other):
        return self.get_expr_result("arith.shl", [other, self])

    def __rshift__(self, other):
        return self.get_expr_result("arith.shr", [self, other], True)

    def __rrshift__(self, other):
        return self.get_expr_result("arith.shr", [other, self], True)

    def __and__(self, other):
        return self.get_expr_result("arith.and", [self, other])

    def __rand__(self, other):
        return self.get_expr_result("arith.and", [other, self])

    def __or__(self, other):
        return self.get_expr_result("arith.or", [self, other])

    def __ror__(self, other):
        return self.get_expr_result("arith.or", [other, self])

    def __xor__(self, other):
        return self.get_expr_result("arith.xor", [self, other])

    def __rxor__(self, other):
        return self.get_expr_result("arith.xor", [other, self])

    def __invert__(self):
        raise Exception("__invert__ is not implemented")

    def __lt__(self, other):
        return self.get_expr_result("arith.cmp", [self, other], True, "lt")

    def __le__(self, other):
        return self.get_expr_result("arith.cmp", [self, other], True, "le")

    def __eq__(self, other):
        return self.get_expr_result("arith.cmp", [self, other], False, "eq")

    def __ne__(self, other):
        return self.get_expr_result("arith.cmp", [self, other], False, "ne")

    def __gt__(self, other):
        return self.get_expr_result("arith.cmp", [self, other], True, "gt")

    def __ge__(self, other):
        return self.get_expr_result("arith.cmp", [self, other], True, "ge")


class Pattern():
    def __init__(self, name: str, benefit: int = 0):
        with GlobalInsertionPoint.get():
            self.pattern_op = pdl.PatternOp(benefit, name)
        self.name = name
        self.benefit = benefit
        self.is_pdl_pattern = True
        GlobalInsertionPoint.save(
            InsertionPoint.at_block_begin(self.pattern_op.body))

    def type(self, dtype):
        """
        Instantiate a PDL TypeOp.
        """
        with GlobalInsertionPoint.get():
            mlir_type = hcl_dtype_to_mlir(dtype) if dtype else None
            return TypeHandle(pdl.TypeOp(mlir_type).result, dtype)

    def attr(self, value, dtype):
        """
        Instantiate a PDL AttributeOp.
        """
        with GlobalInsertionPoint.get():
            attr = None
            if isinstance(value, str):
                attr = StringAttr.get(value)
            elif isinstance(value, bool):
                attr = BoolAttr.get()
            elif isinstance(value, (int, Int, UInt)):
                attr = IntegerAttr.get(hcl_dtype_to_mlir(dtype), value)
            elif isinstance(value, (float, Float)):
                attr = FloatAttr.get(hcl_dtype_to_mlir(dtype), value)
            return AttrHandle(pdl.AttributeOp(None, attr).attr, value, dtype)

    def value(self, dtype):
        """
        Instantiate a PDL ValueOp.
        """
        with GlobalInsertionPoint.get():
            type_ssa = pdl.TypeOp(hcl_dtype_to_mlir(dtype)).result
            value_ssa = pdl.OperandOp(type_ssa).val
        return ValueHandle(dtype, value_ssa)

    def op(self, name: str,
           operand_handles: Sequence[Union[ValueHandle, OpHandle]] = [],
           attr_handles: Mapping[str, AttrHandle] = {},
           type_handles: Sequence[TypeHandle] = []):
        """
        Instantiate a customized PDL OperationOp.
        """
        with GlobalInsertionPoint.get():
            def get_ssa(x):
                if isinstance(x, OpHandle):
                    return x.get_result().ssa
                else:
                    return x.ssa

            def get_attr_ssa(x):
                return (x[0], x[1].ssa)

            operands = list(map(get_ssa, operand_handles))
            attrs = dict(map(get_attr_ssa, attr_handles.items()))
            types = list(map(get_ssa, type_handles)) if len(
                type_handles) else [pdl.TypesOp()]
            op_ssa = pdl.OperationOp(name,  operands, attrs, types).op
            return OpHandle(op_ssa, list(map(lambda x: x.dtype, type_handles)))

    def start_transform(self, target_handle):
        """
        Instantiate a transform.with_pdl_patterns wrapper and a transform.sequence.
        """
        with GlobalInsertionPoint.get():
            rewrite = pdl.RewriteOp(
                target_handle.get_op_ssa(), "transform.dialect")
        pattern = rewrite.operation.parent
        with InsertionPoint(pattern):
            trans = transform.WithPDLPatternsOp()
        trans_root = trans.body.arguments[0]
        trans_ip = InsertionPoint.at_block_begin(trans.body)

        pattern.detach_from_parent()
        trans_ip.insert(pattern)
        with trans_ip:
            seq = transform.SequenceOp(trans_root)
        seq_ip = InsertionPoint.at_block_begin(seq.body)

        nameAttr = StringAttr(pattern.attributes["sym_name"])
        with seq_ip:
            target = transform.PDLMatchOp(trans_root, nameAttr.value)
            terminator = transform.YieldOp()

        GlobalInsertionPoint.save(InsertionPoint(terminator))
        self.pattern_op = trans
        self.is_pdl_pattern = False
        return OpHandle(target.result)

    def get_parent_loop(self, target_handle, num_loops=1):
        """
        Instantiate a Transform HCLGetParentLoopOp.
        """
        with GlobalInsertionPoint.get():
            num_loops_attr = IntegerAttr.get(
                IntegerType.get_signless(64), num_loops)
            loop = HCLGetParentLoopOp(
                target_handle.get_op_ssa(), num_loops_attr)
            return OpHandle(loop.results[0])

    def unroll(self, target_handle, factor=1):
        """
        Instantiate a Transform HCLUnrollOp.
        """
        with GlobalInsertionPoint.get():
            factor_attr = IntegerAttr.get(
                IntegerType.get_signless(64), factor)
            HCLUnrollOp(target_handle.get_op_ssa(), factor_attr)

    def split(self, target_handle, factor=1):
        """
        Instantiate a Transform HCLSplitOp, return the outer and inner loop
        handle after transform.
        """
        with GlobalInsertionPoint.get():
            factor_attr = IntegerAttr.get(
                IntegerType.get_signless(64), factor)
            split = HCLSplitOp(target_handle.get_op_ssa(), factor_attr)
            return OpHandle(split.results[0]), OpHandle(split.results[1])

    def pipeline(self, target_handle, initial_interval=1):
        """
        Instantiate a Transform HCLPipelineOp, return the pipelined loop.
        """
        with GlobalInsertionPoint.get():
            ii_attr = IntegerAttr.get(
                IntegerType.get_signless(64), initial_interval)
            pipeline = HCLPipelineOp(target_handle.get_op_ssa(), ii_attr)
            return OpHandle(pipeline.results[0])

    def start_rewrite(self, target_handle):
        """
        Instantiate a PDL RewriteOp and start to describe the rewriting rule.
        """
        with GlobalInsertionPoint.get():
            rewrite = pdl.RewriteOp(target_handle.get_op_ssa())
        rewrite.add_body()
        GlobalInsertionPoint.save(InsertionPoint.at_block_begin(rewrite.body))

    def replace(self, target_handle, repl_handle):
        """
        Instantiate a PDL ReplaceOp.
        """
        with GlobalInsertionPoint.get():
            pdl.ReplaceOp(target_handle.get_op_ssa(),
                          with_op=repl_handle.get_op_ssa())

    def erase(self, target_handle):
        """
        Instantiate a PDL EraseOp.
        """
        with GlobalInsertionPoint.get():
            pdl.EraseOp(target_handle.get_op_ssa())

    def require(self, predicate: Callable[..., ValueHandle], *values: ValueHandle):
        """
        Describe a require PDL pattern.
        """
        result = predicate(*values)
        with GlobalInsertionPoint.get():
            pdl.OperationOp("hcl.require", [result.ssa])

    def end_transform_or_rewrite(self):
        GlobalInsertionPoint.restore()  # Restore from sequence or rewrite region
        GlobalInsertionPoint.restore()  # Restore from pattern region
