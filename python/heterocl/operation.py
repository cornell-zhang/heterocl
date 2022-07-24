from .context import get_context, get_location, set_context
from .utils import get_dtype_str, hcl_dtype_to_mlir
from .tensor import Array, Tensor
from .schedule import Schedule, Stage
from .dsl import for_
from .context import UniqueName
from .types import Int, Type, UInt, Float, dtype_to_hcl
from . import config
from collections import OrderedDict
from typing import Callable, Sequence, Mapping, Union

from sympy import Integer

import hcl_mlir
import numpy as np
from hcl_mlir import GlobalInsertionPoint
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import scf, pdl, transform
from hcl_mlir.ir import *

from hcl_mlir.dialects._ods_common import _cext as _ods_cext
from hcl_mlir.dialects._ods_common import extend_opview_class as _ods_extend_opview_class, segmented_accessor as _ods_segmented_accessor, equally_sized_accessor as _ods_equally_sized_accessor, get_default_loc_context as _ods_get_default_loc_context, get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values

_ods_ir = _ods_cext.ir

try:
    from . import _transform_ops_ext as _ods_ext_module
except ImportError:
    _ods_ext_module = None


def init(init_dtype=Int(32), raise_assert_exception=True):
    """Initialize a HeteroCL environment with configurations."""
    config.init_dtype = init_dtype
    config.raise_assert_exception = raise_assert_exception


def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs."""
    if name is None:
        name = UniqueName.get("tensor")
    if (
        not dtype == None
        and not isinstance(dtype, (Type, str))
        and not hcl_mlir.is_hcl_mlir_type(dtype)
    ):
        raise RuntimeError("Type error")
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    if shape == ():
        shape = (1,)
    dtype = config.init_dtype if dtype == None else dtype
    tensor = Tensor(shape, dtype, name=name, impl="tensor")
    return tensor


def asarray(np_array, dtype=None):
    if isinstance(dtype, str):
        raise RuntimeError("Should provide hcl.Type. Got string")
    dtype = config.init_dtype if dtype == None else dtype
    return Array(np_array, dtype)


def scalar(init, name=None, dtype=None):
    """Syntactic sugar: single-value tensor
    - init: int, float, or expr
    """
    hcl_mlir.enable_build_inplace()
    if name is None:
        name = UniqueName.get("scalar")
    ret_tensor = placeholder((1,), name=name, dtype=dtype)
    index = hcl_mlir.ConstantOp("index", 0)
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype == None else dtype
    dtype = hcl_dtype_to_mlir(dtype)
    if isinstance(init, int) or isinstance(init, float):
        init = hcl_mlir.ConstantOp(dtype, init)
    elif isinstance(init, Tensor):
        init = init.op
    ret_tensor.init()  # init hcl_mlir type
    hcl_mlir.StoreOp(init, ret_tensor.op, [index])
    return ret_tensor


def reduce_axis(lower, upper, name=None):
    """Create a reduction axis for reduction operations."""
    if name is None:
        name = UniqueName.get("reduction_axis")
    return hcl_mlir.ReduceVar(None, bound=(lower, upper), name=name)


def cast(dtype, expr):
    if isinstance(expr, Tensor):
        raise RuntimeError("Tensor is not supported in hcl.cast. " +
                           "If you are try to cast a hcl.scalar, please use hcl.cast(scalar.v)")
    return hcl_mlir.CastOp(expr, hcl_dtype_to_mlir(dtype))


def const_tensor(values, name=None, dtype=None):
    """Create a constant tensor"""
    if name is None:
        name = UniqueName.get("tensor")
    dtype = config.init_dtype if dtype == None else dtype
    cst = hcl_mlir.ConstantOp(hcl_dtype_to_mlir(dtype), values, name)
    return cst.tensor


def copy(values, name=None, dtype=None):
    """A syntactic sugar for copying an existing tensor."""
    if name is None:
        name = UniqueName.get("tensor")
    dtype = config.init_dtype if dtype == None else dtype
    cst = hcl_mlir.ConstantOp(hcl_dtype_to_mlir(dtype), values, name)
    return cst.tensor


def select(cond, true_val, false_val):
    return hcl_mlir.SelectOp(cond, true_val, false_val)


def sum(data, axis=None, dtype=None, name=""):
    dtype = config.init_dtype if dtype == None else dtype
    return hcl_mlir.SumOp(data, axis, get_dtype_str(dtype))


def max(data, axis=None, dtype=None, name=""):
    dtype = config.init_dtype if dtype == None else dtype
    return hcl_mlir.MaxOp(data, axis, get_dtype_str(dtype))


def min(data, axis=None, dtype=None, name=""):
    dtype = config.init_dtype if dtype == None else dtype
    return hcl_mlir.MinOp(data, axis, get_dtype_str(dtype))


def reduce(data, init_val, reduce_op, axis=None, dtype=None, name=""):
    return hcl_mlir.ReduceOp(data, axis, get_dtype_str(dtype), prefix=name, init_val=init_val, reduce_op={"si": reduce_op})


def pack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Pack a tensor with smaller bitwidth to a tensor with larger bitwidth."""
    if factor is None and dtype is not None:
        factor = dtype.bits // tensor.dtype.bits
    if factor is None or not isinstance(factor, int):
        raise RuntimeError("Should specify factor")
    if not isinstance(tensor.dtype, (Int, UInt)):
        raise RuntimeError("Only support integer packing")
    if name == None or name == "":
        name = UniqueName.get("tensor")
    bitwidth = tensor.dtype.bits
    if isinstance(tensor.dtype, Int):
        new_type = Int(bitwidth * factor)
    else:
        new_type = UInt(bitwidth * factor)
    new_shape = [
        size // factor if i == axis else size for i, size in enumerate(tensor.shape)
    ]

    def assign_val(*indices):
        result = scalar(0, name="packed_" + name, dtype=new_type)
        with for_(0, factor) as i:
            new_indices = [
                (index * factor + i) if j == axis else index
                for j, index in enumerate(indices)
            ]
            val = tensor[tuple(new_indices)]
            result[0][bitwidth * i: bitwidth *
                      (i + 1)] = tensor[tuple(new_indices)]
        return result[0]

    return compute(tuple(new_shape), assign_val, name, new_type)


def unpack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Unpack a tensor with larger bitwidth to a tensor with smaller bitwidth."""
    if factor is None and dtype is not None:
        factor = tensor.dtype.bits // dtype.bits
    if factor is None or not isinstance(factor, int):
        raise RuntimeError("Should specify factor")
    if not isinstance(tensor.dtype, (Int, UInt)):
        raise RuntimeError("Only support integer packing")
    if name == None or name == "":
        name = UniqueName.get("tensor")
    bitwidth = tensor.dtype.bits
    if isinstance(tensor.dtype, Int):
        new_type = Int(bitwidth // factor)
    else:
        new_type = UInt(bitwidth // factor)
    new_shape = [
        size * factor if i == axis else size for i, size in enumerate(tensor.shape)
    ]

    def assign_val(*indices):
        result = scalar(0, name="unpacked_" + name, dtype=new_type)
        new_indices = [
            (index // factor) if j == axis else index for j, index in enumerate(indices)
        ]
        lower = (indices[axis] % factor) * (bitwidth // factor)
        upper = lower + bitwidth // factor
        val = tensor[tuple(new_indices)][lower:upper]
        if val.dtype.width != bitwidth // factor:
            # cast val to the same width as bitwidth // factor
            val = hcl_mlir.CastOp(val, hcl_dtype_to_mlir(new_type))
        result[0][0: bitwidth // factor] = val
        return result[0]

    return compute(tuple(new_shape), assign_val, name, new_type)


def compute(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
    """
    This function call does not directly build IR, it only creates a node
    """
    # check API correctness
    if not isinstance(shape, tuple):
        raise RuntimeError("The shape of compute API must be a tuple")
    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    if name is None:
        name = UniqueName.get("tensor")
    if not dtype == None and not isinstance(dtype, (Type, str)):
        raise RuntimeError("Type error")
    dtype = config.init_dtype if dtype == None else dtype
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    ret_tensor = Tensor(shape, dtype, name=name,
                        fcompute=fcompute, impl="compute")
    for tensor in ret_tensor.op.inputs:
        tensor.add_use(ret_tensor)
    return ret_tensor


def update(tensor: Tensor, fcompute, name=None):
    """
    fcompute: function, callable
    name: str
    """
    # Check tensor type
    if not isinstance(tensor, Tensor):
        raise RuntimeError(
            "Unexpected argument type of the "
            + "first argument: {}, update API expects tensor as input.".format(
                type(tensor)
            )
        )
    if name is None:
        name = tensor.name + "_updated"
    new_tensor = Tensor(
        tensor.shape,
        tensor.dtype,
        fcompute=fcompute,
        name=name,
        impl="compute",
        output=tensor if isinstance(
            tensor.op, hcl_mlir.TensorOp) else tensor.op.output,
    )
    tensor.add_use(new_tensor)
    Schedule._CurrentSchedule.DataflowGraph.add_edge(
        tensor, new_tensor, stateful=True)
    if Schedule._TopFunction != None:
        stage = Stage(name)
        with get_context() as ctx, get_location() as loc:
            stage.stage_handle = hcl_d.CreateOpHandleOp(
                StringAttr.get(name), ip=hcl_mlir.GlobalInsertionPoint.get()
            )
        Schedule._CurrentStage.append(stage)
        Schedule._TopFunction.__setattr__(name, stage)
        stage.__setattr__(tensor.name, new_tensor)


def mutate(domain, fcompute, name=None):
    """
    For now, assume no return value
    """
    # check API correctness
    if not isinstance(domain, tuple):
        raise RuntimeError("The domain of mutate API must be a tuple")
    if name is None:
        name = UniqueName.get("tensor")
    ret_tensor = Tensor(domain, None, name=name,
                        fcompute=fcompute, impl="compute")
    return ret_tensor


def bitcast(tensor, dst_dtype, name=None):
    """Bitcast a HeteroCL tensor or expression to the destination data type of the same bitwidth.
    This API **bitcast** the input tensor from its own data type (source dtype)
    to the destination data type (dst_dtype). The destination data type must have
    the same bitwidth with the source datatype.
    """
    if not isinstance(tensor, Tensor) and not isinstance(tensor, hcl_mlir.ExprOp):
        raise RuntimeError("bitcast input must be HeteroCL Tensor or ExprOp.")

    # check type
    if not isinstance(dst_dtype, Type):
        raise RuntimeError("dst_dtype should be HeteroCL data type.")

    # check bitwidth
    if isinstance(tensor, Tensor):
        src_bitwidth = tensor.dtype.bits
    else:  # ExprOp
        src_bitwidth = hcl_mlir.get_bitwidth(tensor.dtype)
    dst_bitwidth = dst_dtype.bits
    if src_bitwidth != dst_bitwidth:
        raise RuntimeError(
            "Destination datatype bitwidth does not match source bitwidth:"
            + f"source bitwidth: {src_bitwidth} , destination bitwidth {dst_bitwidth}."
        )

    # set up name, shape, and fcompute
    dst_dtype_str = get_dtype_str(dst_dtype)
    if isinstance(tensor, Tensor):
        name = tensor.name + "_" + dst_dtype_str if name is None else name
        shape = tensor.shape
        fcompute = lambda *args: hcl_mlir.BitCastOp(
            hcl_dtype_to_mlir(dst_dtype), tensor[args]
        )
        return compute(shape, fcompute, name=name, dtype=dst_dtype)
    else:
        bitcast = hcl_mlir.BitCastOp(hcl_dtype_to_mlir(dst_dtype), tensor)
        builder = hcl_mlir.ASTVisitor(mode="build")
        builder.visit(bitcast)
        # return an expression
        return bitcast


def cast_np(np_array, dtype):
    """
    Cast a numpy array to a HeteroCL data type.
    """
    if not isinstance(np_array, np.ndarray):
        raise RuntimeError("cast_np input must be numpy array.")
    if not isinstance(dtype, Type):
        raise RuntimeError("dtype should be HeteroCL data type.")
    return asarray(np_array, dtype).asnumpy()


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

                type_ssa = pdl.TypeOp(type).result
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
            type_ssa = pdl.TypeOp(hcl_dtype_to_mlir(result_type)).result
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


def pdl_pattern(pattern_descriptor: Callable, name: str, benefit=0):
    """
    Instantiate a PDL pattern from a pattern descriptor.
    """
    set_context()
    module = Module.create(get_location())
    with get_context(), get_location():
        with InsertionPoint.at_block_begin(module.body):
            pattern = pdl.PatternOp(benefit, name)
        ip = InsertionPoint.at_block_begin(pattern.body)
        GlobalInsertionPoint.save(ip)
        pattern_descriptor()
    print(module)


def pdl_type(dtype):
    """
    Instantiate a PDL TypeOp.
    """
    with GlobalInsertionPoint.get():
        mlir_type = hcl_dtype_to_mlir(dtype) if dtype else None
        return TypeHandle(pdl.TypeOp(mlir_type).result, dtype)


def pdl_attr(value, dtype):
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


def pdl_value(dtype):
    """
    Instantiate a PDL ValueOp.
    """
    with GlobalInsertionPoint.get():
        type_ssa = pdl.TypeOp(hcl_dtype_to_mlir(dtype)).result
        value_ssa = pdl.OperandOp(type_ssa).val
    return ValueHandle(dtype, value_ssa)


def pdl_op(name: str,
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


def pdl_transform(target_handle):
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
    ip = InsertionPoint(terminator)
    GlobalInsertionPoint.save(ip)
    return OpHandle(target.result)


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


def get_parent_loop(target_handle, num_loops=1):
    """
    Instantiate a Transform HCLGetParentLoopOp.
    """
    with GlobalInsertionPoint.get():
        num_loops_attr = IntegerAttr.get(
            IntegerType.get_signless(64), num_loops)
        loop = HCLGetParentLoopOp(target_handle.get_op_ssa(), num_loops_attr)
        return OpHandle(loop.results[0])


@_ods_extend_opview_class(_ods_ext_module)
class HCLLoopUnrollOp(OpView):
    OPERATION_NAME = "transform.hcl.loop_unroll"

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


def loop_unroll(target_handle, factor=1):
    """
    Instantiate a Transform HCLLoopUnrollOp.
    """
    with GlobalInsertionPoint.get():
        factor_attr = IntegerAttr.get(
            IntegerType.get_signless(64), factor)
        HCLLoopUnrollOp(target_handle.get_op_ssa(), factor_attr)


def pdl_rewrite(target_handle):
    """
    Instantiate a PDL RewriteOp and start to describe the rewriting rule.
    """
    with GlobalInsertionPoint.get():
        rewrite = pdl.RewriteOp(target_handle.get_op_ssa())
    rewrite.add_body()
    ip = InsertionPoint.at_block_begin(rewrite.body)
    GlobalInsertionPoint.save(ip)


def pdl_replace(target_handle, repl_handle):
    """
    Instantiate a PDL ReplaceOp.
    """
    with GlobalInsertionPoint.get():
        pdl.ReplaceOp(target_handle.get_op_ssa(),
                      with_op=repl_handle.get_op_ssa())


def pdl_erase(target_handle):
    """
    Instantiate a PDL EraseOp.
    """
    with GlobalInsertionPoint.get():
        pdl.EraseOp(target_handle.get_op_ssa())


def pdl_require(predicate: Callable[..., ValueHandle], *values: ValueHandle):
    """
    Describe a require PDL pattern.
    """
    result = predicate(*values)
    with GlobalInsertionPoint.get():
        pdl.OperationOp("hcl.require", [result.ssa])
