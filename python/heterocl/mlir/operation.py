from collections import OrderedDict

import hcl_mlir
from hcl_mlir.ir import *

from .. import config, types
from ..types import Type, Int, UInt, dtype_to_hcl
from .context import UniqueName
from .schedule import Schedule
from .tensor import Array, Tensor
from .utils import get_dtype_str, hcl_dtype_to_mlir
from .dsl import for_


def init(init_dtype=types.Int(32), raise_assert_exception=True):
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
    dtype = config.init_dtype if dtype == None else dtype
    dtype = hcl_dtype_to_mlir(dtype)
    if isinstance(init, int) or isinstance(init, float):
        init = hcl_mlir.ConstantOp(dtype, init)
    ret_tensor.init()  # init hcl_mlir type
    hcl_mlir.StoreOp(init, ret_tensor.op, [index])
    return ret_tensor


def reduce_axis(lower, upper, name=None):
    """Create a reduction axis for reduction operations."""
    return hcl_mlir.ReduceVar(None, bound=(lower, upper), name=name)


def cast(dtype, expr):
    return hcl_mlir.CastOp(expr, hcl_dtype_to_mlir(dtype))


def const_tensor(values, name=None, dtype=None):
    """Create a constant tensor"""
    dtype = config.init_dtype if dtype == None else dtype
    cst = hcl_mlir.ConstantOp(hcl_dtype_to_mlir(dtype), values)
    return cst.tensor


def copy(values, name=None, dtype=None):
    """A syntactic sugar for copying an existing tensor."""
    dtype = config.init_dtype if dtype == None else dtype
    cst = hcl_mlir.ConstantOp(hcl_dtype_to_mlir(dtype), values)
    return cst.tensor


def select(cond, true_val, false_val):
    return hcl_mlir.SelectOp(cond, true_val, false_val)


def sum(data, axis=None, dtype=None, name=""):
    return hcl_mlir.SumOp(data, axis, get_dtype_str(dtype))


def max(data, axis=None, dtype=None, name=""):
    return hcl_mlir.MaxOp(data, axis, get_dtype_str(dtype))


def min(data, axis=None, dtype=None, name=""):
    return hcl_mlir.MinOp(data, axis, get_dtype_str(dtype))


def pack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Pack a tensor with smaller bitwidth to a tensor with larger bitwidth."""
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
                index if j == axis else (index * factor + i)
                for j, index in enumerate(indices)
            ]
            result[0][bitwidth * i: bitwidth *
                      (i + 1)] = tensor[tuple(new_indices)]
        return result[0]

    return compute(tuple(new_shape), assign_val, name, new_type)


def unpack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Unpack a tensor with larger bitwidth to a tensor with smaller bitwidth."""
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
        size // factor if i == axis else size for i, size in enumerate(tensor.shape)
    ]

    def assign_val(*indices):
        result = scalar(0, name="unpacked_" + name, dtype=new_type)
        new_indices = [
            index if j == axis else (index // factor) for j, index in enumerate(indices)
        ]
        lower = (indices[axis] % factor) * (bitwidth // factor)
        upper = lower + bitwidth // factor
        result[0][0: bitwidth //
                  factor] = tensor[tuple(new_indices)][lower:upper]
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
        output=tensor,
    )
    tensor.add_use(new_tensor)
    Schedule._CurrentSchedule.DataflowGraph.add_edges(tensor, new_tensor)


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
