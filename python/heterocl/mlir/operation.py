from collections import OrderedDict

import hcl_mlir
from hcl_mlir.dialects import memref
from hcl_mlir.ir import *

from .. import config, types
from ..types import Type, dtype_to_hcl
from .context import UniqueName
from .schedule import Schedule
from .tensor import Array, Tensor
from .utils import get_dtype_str, hcl_dtype_to_mlir


def init(init_dtype=types.Int(32), raise_assert_exception=True):
    """Initialize a HeteroCL environment with configurations.
    """
    config.init_dtype = init_dtype
    config.raise_assert_exception = raise_assert_exception


def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs.
    """
    if name is None:
        name = UniqueName.get("tensor")
    if not dtype == None and not isinstance(dtype, (Type, str)):
        raise RuntimeError("Type error")
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype == None else dtype
    tensor = Tensor(shape, dtype, name=name, impl="tensor")
    return tensor


def asarray(np_array, dtype=None):
    dtype = config.init_dtype if dtype == None else dtype
    return Array(np_array, dtype)


def scalar(init, name=None, dtype=None):
    """Syntactic sugar: single-value tensor 
    - init: int, float, or expr
    """
    if name is None:
        name = UniqueName.get("scalar")
    ret_tensor = placeholder((1,), name=name, dtype=dtype)
    index = hcl_mlir.ConstantOp("index", 0)
    if not hcl_mlir.is_hcl_mlir_type(dtype):
        dtype = get_dtype_str(dtype)
    if isinstance(init, int) or isinstance(init, float):
        init = hcl_mlir.ConstantOp(dtype, init)
    hcl_mlir.StoreOp(init, ret_tensor.op, [index])
    return ret_tensor


def reduce_axis(lower, upper, name=None):
    """Create a reduction axis for reduction operations.
    """
    return hcl_mlir.ReduceVar(None, bound=(lower, upper), name=name)


def cast(dtype, expr):
    return hcl_mlir.CastOp(expr, hcl_dtype_to_mlir(dtype))


def select(cond, true_val, false_val):
    return hcl_mlir.SelectOp(cond, true_val, false_val)


def any(*args):
    """Create a new experssion of the union of all conditions in the arguments
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    ret = hcl_mlir.OrOp(args[0], args[1])
    for i in range(2, len(args)):
        ret = hcl_mlir.OrOp(ret, args[i])
    return ret


def all(*args):
    """Create a new experssion of the intersection of all conditions in the
      arguments
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    ret = hcl_mlir.AndOp(args[0], args[1])
    for i in range(2, len(args)):
        ret = hcl_mlir.AndOp(ret, args[i])
    return ret


def sum(data, axis=None, dtype=None, name=""):
    return hcl_mlir.SumOp(data, axis, get_dtype_str(dtype))


def max(data, axis=None, dtype=None, name=""):
    return hcl_mlir.MaxOp(data, axis, get_dtype_str(dtype))


def min(data, axis=None, dtype=None, name=""):
    return hcl_mlir.MinOp(data, axis, get_dtype_str(dtype))


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
        raise RuntimeError("Unexpected argument type of the " +
                           "first argument: {}, update API expects tensor as input.".format(type(tensor)))
    if name is None:
        name = tensor.name + "_updated"
    new_tensor = Tensor(tensor.shape, tensor.dtype, fcompute=fcompute,
                        name=name, impl="compute", output=tensor)
    tensor.add_use(new_tensor)
    Schedule._CurrentSchedule.DataflowGraph.add_edges(tensor, new_tensor)


def mutate(domain, fcompute, name):
    """
    For now, assume no return value
    """
    # check API correctness
    if not isinstance(domain, tuple):
        raise RuntimeError("The domain of mutate API must be a tuple")
    if name is None:
        name = UniqueName.get("stage")
    compute_body(domain, fcompute, None, name)
    return


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
        raise RuntimeError("Destination datatype bitwidth does not match source bitwidth:" +
                           f"source bitwidth: {src_bitwidth} , destination bitwidth {dst_bitwidth}.")

    # set up name, shape, and fcompute
    dst_dtype_str = get_dtype_str(dst_dtype)
    if isinstance(tensor, Tensor):
        name = tensor.name + '_' + dst_dtype_str if name is None else name
        shape = tensor.shape
        fcompute = lambda *args: hcl_mlir.BitCastOp(
            hcl_dtype_to_mlir(dst_dtype), tensor[args])
        return compute(shape, fcompute, name=name, dtype=dst_dtype)
    else:
        bitcast = hcl_mlir.BitCastOp(hcl_dtype_to_mlir(dst_dtype), tensor)
        builder = hcl_mlir.ASTVisitor(mode="build")
        builder.visit(bitcast)
        # return an expression
        return bitcast
