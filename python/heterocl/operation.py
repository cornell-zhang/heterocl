import sys
import inspect
import gc
from collections import OrderedDict

import hcl_mlir
import numpy as np
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import *
from hcl_mlir.exceptions import *

from . import config
from .types import Int, Type, UInt, Struct, dtype_to_hcl
from .context import UniqueName
from .dsl import for_
from .schedule import Schedule, Stage
from .tensor import Array, Tensor
from .utils import get_dtype_str, hcl_dtype_to_mlir, get_func_obj
from .context import get_context, get_location


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
        and not isinstance(name, str)
    ):
        raise APIError("Input type error, got dtype={}, name={}".format(dtype, name))
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    if shape == ():
        shape = (1,)
    dtype = config.init_dtype if dtype == None else dtype
    tensor = Tensor(shape, dtype, name=name, impl="tensor")
    return tensor


def asarray(np_array, dtype=None):
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype == None else dtype
    return Array(np_array, dtype)


def scalar(init, name=None, dtype=None):
    """Syntactic sugar: single-value tensor
    - init: int, float, expr, or tuple
    """
    hcl_mlir.enable_build_inplace()
    if name is None:
        name = UniqueName.get("scalar")

    ret_tensor = placeholder((1,), name=name, dtype=dtype)
    index = hcl_mlir.ConstantOp("index", 0)
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype == None else dtype # dtype is HeteroCL type
    mlir_type = hcl_dtype_to_mlir(dtype)
    if isinstance(dtype, Struct):
        if isinstance(init, tuple):
            init = hcl_mlir.StructConstructOp(list(init))
        elif isinstance(init, int):
            vals = list()
            for ftype in dtype.dtype_dict.values():
                mask = (1 << (ftype.bits+1)) - 1
                val = init & mask
                init = init >> ftype.bits
                vals.append(hcl_mlir.ConstantOp(hcl_dtype_to_mlir(ftype), val))
            init = hcl_mlir.StructConstructOp(vals)
        # TODO(Niansong): support init as a single expr
    elif isinstance(init, int) or isinstance(init, float):
        init = hcl_mlir.ConstantOp(mlir_type, init)
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
        raise APIError("Tensor is not supported in hcl.cast. " +
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
        raise APIError("Should specify factor")
    if not isinstance(tensor.dtype, (Int, UInt)):
        raise APIError("Only support integer packing")
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
        raise APIError("Should specify factor")
    if not isinstance(tensor.dtype, (Int, UInt)):
        raise APIError("Only support integer packing")
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
        raise APIError("The shape of compute API must be a tuple")
    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    if name is None:
        name = UniqueName.get("tensor")
    if not dtype == None and not isinstance(dtype, (Type, str)):
        raise APIError("Type error")
    dtype = config.init_dtype if dtype == None else dtype
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)

    ret_tensor = Tensor(shape, dtype, name=name,
                        fcompute=fcompute, impl="compute")
    for tensor in ret_tensor.op.inputs:
        tensor.add_use(ret_tensor)


    # Check the caller function
    caller_func_name = inspect.stack()[1].function
    if Schedule._TopFunction is None:
        called_from_top = False
    else:
        called_from_top = caller_func_name == Schedule._TopFunction.__name__
    caller_func = get_func_obj(caller_func_name)
    if not called_from_top:
        # Caller function up one level
        caller_parent_func_name = inspect.stack()[2].function
        caller_parent_func = get_func_obj(caller_parent_func_name)
        # If haven't already, attach the caller function
        # to its parent function as an attribute
        if not hasattr(caller_parent_func, caller_func_name):
            caller_parent_func.__setattr__(caller_func_name, caller_func)
    stage = ret_tensor.op.stage
    if Schedule._TopFunction != None:
        caller_func.__setattr__(stage.name, stage)
        # Set up a list of stages for the caller function
        if not hasattr(caller_func, "_stages"):
            caller_func.__setattr__("_stages", [stage])
        else:
            caller_func._stages.append(stage)    
    
    return ret_tensor

def update(tensor: Tensor, fcompute, name=None):
    """
    fcompute: function, callable
    name: str
    """
    # Check the caller function
    caller_func_name = inspect.stack()[1].function
    if Schedule._TopFunction is None:
        called_from_top = False
    else:
        called_from_top = caller_func_name == Schedule._TopFunction.__name__
    caller_func = get_func_obj(caller_func_name)
    if not called_from_top:
        # Caller function up one level
        caller_parent_func_name = inspect.stack()[2].function
        caller_parent_func = get_func_obj(caller_parent_func_name)
        # If haven't already, attach the caller function
        # to its parent function as an attribute
        if not hasattr(caller_parent_func, caller_func_name):
            caller_parent_func.__setattr__(caller_func_name, caller_func)

    # Check tensor type
    if not isinstance(tensor, Tensor):
        raise APIError(
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
        # Attach the stage to the caller function as an attribute
        caller_func.__setattr__(name, stage)
        stage.__setattr__(tensor.name, new_tensor)
        # Set up a list of stages for the caller function
        if not hasattr(caller_func, "_stages"):
            caller_func.__setattr__("_stages", [stage])
        else:
            caller_func._stages.append(stage)


def mutate(domain, fcompute, name=None):
    """
    For now, assume no return value
    """
    # check API correctness
    if not isinstance(domain, tuple):
        raise APIError("The domain of mutate API must be a tuple")
    if name is None:
        name = UniqueName.get("tensor")
    ret_tensor = Tensor(domain, None, name=name,
                        fcompute=fcompute, impl="compute")

    # Check the caller function
    caller_func_name = inspect.stack()[1].function
    if Schedule._TopFunction is None:
        called_from_top = False
    else:
        called_from_top = caller_func_name == Schedule._TopFunction.__name__
    caller_func = get_func_obj(caller_func_name)
    if not called_from_top:
        # Caller function up one level
        caller_parent_func_name = inspect.stack()[2].function
        caller_parent_func = get_func_obj(caller_parent_func_name)
        # If haven't already, attach the caller function
        # to its parent function as an attribute
        if not hasattr(caller_parent_func, caller_func_name):
            caller_parent_func.__setattr__(caller_func_name, caller_func)
    stage = ret_tensor.op.stage
    if Schedule._TopFunction != None:
        caller_func.__setattr__(stage.name, stage)
        # Set up a list of stages for the caller function
        if not hasattr(caller_func, "_stages"):
            caller_func.__setattr__("_stages", [stage])
        else:
            caller_func._stages.append(stage)    
    return ret_tensor


def bitcast(tensor, dst_dtype, name=None):
    """Bitcast a HeteroCL tensor or expression to the destination data type of the same bitwidth.
    This API **bitcast** the input tensor from its own data type (source dtype)
    to the destination data type (dst_dtype). The destination data type must have
    the same bitwidth with the source datatype.
    """
    if not isinstance(tensor, Tensor) and not isinstance(tensor, hcl_mlir.ExprOp):
        raise APIError("bitcast input must be HeteroCL Tensor or ExprOp.")

    # check type
    if not isinstance(dst_dtype, Type):
        raise APIError("dst_dtype should be HeteroCL data type.")

    # check bitwidth
    if isinstance(tensor, Tensor):
        src_bitwidth = tensor.dtype.bits
    else:  # ExprOp
        src_bitwidth = hcl_mlir.get_bitwidth(tensor.dtype)
    dst_bitwidth = dst_dtype.bits
    if src_bitwidth != dst_bitwidth:
        raise APIError(
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
        raise APIError("cast_np input must be numpy array.")
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    elif not isinstance(dtype, Type):
        raise APIError("dtype should be HeteroCL data type.")
    return asarray(np_array, dtype).asnumpy()
