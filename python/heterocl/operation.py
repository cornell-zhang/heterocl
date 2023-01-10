import inspect
from collections import OrderedDict
from typing import List

import hcl_mlir
import numpy as np
import re
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import *
from hcl_mlir.exceptions import *

from . import config
from .types import Int, Type, UInt, Struct, dtype_to_hcl
from .context import NestedStageLevel, UniqueName
from .dsl import for_
from .schedule import Schedule, Stage
from .tensor import Array, Tensor
from .utils import *
from .context import get_context, get_location
from .ast import ast


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
    filename, lineno = get_src_loc(frame=1)
    alloc = ast.AllocOp(name, shape, dtype, ast.Location(filename, lineno))
    return alloc


def asarray(np_array, dtype=None):
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype == None else dtype
    return Array(np_array, dtype)


def scalar(init, name=None, dtype=None):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    if name is None:
        name = UniqueName.get("scalar")
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype == None else dtype # dtype is HeteroCL type
    if isinstance(dtype, Struct) and isinstance(init, int):
        vals = list()
        for ftype in dtype.dtype_dict.values():
            mask = (1 << (ftype.bits+1)) - 1
            val = init & mask
            init = init >> ftype.bits
            vals.append(ast.ConstantOp(val, ftype, loc))
        init = tuple(vals)

    # Generate a ComputeOp
    op = compute_body(name, (1,), lambda x : init, dtype, loc, None)
    return op.tensor


def reduce_axis(lower, upper, name=None):
    """Create a reduction axis for reduction operations."""
    if name is None:
        name = UniqueName.get("reduction_axis")
    # return hcl_mlir.ReduceVar(None, bound=(lower, upper), name=name)
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.ReduceVar(name, parent_loop=None, loc=loc, bound=(lower, upper))


def cast(dtype, expr):
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    if isinstance(expr, Tensor):
        raise APIError("Tensor is not supported in hcl.cast. " +
                        "If you are try to cast a hcl.scalar, please use hcl.cast(scalar.v)")
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.CastOp(expr, dtype, loc)


def const_tensor(values, name=None, dtype=None):
    """Create a constant tensor"""
    if name is None:
        name = UniqueName.get("tensor")
    dtype = config.init_dtype if dtype == None else dtype
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    # convert values to numpy array and handle overflow
    values = np.array(values)
    shape = values.shape
    values = make_const_tensor(values, dtype)
    realtype = Int(64) if isinstance(dtype, (Int, UInt)) else dtype
    cst_op = ast.ConstantTensorOp(values, name, shape, realtype, loc)
    region = ast.scope.get()
    region.append(cst_op)
    if isinstance(dtype, (Int, UInt)):
        return compute(shape, lambda *args : cast(dtype, cst_op.tensor[args]), 
            name=name+"_cast", dtype=dtype)
    return cst_op.tensor


def copy(values, name=None, dtype=None):
    """A syntactic sugar for copying an existing tensor."""
    if name is None:
        name = UniqueName.get("tensor")
    dtype = config.init_dtype if dtype == None else dtype
    return const_tensor(values, name, dtype)


def select(cond, true_val, false_val):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    true_val = ast.immediate_to_constant(true_val, loc)
    false_val = ast.immediate_to_constant(false_val, loc)
    cond = ast.immediate_to_constant(cond, loc)
    return ast.SelectOp(cond, true_val, false_val, loc)


def sum(expr, axis=None, dtype=None, name=None):
    if name is None:
        name = UniqueName.get("op")
    if axis is None:
        raise HCLNotImplementedError("sum with axis=None is not supported")
    if isinstance(axis, tuple):
        axis = list(axis)
    elif isinstance(axis, ast.ReduceVar):
        axis = [axis]
    elif not isinstance(axis, list):
        raise APIError("axis must be a list of reduction axis")
    dtype = config.init_dtype if dtype == None else dtype
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.SumOp(name, expr, axis, dtype, loc)


def max(data, axis=None, dtype=None, name=""):
    raise HCLNotImplementedError("max is not implemented yet")
    dtype = config.init_dtype if dtype == None else dtype
    return hcl_mlir.MaxOp(data, axis, get_dtype_str(dtype))


def min(data, axis=None, dtype=None, name=""):
    raise HCLNotImplementedError("min is not implemented yet")
    dtype = config.init_dtype if dtype == None else dtype
    return hcl_mlir.MinOp(data, axis, get_dtype_str(dtype))


def reduce(data, init_val, reduce_op, axis=None, dtype=None, name=""):
    raise HCLNotImplementedError("reduce is not implemented yet")
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
        new_indices = [
            (index // factor) if j == axis else index for j, index in enumerate(indices)
        ]
        lower = (indices[axis] % factor) * (bitwidth // factor)
        upper = lower + bitwidth // factor
        val = tensor[tuple(new_indices)][lower:upper]
        return val

    return compute(tuple(new_shape), assign_val, name, new_type)
    

def compute_body(name, shape, fcompute, dtype, loc, tensor):
    """ Create an ast.ComputeOp and its body operations

    Parameters
    ----------
    name : str
        The name of the compute op
    shape: tuple
        The shape of the compute op
    fcompute: function
        The compute function
    dtype: hcl.dtype
        The data type of the compute op
    tensor: ast.AllocOp, None, or "no_alloc"
        The tensor to store the result of the compute op
        - ast.AllocOp: hcl.update
        - None: hcl.compute, ComputeOp will allocate new tensor
        - "no_alloc": hcl.mutate, no tensor will be allocated

    Returns
    -------
    ast.ComputeOp
        The compute op
    """
    # Generate a ComputeOp
    compute_op = ast.ComputeOp(name, shape, fcompute, dtype, loc, tensor)
    region = ast.scope.get()
    region.append(compute_op)
    # Analyze input tensors, and update uses for those tensors
    closure_var = inspect.getclosurevars(fcompute).nonlocals
    input_tensors = [v for v in closure_var.values() if isinstance(v, ast.AllocOp)]
    reduce_vars = [v for v in closure_var.values() if isinstance(v, ast.ReduceVar)]
    compute_op.input_tensors.extend(input_tensors)

    # Build AST for fcompute body
    argspec = inspect.getfullargspec(fcompute)
    axis_names = argspec.args
    if len(axis_names) == 0:
        # this is the case where fcompute is lambda *args: ...
        axis_names = ["i" + str(i) for i in range(len(shape))]
    if len(axis_names) != len(shape):
        raise APIError(f"fcompute's number of axis does not match output tensor shape: {axis_names} vs {shape}")
    iter_vars = [ast.IterVar(name, None, loc) for name in axis_names]
    # attach iter_vars to the compute op
    # iter_var's parent_loop will be set in ir.ir_builder.build_compute
    compute_op.iter_vars.extend(iter_vars)
    compute_op.reduce_vars.extend(reduce_vars)
    ast.scope.push(compute_op.body)
    if tensor is None:
        # hcl.compute
        res_expr = fcompute(*iter_vars)
        res_expr = ast.immediate_to_constant(res_expr, loc)
        if isinstance(res_expr, tuple) and isinstance(dtype, Struct):
            res_expr = ast.StructConstructOp(list(res_expr), dtype, loc)
        if res_expr is None:
            if len(compute_op.body) > 0 and isinstance(compute_op.body[-1], ast.ReturnOp):
                res_expr = ast.immediate_to_constant(compute_op.body[-1].expr, loc)
                compute_op.body.pop()
        store_op = ast.StoreOp(compute_op.tensor, compute_op.iter_vars, res_expr, loc)
        compute_op.body.append(store_op)
        ast.scope.pop()
    elif isinstance(tensor, ast.AllocOp):
        # hcl.update
        res_expr = fcompute(*iter_vars)
        res_expr = ast.immediate_to_constant(res_expr, loc)
        if isinstance(res_expr, tuple) and isinstance(dtype, Struct):
            res_expr = ast.StructConstructOp(list(res_expr), dtype, loc)
        if res_expr is None:
            if len(compute_op.body) > 0 and isinstance(compute_op.body[-1], ast.ReturnOp):
                res_expr = ast.immediate_to_constant(compute_op.body[-1].expr, loc)
                compute_op.body.pop()
        store_op = ast.StoreOp(tensor, iter_vars, res_expr, loc)
        compute_op.body.append(store_op)
        ast.scope.pop()
    elif isinstance(tensor, str) and tensor == "no_alloc":
        # hcl.mutate
        res_expr = fcompute(*iter_vars)
        if res_expr is not None:
            raise APIError("hcl.mutate does not support return value")
        ast.scope.pop()
    else:
        raise APIError("Invalid tensor type")

    return compute_op

def compute(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
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

    # Generate a ComputeOp
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    compute_op = compute_body(name, shape, fcompute, dtype, loc, None)
    return compute_op.tensor


def update(tensor, fcompute, name=None):
    if not isinstance(tensor, ast.AllocOp):
        raise APIError("The input of update API must be an allocated tensor")
    if name is None:
        name = tensor.name + "_updated"
    
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    compute_body(name, tensor.shape, fcompute, tensor.dtype, loc, tensor)
    return tensor



def mutate(domain, fcompute, name=None):
    if not isinstance(domain, tuple):
        raise APIError("The domain of mutate API must be a tuple")
    if name is None:
        name = UniqueName.get("tensor")
    
    # Generate a ComputeOp
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    compute_body(name, domain, fcompute, None, loc, "no_alloc")
    return



def bitcast(tensor, dst_dtype, name=None):
    """Bitcast a HeteroCL tensor or expression to the destination data type of the same bitwidth.
    This API **bitcast** the input tensor from its own data type (source dtype)
    to the destination data type (dst_dtype). The destination data type must have
    the same bitwidth with the source datatype.
    """
    if not isinstance(tensor, ast.AllocOp) and not isinstance(tensor, ast.Expr):
        raise APIError("bitcast input must be HeteroCL Tensor or Expression.")

    # check type
    if isinstance(dst_dtype, str):
        dst_dtype = dtype_to_hcl(dst_dtype)
    elif not isinstance(dst_dtype, Type):
        raise APIError("dst_dtype should be HeteroCL data type.")
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    # set up name, shape, and fcompute
    dst_dtype_str = get_dtype_str(dst_dtype)
    if isinstance(tensor, ast.AllocOp):
        name = tensor.name + "_" + dst_dtype_str if name is None else name
        shape = tensor.shape
        fcompute = lambda *args: ast.BitCastOp(tensor[args], dst_dtype, loc)
        return compute(shape, fcompute, name=name, dtype=dst_dtype)
    else:
        bitcast = ast.BitCastOp(tensor, dst_dtype, loc)
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


def match(scope, pattern):
    """Match the pattern in the given scope.
    Parameters
    ----------
    scope : Scope
        The scope to be matched. Either a function or a stage.
    pattern : Pattern
        The pattern to be matched. Python regular expression.
    Returns
    -------
    matched : list
        A list of matched stages.
    """
    # Check if scope is a function or a stage
    if not inspect.isfunction(scope) and not isinstance(scope, Stage):
        raise APIError("The scope of match API must be a function or a stage.")
    if not isinstance(pattern, str) and not inspect.isfunction(pattern):
        raise APIError("The pattern of match API must be a string or a lambda function.")
    
    matched = []
    if isinstance(pattern, str):
        # Check if pattern is a valid regular expression
        try:
            re.compile(pattern)
        except re.error:
            raise APIError("The pattern of match API must be a valid regular expression.")

    def _ismatch(pattern, stage):
        if isinstance(pattern, str):
            return re.match(pattern, stage.name)
        else:
            return pattern(stage)

    # Check if scope is the top function
    if inspect.isfunction(scope):
        if scope == Schedule._TopFunction:
            # search in the top function
            for _, stage in Stage._mapping:
                if _ismatch(pattern, stage):
                    if stage not in matched:
                        matched.append(stage)
        else: # search in local function
            for stage in scope._stages:
                if _ismatch(pattern, stage):
                    if stage not in matched:
                        matched.append(stage)
    else: # search in stage
        for stage in scope._sub_stages:
            if _ismatch(pattern, stage):
                if stage not in matched:
                    matched.append(stage)
    return matched