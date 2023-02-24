# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import inspect

import hcl_mlir
import numpy as np
from hcl_mlir.exceptions import APIError

from . import config
from .types import Int, Type, UInt, Struct, dtype_to_hcl
from .context import UniqueName
from .dsl import for_
from .schedule import Schedule, Stage
from .tensor import Array
from .utils import (
    get_src_loc,
    make_const_tensor,
    get_max_value,
    get_min_value,
    get_dtype_str,
)
from .ast import ast


def init(init_dtype=Int(32), raise_assert_exception=True):
    """Initialize a HeteroCL environment with configurations."""
    config.init_dtype = init_dtype
    config.raise_assert_exception = raise_assert_exception


def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs."""
    name = UniqueName.get(name, "tensor")

    if (
        not dtype is None
        and not isinstance(dtype, (Type, str))
        and not hcl_mlir.is_hcl_mlir_type(dtype)
        and not isinstance(name, str)
    ):
        raise APIError(f"Input type error, got dtype={dtype}, name={name}")
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    if shape == ():
        shape = (1,)
    dtype = config.init_dtype if dtype is None else dtype
    filename, lineno = get_src_loc(frame=1)
    alloc = ast.AllocOp(name, shape, dtype, ast.Location(filename, lineno))
    return alloc


def asarray(np_array, dtype=None):
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype is None else dtype
    return Array(np_array, dtype)


def scalar(init_val, name=None, dtype=None):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    name = UniqueName.get(name, "scalar")
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype is None else dtype  # dtype is HeteroCL type
    if isinstance(dtype, Struct) and isinstance(init_val, int):
        vals = []
        for ftype in dtype.dtype_dict.values():
            mask = (1 << (ftype.bits + 1)) - 1
            val = init_val & mask
            init_val = init_val >> ftype.bits
            vals.append(ast.ConstantOp(val, ftype, loc))
        init_val = tuple(vals)

    # Generate a ComputeOp
    op = compute_body(name, (1,), lambda x: init_val, dtype, loc, None)
    return op.tensor


def reduce_axis(lower, upper, name=None):
    """Create a reduction axis for reduction operations."""
    name = UniqueName.get(name, "r")  # r stands for reduction
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.ReduceVar(name, parent_loop=None, loc=loc, bound=(lower, upper))


def cast(dtype, expr):
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    if isinstance(expr, ast.AllocOp):
        raise APIError(
            "Tensor is not supported in hcl.cast. "
            + "If you are try to cast a hcl.scalar, please use hcl.cast(scalar.v)"
        )
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.CastOp(expr, dtype, loc)


def const_tensor(values, name=None, dtype=None):
    """Create a constant tensor"""
    name = UniqueName.get(name, "tensor")
    dtype = config.init_dtype if dtype is None else dtype
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    # convert values to numpy array and handle overflow
    values = np.array(values)
    shape = values.shape
    values = make_const_tensor(values, dtype)
    cst_op = ast.ConstantTensorOp(values, name, shape, dtype, loc)
    region = ast.scope.get()
    region.append(cst_op)
    return cst_op.tensor


def copy(values, name=None, dtype=None):
    """A syntactic sugar for copying an existing tensor."""
    name = UniqueName.get(name, "tensor")
    dtype = config.init_dtype if dtype is None else dtype
    return const_tensor(values, name, dtype)


def select(cond, true_val, false_val):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    true_val = ast.immediate_to_constant(true_val, loc)
    false_val = ast.immediate_to_constant(false_val, loc)
    cond = ast.immediate_to_constant(cond, loc)
    return ast.SelectOp(cond, true_val, false_val, loc)


def reducer(init_val, freduce, dtype="int32", name=None):
    """Create a reducer for a reduction operation.
    This API creates a reducer according to the initial value `init_val` and the
    reduction function `freduce`. The initial value can be either an
    expression or a tensor. With the reducer, users can create a reduction
    operation, where the users can further specify the input to be reduced
    `expr`, its axis `axis`, and the condition `where`. The general rule of
    the reduction operation is shown below. Note that for the reduction
    function, **the first argument is the input while the second argument
    is the accumulator**. Moreover, if the accumulator is an expression,
    the reduction function **should return an expression**. On the other hand,
    if the accumulator is a list or a tensor, the reduction function **should
    not return anything**.
    .. code-block:: python
        # this can be a tensor
        output = init_val
        # the specified reduction axis
        for i in reduction_domain:
            if (where):
                output = freduce(input[..., i, ...], output)
    Users can further specify the data type for the reduction operation. For
    a multi-dimensional reduction operation, users can have multiple reduce
    axes. In this case, we can write them together in a list.
    Parameters
    ----------
    init_val : Expr or Tensor
        The initial value of the accumulator
    freduce : callable
        The reduction function that takes in two arguments. The first argument
        is the new input value and the second argument is the accumulator
    dtype : Type, optional
        The data type of the accumulator
    name : str, optional
        The name of the generated reducer
    Returns
    -------
    callable
    See Also
    --------
    sum, max
    Examples
    --------
    .. code-block:: python
        # example 1.1 - basic reduction : summation
        my_sum = hcl.reducer(0, lambda x, y: x+y)
        A = hcl.placeholder((10,))
        r = hcl.reduce_axis(0, 10)
        B = hcl.compute((1,), lambda x: my_sum(A[r], axis=r))
        # equivalent code
        B[0] = 0
        for r in (0, 10):
            B[0] = A[r] + B[0]
        # example 1.2 - with condition
        B = hcl.compute((1,), lambda x: my_sum(A[r], axis=r, where=A[r]>5))
        # equivalent code
        B[0] = 0
        for r in (0, 10):
            if A[r] > 5:
                B[0] = A[r] + B[0]
        # example 1.3 - with data type specification
        B = hcl.compute((1,), lambda x: my_sum(A[r], axis=r, dtype=hcl.UInt(4)))
        # the output will be downsize to UInt(4)
        # example 2 = a more complicated reduction
        # x is the input, y is the accumulator
        def my_reduction(x, y):
            with hcl.if_(x > 5):
                hcl.return_(y + x)
            with hcl.else_():
                hcl.return_(y - x)
        my_sum = hcl.reducer(0, my_reduction)
        A = hcl.placeholder((10,))
        r = hcl.reduce_axis(0, 10)
        B = hcl.compute((1,), lambda x: my_sum(A[r], axis=r))
        # equivalent code
        B[0] = 0
        for r in range(0, 10):
            if A[r] > 5:
                B[0] = B[0] + A[r]
            else:
                B[0] = B[0] - A[r]
        # example 3 - multiple reduce axes
        A = hcl.placeholder((10, 10))
        r1 = hcl.reduce_axis(0, 10)
        r2 = hcl.reduce_axis(0, 10)
        B = hcl.compute((1,), lambda x: my_sum(A[r1, r2], axis=[r1, r2]))
        # equivalent code
        B[0] = 0
        for r1 in (0, 10):
            for r2 in (0, 10):
                B[0] = A[r1, r2] + B[0]
        # example 4 - write a sorting algorithm with reduction
        init_val = hcl.compute((10,), lambda x: 11)
        def freduce(x, Y):
            with hcl.for_(0, 10) as i:
                with hcl.if_(x < Y[i]):
                    with hcl.for_(9, i, -1) as j:
                        Y[j] = Y[j-1]
                    Y[i] = x
                    hcl.break_()
        my_sort = hcl.reducer(init_val, freduce)
        A = hcl.placeholder((10, 10))
        r = hcl.reduce_axis(0, 10)
        # note that we need to use the underscore the mark the reduction axis
        B = hcl.compute(A.shape, lambda _x, y: my_sort(A[r, y], axis=r))
    """

    def make_reduce(expr, axis, where=True, name=name, dtype=dtype):
        name = UniqueName.get(name, "op")
        if isinstance(axis, tuple):
            axis = list(axis)
        elif isinstance(axis, ast.ReduceVar):
            axis = [axis]
        elif not isinstance(axis, list):
            raise APIError("axis must be a list of reduction axis")

        # Set up data type
        # TODO: deduce dtype from expr
        dtype = config.init_dtype if dtype is None else dtype
        if isinstance(dtype, str):
            dtype = dtype_to_hcl(dtype)
        # Set up file location
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)

        nonlocal init_val
        if init_val is None:
            if freduce == ast.Min:
                init_val = get_max_value(dtype)
            elif freduce == ast.Max:
                init_val = get_min_value(dtype)
            else:
                raise APIError(
                    "init_val must be specified if freduce is not a reduction operator"
                )

        reduce_op = ast.ReduceOp(name, expr, freduce, axis, dtype, init_val, loc)
        ast.scope.push(reduce_op.body)
        if_op = ast.IfOp(where, loc)
        reduce_op.body.append(if_op)
        # trace the reduction freduce
        ast.scope.push(if_op.body)

        if freduce == ast.Min:
            res = ast.Min(expr, reduce_op.scalar[0], loc)
        elif freduce == ast.Max:
            res = ast.Max(expr, reduce_op.scalar[0], loc)
        elif callable(freduce):
            res = freduce(expr, reduce_op.scalar[0])
        else:
            raise APIError("freduce must be a callable or a reduction operator")

        casted = ast.CastOp(res, dtype, loc)
        store_op = ast.StoreOp(reduce_op.scalar, (0,), casted, loc)
        if_op.body.append(store_op)
        ast.scope.pop()  # if_op.body
        ast.scope.pop()  # reduce_op.body
        return reduce_op

    return make_reduce


# pylint: disable=redefined-builtin
sum = reducer(0, lambda x, y: x + y, name="sum")
max = reducer(None, ast.Max, name="max")
min = reducer(None, ast.Min, name="min")


def pack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Pack a tensor with smaller bitwidth to a tensor with larger bitwidth."""
    if factor is None and dtype is not None:
        factor = dtype.bits // tensor.dtype.bits
    if factor is None or not isinstance(factor, int):
        raise APIError("Should specify factor")
    if not isinstance(tensor.dtype, (Int, UInt)):
        raise APIError("Only support integer packing")
    name = UniqueName.get(name, "tensor")
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
            result[0][bitwidth * i : bitwidth * (i + 1)] = tensor[tuple(new_indices)]
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
    name = UniqueName.get(name, "tensor")
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
    """Create an ast.ComputeOp and its body operations

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
    loc: Location
        The location of the compute op
    tensor: Union[ast.AllocOp, None, or "no_alloc"]
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
        raise APIError(
            f"fcompute's number of axis does not match output tensor shape: {axis_names} vs {shape}"
        )
    # unique axis names
    axis_names = [UniqueName.get(name, "axis") for name in axis_names]
    iter_vars = [ast.IterVar(name, None, loc) for name in axis_names]
    # attach iter_vars to the compute op
    # iter_var's parent_loop will be set in ir.ir_builder.build_compute
    compute_op.iter_vars.extend(iter_vars)
    compute_op.reduce_vars.extend(reduce_vars)
    ast.scope.push(compute_op.body)
    if tensor is None:
        # hcl.compute
        # pylint: disable=redefined-variable-type
        res_expr = fcompute(*iter_vars)
        res_expr = ast.immediate_to_constant(res_expr, loc)
        if isinstance(res_expr, tuple) and isinstance(dtype, Struct):
            res_expr = ast.StructConstructOp(list(res_expr), dtype, loc)
        if res_expr is None:
            if len(compute_op.body) > 0 and isinstance(
                compute_op.body[-1], ast.ReturnOp
            ):
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
            if len(compute_op.body) > 0 and isinstance(
                compute_op.body[-1], ast.ReturnOp
            ):
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


def compute(shape, fcompute, name=None, dtype=None):
    if not isinstance(shape, tuple):
        raise APIError("The shape of compute API must be a tuple")
    shape = tuple(int(s) if isinstance(s, float) else s for s in shape)
    name = UniqueName.get(name, "tensor")
    if not dtype is None and not isinstance(dtype, (Type, str)):
        raise APIError("Type error")
    dtype = config.init_dtype if dtype is None else dtype
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
    name = UniqueName.get(name, "tensor")

    # Generate a ComputeOp
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    compute_body(name, domain, fcompute, None, loc, "no_alloc")


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
    # pylint: disable=no-else-return
    if isinstance(tensor, ast.AllocOp):
        name = tensor.name + "_" + dst_dtype_str if name is None else name
        shape = tensor.shape
        return compute(
            shape,
            lambda *args: ast.BitCastOp(tensor[args], dst_dtype, loc),
            name=name,
            dtype=dst_dtype,
        )
    else:
        bitcast_op = ast.BitCastOp(tensor, dst_dtype, loc)
        return bitcast_op


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
        raise APIError(
            "The pattern of match API must be a string or a lambda function."
        )

    matched = []
    if isinstance(pattern, str):
        # Check if pattern is a valid regular expression
        try:
            re.compile(pattern)
        except Exception as exc:
            raise APIError(
                "The pattern of match API must be a valid regular expression."
            ) from exc

    def _ismatch(pattern, stage):
        if isinstance(pattern, str):
            return re.match(pattern, stage.name)
        return pattern(stage)

    # Check if scope is the top function
    if inspect.isfunction(scope):
        if scope == Schedule._TopFunction:
            # search in the top function
            for _, stage in Stage._mapping:
                if _ismatch(pattern, stage):
                    if stage not in matched:
                        matched.append(stage)
        else:  # search in local function
            for stage in scope._stages:
                if _ismatch(pattern, stage):
                    if stage not in matched:
                        matched.append(stage)
    else:  # search in stage
        for stage in scope._sub_stages:
            if _ismatch(pattern, stage):
                if stage not in matched:
                    matched.append(stage)
    return matched
