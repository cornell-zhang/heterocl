"""Compute APIs in HeteroCL"""
#pylint: disable=no-member, redefined-builtin, too-many-arguments, missing-docstring
import numbers
import numpy as np
from collections import OrderedDict
from .tvm import expr as _expr, stmt as _stmt, make as _make
from .tvm.api import _IterVar, min_value
from .util import get_index, get_name, get_type, get_tvm_dtype, make_for, CastRemover
from .tensor import Scalar, Tensor, TensorSlice
from .types import Fixed, UFixed, Struct, Type, dtype_to_str, dtype_to_hcl
from .schedule import Stage
from .debug import APIError
from .dsl import if_, for_
from .mutator import Mutator
from .module import Module
from . import util

##############################################################################
# Helper classes and functions
##############################################################################

class ReplaceReturn(CastRemover):
    """Replace all Return statement with a Store statement.

    Attributes
    ----------
    buffer_var : Var
        The buffer variable of the Store statement

    dtype : Type
        The data type of the Store statement

    index : Expr
        The index of the Store statement
    """
    def __init__(self, buffer_var, dtype, index):
        self.buffer_var = buffer_var
        self.dtype = dtype
        self.index = index

    def mutate_KerenlDef(self, node):
        """Omit the KernelDef statement

        We do not need to replace the Return statement inside.
        """
        #pylint: disable=no-self-use
        return node

    def mutate_Return(self, node):
        """Replace the Return statement with a Store statement

        """
        return _make.Store(self.buffer_var, _make.Cast(self.dtype, node.value), self.index)

def process_fcompute(fcompute, shape):
    """Pre-process the fcompute field of an API.

    """
    # check API correctness
    if not callable(fcompute):
        raise APIError("The construction rule must be callable")
    # prepare the iteration variables
    args = [] # list of arguments' names
    nargs = 0 # number of arguments
    if isinstance(fcompute, Module):
        args = fcompute.arg_names
        nargs = len(args)
    else:
        args = list(fcompute.__code__.co_varnames)
        nargs = fcompute.__code__.co_argcount
    # automatically create argument names
    if nargs < len(shape):
        for i in range(nargs, len(shape)):
            args.append("args" + str(i))
    elif nargs > len(shape):
        raise APIError("The number of arguments exceeds the number of dimensions")
    return args, len(shape)

def compute_body(name,
                lambda_ivs,
                fcompute,
                shape=(),
                dtype=None,
                tensor=None,
                attrs=OrderedDict()):
    """Create a stage and perform the computation.

    If `tensor` is `None`, no tensor is returned.

    Parameters
    ----------
    name : str
        The name of the stage

    lambda_ivs : list of IterVar
        A list contains the iteration variables in the lambda function if
        exists

    fcompute : callable
        The computation rule

    shape : tuple, optional
        The output shape or the iteration domain

    dtype : Type, optional
        The data type of the output/updated tensor

    tensor : Tensor, optional
        The tensor to be updated. Create a new one if it is `None`

    Returns
    -------
    Tensor or None
    """
    name = util.legalize_name(name)
    var_list = [i.var for i in lambda_ivs]
    return_tensor = True if tensor is None else False

    with Stage(name, dtype, shape) as stage:
        if not return_tensor:
            stage.input_stages.add(tensor.last_update)
        else:
            tensor = Tensor(shape, stage._hcl_dtype, name, stage._buf)
        buffer_var = tensor._buf.data
        dtype = tensor.dtype
        shape = tensor.shape

        stage.stmt_stack.append([])
        ret = fcompute(*var_list)

        stage.lhs_tensors.add(tensor)
        for t in stage.lhs_tensors:
            t.last_update = stage

        stmt = None
        if ret is None:
            # replace all hcl.return_ with Store stmt
            indices = lambda_ivs
            index, _, _ = get_index(shape, indices, 0)
            stmt = stage.pop_stmt()
            stmt = ReplaceReturn(buffer_var, dtype, index).mutate(stmt)
            stmt = make_for(indices, stmt, 0, name)
        elif isinstance(ret, (tuple, list)):
            indices = lambda_ivs
            index, _, _ = get_index(shape, indices, 0)
            hcl_dtype = tensor.hcl_dtype
            if not isinstance(hcl_dtype, Struct):
                raise TensorError("Cannot assign a tuple/list to a non-struct-type tensor")
            start = 0
            end = 0
            for sdtype, expr in zip(hcl_dtype.dtype_dict.values(), ret):
                end = start + sdtype.bits
                sdtype = dtype_to_str(sdtype)
                load = _make.Load(dtype, buffer_var, index)
                expr = _make.Cast(sdtype, expr)
                if get_type(sdtype) != "uint":
                    ty = "uint" + str(get_type(sdtype)[1])
                    expr = _make.Call(ty, "bitcast", [expr], _expr.Call.PureIntrinsic, None, 0)
                expr = _make.SetSlice(load, expr, end, start)
                stage.emit(_make.Store(buffer_var,
                                       _make.Cast(dtype, expr),
                                       index))
                start = end
            stmt = make_for(indices, stage.pop_stmt(), 0, name)
        elif isinstance(ret, (TensorSlice, Scalar, _expr.Expr, numbers.Number)):
            indices = lambda_ivs
            index, _, _ = get_index(shape, indices, 0)
            stage.emit(_make.Store(buffer_var, _make.Cast(dtype, ret), index))
            stmt = make_for(indices, stage.pop_stmt(), 0, name)
        elif isinstance(ret, Tensor): # reduction
            ret_ivs = [_IterVar((0, ret.shape[i]), ret.name+"_i" + str(i), 0)
                       for i in range(0, len(ret.shape))]
            non_reduce_ivs = []
            indices = []
            rid = 0
            for iv in lambda_ivs:
                if iv.var.name[0] == "_":
                    indices.append(ret_ivs[rid])
                    rid += 1
                else:
                    indices.append(iv)
                    non_reduce_ivs.append(iv)
            if rid != len(ret.shape):
                raise APIError("Incorrect number of reduction axes in lambda arguments")
            index, _, _ = get_index(shape, indices, 0)
            st = _make.Store(buffer_var, _make.Cast(dtype, ret[tuple(ret_ivs)]), index)
            stage.emit(make_for(ret_ivs, st, 0, name))
            stmt = stage.pop_stmt()
            stage.input_stages.remove(stage)
            if non_reduce_ivs:
                stmt = make_for(non_reduce_ivs, stmt, 0, name)
        else:
            raise APIError("Unknown return type of the computation rule")
        # add attributes to the loop
        if isinstance(stmt, _stmt.For):
            stmt = _make.For(stmt.loop_var,
                             stmt.min, stmt.extent,
                             0, 0, stmt.body,
                             list(stmt.annotate_keys) + list(attrs.keys()),
                             list(stmt.annotate_values) + list(attrs.values()))
        stage.emit(stmt)
        stage.axis_list = indices + stage.axis_list

    if return_tensor:
        tensor._tensor = stage._op
        return tensor
    return None

##############################################################################
# APIs exposed to users
##############################################################################

def compute(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
    """Construct a new tensor based on the shape and the compute function.

    The API **returns a new tensor**. The shape must be a tuple. The number of
    elements in the tuple decides the dimension of the returned tensor. The
    second field `fcompute` defines the construction rule of the returned
    tensor, which must be callable. The number of arguments should match the
    dimension defined by `shape`, which *we do not check*. This, however,
    provides users more programming flexibility.

    The compute function specifies how we calculate each element of the
    returned tensor. It can contain other HeteroCL APIs, even imperative DSL.

    Parameters
    ----------
    shape : tuple
        The shape of the returned tensor

    fcompute : callable
        The construction rule for the returned tensor

    name : str, optional
        The name of the returned tensor

    dtype : Type, optional
        The data type of the placeholder

    Returns
    -------
    Tensor

    Examples
    --------
    .. code-block:: python

        # example 1.1 - anonymous lambda function
        A = hcl.compute((10, 10), lambda x, y: x+y)

        # equivalent code
        for x in range(0, 10):
            for y in range(0, 10):
                A[x][y] = x + y

        # example 1.2 - explicit function
        def addition(x, y):
            return x+y
        A = hcl.compute((10, 10), addition)

        # example 1.3 - imperative function definition
        @hcl.def_([(), ()])
        def addition(x, y):
            hcl.return_(x+y)
        A = hcl.compute((10, 10), addition)

        # example 2 - undetermined arguments
        def compute_tanh(X):
            return hcl.compute(X.shape, lambda *args: hcl.tanh(X[args]))

        A = hcl.placeholder((10, 10))
        B = hcl.placeholder((10, 10, 10))
        tA = compute_tanh(A)
        tB = compute_tanh(B)

        # example 3 - mixed-paradigm programming
        def return_max(x, y):
            with hcl.if_(x > y):
                hcl.return_(x)
            with hcl.else_:
                hcl.return_(y)
        A = hcl.compute((10, 10), return_max)
    """
    # check API correctness
    if not isinstance(shape, tuple):
        raise APIError("The shape of compute API must be a tuple")

    # properties for the returned tensor
    shape = CastRemover().mutate(shape)
    name = get_name("compute", name)

    # prepare the iteration variables
    args, nargs = process_fcompute(fcompute, shape)
    lambda_ivs = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]

    # call the helper function that returns a new tensor
    tensor = compute_body(name, lambda_ivs, fcompute, shape, dtype, attrs=attrs)

    return tensor

def update(tensor, fcompute, name=None):
    """Update an existing tensor according to the compute function.

    This API **update** an existing tensor. Namely, no new tensor is returned.
    The shape and data type stay the same after the update. For more details
    on `fcompute`, please check :obj:`compute`.

    Parameters
    ----------
    tensor : Tensor
        The tensor to be updated

    fcompute: callable
        The update rule

    name : str, optional
        The name of the update operation

    Returns
    -------
    None
    """
    # properties for the returned tensor
    shape = tensor.shape
    name = get_name("update", name)

    # prepare the iteration variables
    args, nargs = process_fcompute(fcompute, shape)
    lambda_ivs = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]

    # call the helper function that updates the tensor
    compute_body(name, lambda_ivs, fcompute, tensor=tensor)

def mutate(domain, fcompute, name=None):
    """
    Perform a computation repeatedly in the given mutation domain.

    This API allows users to write a loop in a tensorized way, which makes it
    easier to exploit the parallelism when performing optimizations. The rules
    for the computation function are the same as that of :obj:`compute`.

    Parameters
    ----------
    domain : tuple
        The mutation domain

    fcompute : callable
        The computation function that will be performed repeatedly

    name : str, optional
        The name of the operation

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        # this example finds the max two numbers in A and stores it in M

        A = hcl.placeholder((10,))
        M = hcl.placeholder((2,))

        def loop_body(x):
            with hcl.if_(A[x] > M[0]):
                with hcl.if_(A[x] > M[1]):
                    M[0] = M[1]
                    M[1] = A[x]
                with hcl.else_():
                    M[0] = A[x]
        hcl.mutate(A.shape, lambda x: loop_body(x))
    """
    # check API correctness
    if not isinstance(domain, tuple):
        raise APIError("The mutation domain must be a tuple")
    name = get_name("mutate", name)

    # prepare the iteration variables
    args, nargs = process_fcompute(fcompute, domain)
    indices = [_IterVar((0, domain[n]), args[n], 0) for n in range(0, nargs)]
    var_list = [i.var for i in indices]

    # perform the computation
    with Stage(name) as stage:
        stage.stmt_stack.append([])
        fcompute(*var_list)
        body = stage.pop_stmt()
        stage.emit(make_for(indices, body, 0, name))
        stage.axis_list = indices + stage.axis_list

def scalar(init=0, name=None, dtype=None):
    """A syntactic sugar for a single-element tensor.

    This is equivalent to ``hcl.compute((1,), lambda x: init, name, dtype)``

    Parameters
    ----------
    init : Expr, optional
        The initial value for the returned tensor. The default value is 0.

    name : str, optional
        The name of the returned tensor

    dtype : Type, optional
        The data type of the placeholder

    Returns
    -------
    Tensor
    """
    name = get_name("scalar", name)
    return compute((1,), lambda x: init, name, dtype)

def copy(tensor, name=None, dtype=None):
    """A syntactic sugar for copying an existing tensor.

    Parameters
    ----------
    tensor : Tensor or list or numpy.ndarray
        The tensor to be copied from

    name : str, optional
        The name of the returned tensor

    Returns
    -------
    Tensor

    Examples
    --------
    .. code-block:: python

        # example 1 - copy from a HeteroCL tensor
        A = hcl.placeholder((10,), "A", hcl.UInt(32))
        B1 = hcl.copy(A, "B1")

        # example 2 - copy from a Python list
        pA = [[1, 2, 3], [4, 5, 6]]
        # The data type is NOT inferred from the list
        B2 = hcl.copy(pA, "B2", hcl.Int())

        # example 3 - copy from a Numpy array
        nA = numpy.array(pA)
        # The data type is determined by using nA.dtype
        B3 = hcl.copy(nA, "B3")
    """
    name = get_name("copy", name)
    if isinstance(tensor, Tensor):
        return compute(
                tensor.shape,
                lambda *args: tensor[args],
                name,
                tensor.dtype)
    elif isinstance(tensor, (list, np.ndarray)):
        if isinstance(tensor, np.ndarray):
            shape = tensor.shape
            _tensor = tensor = tensor.tolist()
        else:
            _tensor = tensor
            shape = []
            while isinstance(_tensor, list):
                shape.append(len(_tensor))
                _tensor = _tensor[0]
            shape = tuple(shape)


        def _iter_tensor(_tensor, tensor, indices, buffer_var):
            if isinstance(tensor, list):
                for x in range(0, len(tensor)):
                    indices.append(x)
                    _iter_tensor(_tensor, tensor[x],
                                 indices, buffer_var)
                    indices.pop()
            else:
                index, _, _ = get_index(shape, indices, 0)
                stage.emit(
                        _make.Store(
                            buffer_var,
                            _make.Cast(stage._dtype, tensor),
                            index))

        with Stage(name, dtype, shape) as stage:
            _tensor = Tensor(shape, stage._dtype, name, stage._buf)
            _iter_tensor(_tensor, tensor, [], _tensor._buf.data)
            stage.lhs_tensors.add(_tensor)
            for t in stage.lhs_tensors:
                t.last_update = stage
        _tensor._tensor = stage._op
        return _tensor
    else:
        raise APIError("Unkown tensor type. Should be either HeteroCL tensor, \
                Python list, or Numpy array.")

def unpack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Unpack a tensor with larger bitwidth to a tensor with smaller bitwidth.

    This API unpacks the `axis`-th dimension of `tensor` to a new tensor
    according to the given `factor` or `dtype`. The number of dimensions stays
    the same after unpacking. Once `factor` is specified, `dtype` is not taken
    into consideration. If `factor` is not specified, users can have several
    ways to specify `dtype`. First, we use the data type specified by
    the quantization scheme. Second, if `dtype` is specified, we use the value.
    Finally, we use the data type specified via the :obj:`heterocl.init` API.
    Since we are performing an unpacking operation, the number of resulting
    elements should be larger then that of the elements in the input tensor.
    Namely, *the factor should be greater or equal to 1*.

    Parameters
    ----------
    tensor : Tensor
        The tensor to be unpacked

    axis : int, optional
        The dimension to be unpacked

    factor : int, optional
        The unpack factor

    name : str, optional
        The name of the unpacked tensor

    dtype : Type, optional
        The data type of the **unpacked tensor**

    Returns
    -------
    Tensor

    Examples
    --------
    .. code-block:: python

        # example 1.1 - unpack with factor
        A = hcl.placeholder((10,), "A", hcl.UInt(32))
        B = hcl.unpack(A, factor=4)
        print B.shape # (40,)
        print B.dtype # "uint8"

        # example 1.2 - unpack with dtype
        A = hcl.placeholder((10,), "A", hcl.UInt(32))
        B = hcl.unpack(A, dtype=hcl.UInt(8))
        # the results are the same as example 1.1

        # example 1.3 - unpack with quantization scheme
        A = hcl.placeholder((10,), "A", hcl.UInt(32))
        def unpack_A(A):
            return hcl.unpack(A, name="B")
        s = hcl.create_scheme(A, unpack_A)
        s.downsize(unpack_A.B, hcl.UInt(8))
        # the results are the same as example 1.1

        # example 2 - unpack multi-dimensional tensor
        A = hcl.placeholder((10, 10), "A", hcl.UInt(32))
        B = hcl.unpack(A, factor=4)         # B.shape = (40, 10)
        C = hcl.unpack(A, axis=1, factor=4) # C.shape = (10, 40)
    """
    name = get_name("unpack", name)

    # derive the final factor and dtype
    if factor is None:
        # if factor is not given, we need to check the quantization scheme
        # to do so, we will need the name
        name_ = name if Stage.get_len() == 0 \
                     else Stage.get_current().name_with_prefix + "." + name
        dtype = get_tvm_dtype(dtype, name_)
        ret = get_type(dtype)
        factor = tensor.type.bits // ret[1]
        bitwidth = ret[1]
    else:
        ret = get_type(tensor.dtype)
        bitwidth = ret[1] // factor
        dtype = ret[0] + str(bitwidth)

    # derive the new shape
    ndim = len(tensor.shape)
    if axis > ndim:
        raise APIError("The unpack dimension exceeds the number of dimensions")
    new_shape = []
    for i in range(0, ndim):
        if i == axis:
            new_shape.append(tensor.shape[i] * factor)
        else:
            new_shape.append(tensor.shape[i])

    # derive the output tensor
    def assign_val(*indices):
        temp = scalar(0, name+"_temp", dtype)
        new_indices = []
        for i in range(0, ndim):
            if i == axis:
                new_indices.append(indices[i] // factor)
            else:
                new_indices.append(indices[i])
        index = indices[axis]
        lower = (index%factor) * bitwidth
        upper = lower + bitwidth
        temp[0][bitwidth:0] = tensor[tuple(new_indices)][upper:lower]
        return temp[0]

    return compute(tuple(new_shape), assign_val, name, dtype)

def pack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Pack a tensor with smaller bitwidth to a tensor with larger bitwidth.

    This API packs the `axis`-th dimension of `tensor` to a new tensor
    according to the given `factor` or `dtype`. The usage is the same as
    :obj:`unpack`.

    Parameters
    ----------
    tensor : Tensor
        The tensor to be packed

    axis : int, optional
        The dimension to be packed

    factor : int, optional
        The pack factor

    name : str, optional
        The name of the packed tensor

    dtype : Type, optional
        The data type of the **packed tensor**

    Returns
    -------
    Tensor
    """
    name = get_name("pack", name)

    # derive the final factor and dtype
    if factor is None:
        # if factor is not given, we need to check the quantization scheme
        # to do so, we will need the name
        name_ = name if Stage.get_len() == 0 \
                     else Stage.get_current().name_with_prefix + "." + name
        dtype = get_tvm_dtype(dtype, name_)
        ret = get_type(dtype)
        factor = ret[1] // tensor.type.bits
        bitwidth = tensor.type.bits
    else:
        ret = get_type(tensor.dtype)
        bitwidth = ret[1]
        dtype = ret[0] + str(bitwidth * factor)

    # derive the new shape
    ndim = len(tensor.shape)
    if axis > ndim:
        raise APIError("The pack dimension exceeds the number of dimensions")
    new_shape = []
    for i in range(0, ndim):
        if i == axis:
            new_shape.append(tensor.shape[i] // factor)
        else:
            new_shape.append(tensor.shape[i])

    # derive the packed tensor
    def assign_val(*indices):
        temp = scalar(0, name+"_temp", dtype)
        with for_(0, factor) as i:
            new_indices = []
            for j in range(0, ndim):
                if j == axis:
                    new_indices.append(indices[j]*factor+i)
                else:
                    new_indices.append(indices[j])
            temp[0][bitwidth*(i+1) : bitwidth*i] = tensor[tuple(new_indices)]
        return temp[0]

    return compute(tuple(new_shape), assign_val, name, dtype)


def reduce_axis(lower, upper, name=None):
    """Create a reduction axis for reduction operations.

    The upper- and lower-bound of the range can be arbitrary integers. However,
    the upper-bound should be greater than the lower-bound.

    Parameters
    ----------
    lower : Expr
        The lower-bound of the reduction domain

    upper : Expr
        The upper-bound of the reduction domain

    name : str, optional
        The name of the reduction axis

    Returns
    -------
    IterVar
    """
    # check the correctness
    if upper <= lower:
        raise APIError("The upper-bound should be greater then the lower-bound")

    name = get_name("ra", name)
    name = util.legalize_name(name)
    return _IterVar((lower, upper), name, 2)

def const_tensor(values, name=None, dtype=None):
    """Create a constant tensor
    """
    name = get_name("const", name)
    name = util.legalize_name(name)

    if not isinstance(values, np.ndarray):
        values = np.array(values)
    shape = values.shape
    values = values.flatten()
    values = values.tolist()

    tensor = None
    with Stage(name, dtype, shape) as stage:
        tensor = Tensor(shape, stage._hcl_dtype, name, stage._buf)
        tensor.last_update = stage
        stage.init_values = values
        stage.is_const = True

    tensor._tensor = stage._op
    return tensor


def reducer(init, freduce, dtype="int32", name=None):
    """Create a reducer for a reduction operation.

    This API creates a reducer according to the initial value `init` and the
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
        output = init
        # the specified reduction axis
        for i in reduction_domain:
            if (where):
                output = freduce(input[..., i, ...], output)

    Users can further specify the data type for the reduction operation. For
    a multi-dimensional reduction operation, users can have multiple reduce
    axes. In this case, we can write them together in a list.

    Parameters
    ----------
    init : Expr or Tensor
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
        init = hcl.compute((10,), lambda x: 11)
        def freduce(x, Y):
            with hcl.for_(0, 10) as i:
                with hcl.if_(x < Y[i]):
                    with hcl.for_(9, i, -1) as j:
                        Y[j] = Y[j-1]
                    Y[i] = x
                    hcl.break_()
        my_sort = hcl.reducer(init, freduce)
        A = hcl.placeholder((10, 10))
        r = hcl.reduce_axis(0, 10)
        # note that we need to use the underscore the mark the reduction axis
        B = hcl.compute(A.shape, lambda _x, y: my_sort(A[r, y], axis=r))
    """
    def make_reduce(expr, axis, where=True, name=name, dtype=dtype):
        if not isinstance(axis, (tuple, list)):
            axis = [axis]
        stage = Stage.get_current()
        out = None
        name = get_name("reducer", name)
        # the accumulator is an expression
        if isinstance(init, (_expr.Expr, numbers.Number, Scalar)):
            out = scalar(init, name, dtype)
            def reduce_body():
                stage.stmt_stack.append([])
                with if_(where):
                    ret = freduce(expr, out[0])
                    if ret is None:
                        stmt = stage.pop_stmt()
                        stmt = ReplaceReturn(out._buf.data, out.dtype, 0).mutate(stmt)
                        stage.emit(stmt)
                    else:
                        if not isinstance(ret, (_expr.Expr, numbers.Number, Scalar)):
                            raise APIError("The returned type of the \
                                    reduction function should be an expression")
                        out[0] = ret
                        stmt = stage.pop_stmt()
                        stage.emit(stmt)
                return out[0]
            stage.stmt_stack.append([])
            ret = reduce_body()
        # the accumulator is a tensor
        else:
            out = copy(init, name)
            def reduce_body():
                with if_(where):
                    freduce(expr, out)
                return out
            stage.stmt_stack.append([])
            ret = reduce_body()
        body = stage.pop_stmt()
        stage.input_stages.add(out.last_update)
        body = make_for(axis, body, 0, stage.name)
        stage.axis_list += axis
        stage.emit(body)
        return ret

    doc_str = """Compute the {0} of the given expression on axis.

              Parameters
              ----------
              expr : Expr
                  The expression to be reduced

              axis : IterVar
                  The axis to be reduced

              where : Expr, optional
                  The filtering condition for the reduction

              name : str, optional
                  The name of the accumulator

              dtype : Type, optional
                  The data type of the accumulator

              Returns
              -------
              Expr

              See Also
              --------
              reducer
              """
    name = util.legalize_name(name)
    make_reduce.__doc__ = doc_str.format(name)
    return make_reduce


def bitcast(tensor, dst_dtype, name=None):
    """Bitcast a HeteroCL tensor or expression to the destination data type of the same bitwidth.

    This API **bitcast** the input tensor from its own data type (source dtype)
    to the destination data type (dst_dtype). The destination data type must have
    the same bitwidth with the source datatype. 

    Parameters
    ----------
    tensor : Tensor or Expr
        The input tensor or expression of the source data type

    dst_dtype : Type
        The destination data type. For example, hcl.UInt(32)

    Name : str, optional
        The name of the returned tensor
    
    Returns
    -------
    Tensor of Expr
        The bitcasted tensor or expression of the destination data type
    """
    
    # check type
    if not isinstance(tensor, Tensor) and not isinstance(tensor, _expr.Expr):
        raise APIError("bitcast input must be HeteroCL Tensor or Expr.")

    if not isinstance(dst_dtype, Type):
        raise APIError("dst_dtype should be HeteroCL data type.")

    # check bitwidth
    src_bitwidth = get_type(tensor.dtype)[1]
    dst_bitwidth = dst_dtype.bits 
    if src_bitwidth != dst_bitwidth:
        raise APIError("Destination datatype bitwidth does not match source bitwidth:" +
                        f"source bitwidth: {src_bitwidth} , destination bitwidth {dst_bitwidth}.") 

    # set up name, shape, and fcompute
    name = util.legalize_name(name)
    dst_dtype_str = dtype_to_str(dst_dtype)
    if isinstance(tensor, Tensor):
        name = tensor.name + '_' + dst_dtype_str if name is None else name
        shape = tensor.shape
        fcompute = lambda *args : _make.Call(dst_dtype_str, "bitcast", [tensor[args]], _expr.Call.PureIntrinsic, None, 0)
        return compute(shape, fcompute, name=name, dtype=dst_dtype) 
    else: # expression
        expr = _make.Call(dst_dtype_str, "bitcast", [tensor], _expr.Call.PureIntrinsic, None, 0)
        return expr

sum = reducer(0, lambda x, y: x + y, name="sum")
max = reducer(min_value("float"), _make.Max, name="max")
