import numbers
import inspect
from .tvm import expr as _expr, stmt as _stmt, make as _make
from .tvm import _api_internal
from .tvm.api import _IterVar, decl_buffer, convert, min_value
from .util import get_index, make_for, get_dtype, CastRemover
from .tensor import Tensor, TensorSlice
from .schedule import Stage
from .debug import APIError, HCLError
from .dsl import *
from .mutator import Mutator
from .module import Module

class ReplaceReturn(Mutator):

    def __init__(self, buffer_var, dtype, index):
        self.buffer_var = buffer_var
        self.dtype = dtype
        self.index = index

    def mutate_KerenlDef(self, node):
        return node

    def mutate_Return(self, node):
        value = self.mutate(node.value)
        return _make.Store(self.buffer_var, _make.Cast(self.dtype, value), self.index)


def process_fcompute(fcompute, shape):
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

def compute_body(name, lambda_ivs, fcompute, shape=(), dtype=None, tensor=None):
    var_list = [i.var for i in lambda_ivs]
    return_tensor = True if tensor is None else False

    with Stage(name, dtype, shape) as stage:
        dtype = stage._dtype
        if not return_tensor:
            stage.input_stages.add(tensor.last_update)
        else:
            tensor = Tensor(shape, dtype, name, stage._buf)
        buffer_var = tensor._buf.data
        dtype = tensor.dtype
        shape = tensor.shape

        stage.stmt_stack.append([])
        ret = fcompute(*var_list)

        stage.lhs_tensors.add(tensor)
        for t in stage.lhs_tensors:
            t.last_update = stage

        if ret is None:
            # replace all hcl.return_ with Store stmt
            indices = lambda_ivs
            index, _, _ = get_index(shape, indices, 0)
            stmt = stage.pop_stmt()
            stmt = ReplaceReturn(buffer_var, dtype, index).mutate(stmt)
            stage.emit(make_for(indices, stmt, 0))
        elif isinstance(ret, (TensorSlice, _expr.Expr, numbers.Number)):
            indices = lambda_ivs
            index, _, _ = get_index(shape, indices, 0)
            stage.emit(_make.Store(buffer_var, _make.Cast(dtype, ret), index))
            stmt = stage.pop_stmt()
            stage.emit(make_for(indices, stmt, 0))
        elif isinstance(ret, Tensor): # reduction
            ret_ivs = [_IterVar((0, ret.shape[i]), ret.name + "_i" + str(i), 0) for i in range(0, len(ret.shape))]
            indices = []
            rid = 0
            for iv in lambda_ivs:
                if iv.var.name[0] == "_":
                    indices.append(ret_ivs[rid])
                    rid += 1
                else:
                    indices.append(iv)
            if len(indices) != len(shape):
                raise HCLError("Incorrect number of lambda arguments", inspect.stack()[2])
            index, _, _ = get_index(shape, indices, 0)
            stage.emit(_make.Store(buffer_var, _make.Cast(dtype, ret[tuple(ret_ivs)]), index))
            stmt = stage.pop_stmt()
            stage.emit(make_for(indices, stmt, 0))
        else:
            print ret
            #raise ValueError("Unrecognized return type")

        stage.axis_list = indices + stage.axis_list

    if return_tensor:
        tensor._tensor = stage._op
        return tensor

def compute(shape, fcompute, name=None, dtype=None):
    """Construct a new tensor based on the shape and the compute function.

    The API **returns a new tensor**. The shape must be a tuple. The number of
    elements in the tuple decides the dimension of the returned tensor. The
    second field `fcompute` defines the construction rule of the returned
    tensor, which must be callable. The number of arguemnts should match the
    dimension defined by `shape`, which *we do not check*. This, however,
    provides users more programming flexibility.

    The compute function specifies how we calculate each element of the
    returned tensor. It can contain other HeteroCL APIs, even imperative DSL.

    Examples
    --------
    .. code-block:: python

        # example 1.1 - anonymoous lambda function
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
    """
    # check API correctness
    if not isinstance(shape, tuple):
        raise APIError("The shape of compute API must be a tuple")

    # properties for the returned tensor
    shape = util.CastRemover().mutate(shape)
    name = util.get_name("compute", name)

    # prepare the iteration variables
    args, nargs = process_fcompute(fcompute, shape)
    lambda_ivs = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]

    # call the helper function that returns a new tensor
    tensor = compute_body(name, lambda_ivs, fcompute, shape, dtype)

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
    name = util.get_name("update", name)

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
    """
    # check API correctness
    if not isinstance(domain, tuple):
        raise APIError("The mutation domain must be a tuple")
    name = util.get_name("mutate", name)

    # prepare the iteration variables
    args, nargs = process_fcompute(fcompute, domain)
    indices = [_IterVar((0, domain[n]), args[n], 0) for n in range(0, nargs)]
    var_list = [i.var for i in indices]

    # perform the computation
    with Stage(name) as stage:
        stage.stmt_stack.append([])
        fcompute(*var_list)
        body = stage.pop_stmt()
        stage.emit(util.make_for(indices, body, 0))
        stage.axis_list = indices + stage.axis_list

def local(init=0, name=None, dtype=None):
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
    name = util.get_name("local", name)
    return compute((1,), lambda x: init, name, dtype)

def copy(tensor, name=None):
    """A syntactic sugar for copying an existing tensor.

    Parameters
    ----------
    tensor : Tensor
        The tensor to be copied from

    name : str, optional
        The name of the returned tensor

    Returns
    -------
    Tensor
    """
    name = util.get_name("copy", name)
    return compute(tensor.shape, lambda *args: tensor[args], name, tensor.dtype)

def unpack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Unpack a tensor with larger bitwidth to a tensor with smaller bitwidth

    This API unpacks the `axis`-th dimenson of `tensor` to a new tensor
    according to the given `factor` or `dtype`. The number of dimensions stays
    the same after unpacking. Once `factor` is specified, `dtype` is not taken
    into consideration. If `factor` is not specfied, users can have several
    ways to specify `dtype`. First, we use the data type specified by
    the quantization scheme. Second, if `dtype` is specfied, we use the value.
    Finally, we use the data type specified via the :obj:`init` API. Since we
    are performing an unpacking operation, the number of resulting elements
    should be larger then that of the elements in the input tensor. Namely,
    *the factor should be greater or equal to 1*.

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
        s = hcl.creat_scheme(A, unpack_A)
        s.downsize(unpack_A.B, hcl.UInt(8))
        # the results are the same as example 1.1

        # example 2 - unpack multi-dimensional tensor
        A = hcl.placeholder((10, 10), "A", hcl.UInt(32))
        B = hcl.unpack(A, factor=4)         # B.shape = (40, 10)
        C = hcl.unpack(A, axis=1, factor=4) # C.shape = (10, 40)

    Parameters
    ----------
    tensor : Tensor
        The tesnor to be unpacked

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
    """
    name = util.get_name("unpack", name)

    # derive the final factor and dtype
    if factor is None:
        # if factor is not given, we need to check the quantization schem
        # to do so, we will need the name
        name_ = name if Stage.get_len() == 0 \
                     else Stage.get_current().name_with_prefix + "." + name
        dtype = util.get_dtype(dtype, name_)
        ret = util.get_type(dtype)
        factor = tensor.type.bits / ret[1]
        bitwidth = ret[1]
    else:
        ret = util.get_type(tensor.dtype)
        bitwidth = ret[1]/factor
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
        temp = local(0, name+"_temp", dtype)
        new_indices = []
        for i in range(0, ndim):
            if i == axis:
                new_indices.append(indices[i]/factor)
            else:
                new_indices.append(indices[i])
        index = indices[axis]
        lower =(index%factor) * bitwidth
        upper = lower + bitwidth
        temp[0][bitwidth:0] = tensor[tuple(new_indices)][upper:lower]
        return temp[0]

    return compute(tuple(new_shape), assign_val, name, dtype)

def pack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Pack a tensor with smaller bitwidth to a tensor with larger bitwidth

    This API packs the `axis`-th dimenson of `tensor` to a new tensor
    according to the given `factor` or `dtype`. The usage is the same as
    :obj:`unpack`.

    Parameters
    ----------
    tensor : Tensor
        The tesnor to be packed

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
    name = util.get_name("pack", name)

    # derive the final factor and dtype
    if factor is None:
        # if factor is not given, we need to check the quantization schem
        # to do so, we will need the name
        name_ = name if Stage.get_len() == 0 \
                     else Stage.get_current().name_with_prefix + "." + name
        dtype = util.get_dtype(dtype, name)
        ret = util.get_type(dtype)
        factor = ret[1] / tensor.type.bits
        bitwidth = tensor.type.bits
    else:
        ret = util.get_type(tensor.dtype)
        bitwidth = ret[1]
        dtype = ret[0] + str(bitwidth * factor)

    # derive the new shape
    ndim = len(tensor.shape)
    if axis > ndim:
        raise APIError("The pack dimension exceeds the number of dimensions")
    new_shape = []
    for i in range(0, ndim):
        if i == axis:
            new_shape.append(tensor.shape[i] / factor)
        else:
            new_shape.append(tensor.shape[i])

    # derive the packed tensor
    def assign_val(*indices):
        temp = local(0, name+"_temp", dtype)
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

def reduce_axis(min_, max_, name = "ra"):
    return _IterVar((min_, max_), name, 2)

def reducer(init, freduce, dtype = "int32"):
    def make_reduce(expr, axis, where = True, name = None, dtype = dtype):
        if not isinstance(axis, (tuple, list)):
            axis = [axis]
        stage = Stage.get_current()
        out = None
        name = util.get_name("reducer", name)
        if isinstance(init, (_expr.Expr, numbers.Number)):
            out = local(init, name, dtype)
            def reduce_body():
                with if_(where):
                    out[0] = freduce(expr, out[0])
                return out[0]
            stage.stmt_stack.append([])
            ret = reduce_body()
        else: # a list or tensor
            out = copy(init, name)
            def reduce_body():
                with if_(where):
                    new_out = freduce(expr, out)
                if not new_out is None:
                    copy_inplace(out, new_out)
                return out
            stage.stmt_stack.append([])
            ret = reduce_body()
        body = stage.pop_stmt()
        stage.input_stages.add(out.last_update)
        body = util.make_for(axis, body, 0)
        stage.axis_list += axis
        stage.emit(body)
        return ret

    return make_reduce

sum = reducer(0, lambda x, y: x + y)
max = reducer(min_value("float"), lambda x, y: _make.Max(x, y))

