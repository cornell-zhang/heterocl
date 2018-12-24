"""This module contains all HeteroCL APIs"""
import inspect
import numbers
from ordered_set import OrderedSet
from .tvm.api import _IterVar, decl_buffer, convert, min_value
from tvm.build_module import build as _build, lower as _lower
from tvm.ndarray import array, cpu
from .tvm import _api_internal as tvm_api
from tvm import schedule as _schedule
from tvm import make as _make
from tvm import expr as _expr
from tvm import stmt as _stmt
from . import util
from . import types
from . import config
from . import api_util
from .tensor import Scalar, Tensor, TensorSlice
from .schedule import Stage
from .schedule import Schedule
from .module import Module
from .dsl import *
from .function import *
from .debug import APIError

def init(init_dtype="int32"):
    """Initialze a HeteroCL environment with configurations.

    This API must be called each time the users write an application.
    Within the same HeteroCL environment, users can try different
    combinations of customization primitives.

    Examples
    --------
    .. code-block:: python

        # app 1
        hcl.init()
        A = hcl.placeholder(...)
        B = hcl.placeholder(...)
        def app1(A, B):
            # define the algorithm for app 1
        s = hcl.create_scheme([A, B], app1)
        # apply customization primitives
        f1 = hcl.build(s)
        # execute f1

        # app 2 - initialize again with a different data type
        hcl.init(hcl.Float())
        A = hcl.placeholder(...)
        B = hcl.placeholder(...)
        C = hcl.placeholder(...)
        def app2(A, B, C):
            # define the algorithm for app 2
        s = hcl.create_scheme([A, B, C], app2)
        f2 = hcl.build(s)
        # execute f2

    Parameters
    ----------
    init_dtype : Type, optional
        The default data type for each variables
    """
    # set the configurations
    config.init_dtype = init_dtype
    # initialize global variables
    Schedule.stage_ops = []
    Schedule.last_stages = set([])
    Function.current = None

def var(name=None, dtype=None):
    """Construct a HeteroCL variable.

    Parameters
    ----------
    name : str, optional
        The name of the variable

    dtype : Type, optional
        The data type of the variable

    Returns
    -------
    Scalar
    """
    name = util.get_name("var", name)
    dtype = util.get_dtype(dtype)

    return Scalar(tvm_api._Var(name, dtype))

def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs.

    If the shape is an empty tuple, the returned value is a scalar.

    Examples
    --------
    .. code-block:: python

        # scalar - cannot be updated
        a = hcl.placeholder((), "a")
        # 1-dimensional tensor - can be updated
        A = hcl.placeholder((1,), "A")

    Parameters
    ----------
    shape : tuple
        The shape of the placeholder.

    name : str, optional
        The name of the placeholder.

    dtype : Type, optional
        The data type of the placeholder

    Returns
    -------
    Scalar or Tensor
    """
    name = util.get_name("placeholder", name)
    dtype = util.get_dtype(dtype)

    if shape == ():
        return Scalar(tvm_api._Var(name, dtype))
    else:
        tensor = Tensor(shape, dtype, name)
        tensor.tensor = tvm_api._Placeholder(tensor.buf.shape, dtype, name)

        # placeholder is also a stage
        stage = Stage(name)
        stage._op = tensor.tensor
        stage._buf = tensor._buf
        tensor.first_update = stage
        tensor.last_update = stage
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
    if not callable(fcompute):
        raise APIError("The construction rule must be callable")

    # properties for the returned tensor
    shape = util.CastRemover().mutate(shape)
    name = util.get_name("compute", name)

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
    lambda_ivs = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, len(shape))]

    # call the helper function that returns a new tensor
    tensor = api_util.compute_body(name, lambda_ivs, fcompute, shape, dtype)

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
    # check API correctness
    if not callable(fcompute):
        raise APIError("The construction rule must be callable")

    # properties for the returned tensor
    shape = tensor.shape
    name = util.get_name("update", name)

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
    lambda_ivs = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]

    # call the helper function that updates the tensor
    api_util.compute_body(name, lambda_ivs, fcompute, tensor=tensor)

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
    code = fcompute.__code__
    args = code.co_varnames
    nargs = code.co_argcount

    name = util.get_name("vector", name)

    #assert (len(shape) == nargs), "fcompute does not match output dimension"

    indices = [_IterVar((0, domain[n]), args[n], 0) for n in range(0, len(domain))]
    var_list = [i.var for i in indices]

    with Stage(name) as stage:
        stage.stmt_stack.append([])
        fcompute(*var_list)
        ret = stage.pop_stmt()
        stage.emit(util.make_for(indices, ret, 0))
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


# For first dimension only
# You need to specify either factor or dtype
# If the factor is specified, the dtype will be inferred automatically
def unpack(tensor, axis = 0, factor = None, name = None, dtype = None):
    name = util.get_name("unpack", name)

    if factor is None:
        name_ = name if Stage.get_len() == 0 else Stage.get_current().name_with_prefix + "." + name
        dtype = util.get_dtype(dtype, name_)
        ret = util.get_type(dtype)
        factor = tensor.type.bits / ret[1]
        bitwidth = ret[1]
    else:
        ret = util.get_type(tensor.dtype)
        assert len(ret) == 2
        bitwidth = ret[1]/factor
        dtype = ret[0] + str(bitwidth)

    dim = len(tensor.shape)
    new_shape = []
    for i in range(0, dim):
        if i == axis:
            new_shape.append(tensor.shape[i] * factor)
        else:
            new_shape.append(tensor.shape[i])

    """
    def assign_val(val):
        temp = local(0, dtype = dtype, name = name + "_temp")
        temp[0][bitwidth : 0] = val
        return temp[0]
    """

    def assign_val(*indices):
        temp = local(0, dtype = dtype, name = name + "_temp")
        new_indices = []
        for i in range(0, dim):
            if i == axis:
                new_indices.append(indices[i]/factor)
            else:
                new_indices.append(indices[i])
        index = indices[axis]
        temp[0][bitwidth : 0] = tensor[tuple(new_indices)][(index%factor+1)*bitwidth : (index%factor)*bitwidth]
        return temp[0]

    #return compute(tuple(new_shape), lambda x: assign_val(tensor[x/factor][(x%factor+1)*bitwidth : (x%factor)*bitwidth]), name, dtype)
    return compute(tuple(new_shape), lambda *indices: assign_val(*indices), name, dtype)

def pack(tensor, axis=0, factor=None, name=None, dtype=None):
    name = util.get_name("pack", name)

    if factor is None:
        assert dtype is not None
        name_ = name if Stage.get_len() == 0 else Stage.get_current().name_with_prefix + "." + name
        dtype = util.get_dtype(dtype, name)
        ret = util.get_type(dtype)
        factor = ret[1] / tensor.type.bits
        bitwidth = tensor.type.bits
    else:
        ret = util.get_type(tensor.dtype)
        assert len(ret) == 2
        bitwidth = ret[1]
        dtype = ret[0] + str(bitwidth * factor)

    dim = len(tensor.shape)
    new_shape = []
    for i in range(0, dim):
        if i == axis:
            new_shape.append(tensor.shape[i] / factor)
        else:
            new_shape.append(tensor.shape[i])

    def assign_val(*indices):
        temp = local(0, dtype = dtype)
        with for_(0, factor) as i:
            new_indices = []
            for j in range(0, dim):
                if j == axis:
                    new_indices.append(indices[j]*factor+i)
                else:
                    new_indices.append(indices[j])
            temp[0][bitwidth*(i+1) : bitwidth*i] = tensor[tuple(new_indices)]
        return temp[0]

    return compute(tuple(new_shape), lambda *indices: assign_val(*indices), name, dtype)

def module(shapes, dtypes=None, ret_dtype=None, name=None):
    """
    Add a HeteroCL module from exsiting Python function.
    This is a decorator
    """
    def decorator(fmodule, shapes=shapes, dtypes=dtypes, ret_dtype=ret_dtype, name=name):
        name = name if name is not None else fmodule.__name__
        code = fmodule.__code__
        names = code.co_varnames
        nargs = code.co_argcount

        with Stage(name) as s:
            # prepare names
            new_names = [s.name_with_prefix + "." + name_ for name_ in names]
            # prepare dtypes
            if dtypes is None:
                dtypes = []
                for name_ in new_names:
                    dtypes.append(util.get_dtype(None, name_))
            elif isinstance(dtypes, list):
                if len(dtypes) != nargs:
                    raise APIError("The number of data types does not match the number of arguments")
                for name_ in new_names:
                    dtypes[i] = util.get_dtype(dtype[i], name_)
            else:
                dtype = util.get_dtype(dtypes)
                dtypes = []
                for name_ in new_names:
                    dtypes.append(util.get_dtype(dtype, name_))
            ret_dtype = util.get_dtype(ret_dtype, s.name_with_prefix)
            # prepare inputs for IR generation
            inputs = []
            inputs_tvm = []
            for shape, name_, dtype in zip(shapes, new_names, dtypes):
                if shape == ():
                    var_ = var(name_, dtype)
                    inputs.append(var_)
                    inputs_tvm.append(var_.var)
                else:
                    placeholder_ = placeholder(shape, name_, dtype)
                    inputs.append(placeholder_)
                    inputs_tvm.append(placeholder_.buf.data)

            s.ret_dtype = ret_dtype
            fmodule(*inputs)
            lhs = []
            for tensor in s.lhs_tensors:
                try:
                    lhs.append(inputs.index(tensor))
                except ValueError:
                    pass
            ret_void = _make.UIntImm("uint1", 0) if s.has_return else _make.UIntImm("uint1", 1)
            body = s.pop_stmt()
            s.stmt_stack.append([])
            s.emit(_make.KernelDef(inputs_tvm, body, ret_void, ret_dtype, name))
            for name_, i in zip(names, inputs):
                s.var_dict[name_] = i
            s.input_stages.clear()

        return Module(shapes, names, name, not s.has_return, lhs, ret_dtype)
    return decorator

def cast(dtype, expr):
    dtype = util.get_dtype(dtype)
    return _make.Cast(dtype, expr)

def create_schedule(inputs, f=None):
    if f is not None:
        Schedule.stage_ops = []
        Schedule.last_stages = OrderedSet([])
        ret = f(*inputs)
        if ret is not None:
            if isinstance(ret, tuple):
                inputs += list(ret)
            else:
                inputs.append(ret)
        Function.current = None
        for op in Schedule.stage_ops:
            f.__setattr__(op.name, op)
    t = Schedule.last_stages
    ops = [t_._op.op for t_ in t]
    return Schedule(_schedule.create_schedule(ops), inputs)

def create_schedule_from_scheme(func):
    Function.current = func
    for i in func.inputs:
        if isinstance(i, Tensor):
            i.var_dict = {}
            i.last_update = i.first_update
    return create_schedule(func.inputs, func.func)

def create_scheme(inputs, f):
    f(*inputs)
    func = Function(inputs, f)
    for op in Schedule.stage_ops:
        f.__setattr__(op.name, op)
    return func

def lower(schedule):
    new_inputs = []
    for i in schedule.inputs:
        if isinstance(i, Tensor):
            new_inputs.append(i.tensor)
        elif isinstance(i, Stage):
            new_inputs.append(i._op)
        else:
            new_inputs.append(i.var)
    return _lower(schedule.sch, new_inputs, simple_mode = True)

def build(schedule, target=None):
    new_inputs = []
    for i in schedule.inputs:
        if isinstance(i, Tensor):
            new_inputs.append(i.tensor)
        else:
            new_inputs.append(i.var)

    return _build(schedule.sch, new_inputs, target=target)

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

def asarray(arr, dtype = None, ctx = cpu(0)):
    #if dtype is None:
    #  dtype = arr.dtype
    dtype = util.get_dtype(dtype)
    return array(arr, dtype, ctx)

def get_bits(dtype):
    dtype = util.get_dtype(dtype)
    ret = util.get_type(dtype)
    return ret[1]

def get_data_type(dtpye):
    dtype = util.get_dtype(dtype)
    ret = util.get_type(dtype)
    return ret[0]

def get_fracs(dtype):
    dtype = util.get_dtype(dtype)
    ret = util.get_type(dtype)
    return ret[2]

sum = reducer(0, lambda x, y: x + y)
max = reducer(min_value("float"), lambda x, y: _make.Max(x, y))

