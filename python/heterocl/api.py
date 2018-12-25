"""This module contains all HeteroCL APIs"""
import inspect
from ordered_set import OrderedSet
from .tvm.api import _IterVar, decl_buffer, convert, min_value
from tvm.build_module import build as _build, lower as _lower
from tvm.ndarray import array, cpu
from .tvm import _api_internal as tvm_api
from .tvm import schedule as _schedule
from .tvm import make as _make
from .tvm import expr as _expr
from .tvm import stmt as _stmt
from . import util
from . import types
from . import config
from .tensor import Scalar, Tensor, TensorSlice
from .schedule import Stage, Schedule
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


