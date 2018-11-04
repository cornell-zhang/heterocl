"""This module contains all HeteroCL APIs"""
import inspect
import numbers
from .tvm.api import _IterVar, decl_buffer, convert, min_value
from tvm.build_module import build as _build, lower as _lower
from tvm.ndarray import array, cpu
from .tvm import _api_internal as tvm_api
from tvm import schedule as _schedule
from tvm import make as _make
from tvm import expr as _expr
from tvm import stmt as _stmt
#from . import kernel as _kernel
from . import util
from . import types
from . import config
from . import api_util
from .tensor import Var, Tensor, TensorSlice
from .schedule import Stage
from .resizer import Resizer, Downsizer, CastRemover
from .schedule import Schedule
from .dsl import *
from .function import *
from .debug import APIError

def init(init_dtype="int32"):
    """Initialze a HeteroCL environment with configurations

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
    Var
    """
    name = util.get_name("var", name)
    dtype = util.get_dtype(dtype)

    return Var(tvm_api._Var(name, dtype))

def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs.

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
    Tensor
    """
    name = util.get_name("placeholder", name)
    dtype = util.get_dtype(dtype)

    tensor = Tensor(shape, dtype, name)
    tensor.tensor = tvm_api._Placeholder(tensor.buf.shape, dtype, name)

    stage = Stage(name)
    stage._op = tensor.tensor
    stage._buf = tensor._buf
    tensor.first_update = stage
    tensor.last_update = stage

    return tensor

def compute(shape, fcompute, name = None, dtype = None):
    args = fcompute.__code__.co_varnames
    nargs = fcompute.__code__.co_argcount
    shape = CastRemover().mutate(shape)

    if not isinstance(shape, tuple):
        raise HCLError("The shape must be a tuple", inspect.stack()[1])

    # if nargs != len(shape):
    #       raise HCLError("The length of shape and the number of lambda args do not match", inspect.stack()[1])

    # create the returned tensor
    name = util.get_name("compute", name)
    dtype = util.get_dtype(dtype, name)

    # get the used inputs and all indices
    lambda_ivs = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, len(shape))]
    tensor = api_util.compute_body(name, lambda_ivs, fcompute, shape, dtype)

    return tensor

def local(init = 0, name = None, dtype = None):
    name = util.get_name("local", name)
    return compute((1,), lambda x: init, name, dtype)

# Do not return anything
def update(_tensor, fcompute, name = None):
    args = fcompute.__code__.co_varnames
    nargs = fcompute.__code__.co_argcount
    shape = _tensor.shape

    if not isinstance(shape, tuple):
        raise HCLError("The shape must be a tuple", inspect.stack()[1])
    if nargs != len(shape):
        raise HCLError("The length of shape and the number of lambda args do not match", inspect.stack()[1])

    name = util.get_name("update", name)

    lambda_ivs = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]
    api_util.compute_body(name, lambda_ivs, fcompute, tensor=_tensor)

# copy a tensor
def copy_from(_tensor, name = None):
    name = util.get_name("copy", name)

    indices = [_IterVar((0, _tensor.shape[n]), "copy_i" + str(n), 0) for n in range(0, len(_tensor.shape))]

    with Stage(name, _tensor.dtype, _tensor.shape) as stage:
        tensor = Tensor(_tensor.shape, _tensor.dtype, name, stage._buf)
        stage.lhs_tensors.add(tensor)
        for t in stage.lhs_tensors:
            t.last_update = stage

        index, _, _ = util.get_index(_tensor.shape, indices, 0)
        body = _make.Store(tensor.buf.data, _make.Cast(_tensor.dtype, _tensor[tuple(indices)]), index)
        body = util.make_for(indices, body, 0)

    tensor._tensor = stage._op

    return tensor

def update_from(_tensor, _from, name = None):
    name = util.get_name("update", name)

    indices = [_IterVar((0, _tensor.shape[n]), "update_i" + str(n), 0) for n in range(0, len(_tensor.shape))]

    with Stage(name) as stage:
        stage.input_stages.add(_tensor.last_update)
        stage.lhs_tensors.add(_tensor)
        for t in stage.lhs_tensors:
            t.last_update = stage

        index, _, _ = util.get_index(_tensor.shape, indices, 0)
        body = _make.Store(_tensor.buf.data, _make.Cast(_tensor.dtype, _from[tuple(indices)]), index)
        body = util.make_for(indices, body, 0)

def mut_compute(shape, fcompute, name = None):
    code = fcompute.__code__
    args = code.co_varnames
    nargs = code.co_argcount

    name = util.get_name("vector", name)

    assert (len(shape) == nargs), "fcompute does not match output dimension"

    indices = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, len(shape))]
    var_list = [i.var for i in indices]

    with Stage(name) as stage:
        stage.stmt_stack.append([])
        fcompute(*var_list)
        ret = stage.pop_stmt()
        stage.emit(util.make_for(indices, ret, 0))
        stage.axis_list = indices + stage.axis_list

# For first dimension only
# You need to specify either factor or dtype
# If the factor is specified, the dtype will be inferred automatically
def unpack(tensor, axis = 0, factor = None, name = None, dtype = None):
    name = util.get_name("unpack", name)

    if factor is None:
        dtype = util.get_dtype(dtype, name)
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

    def assign_val(val):
        temp = local(0, dtype = dtype, name = name + "_temp")
        temp[0][bitwidth : 0] = val
        return temp[0]

    return compute(tuple(new_shape), lambda x: assign_val(tensor[x/factor][(x%factor+1)*bitwidth : (x%factor)*bitwidth]), name, dtype)

def pack(tensor, axis = 0, factor = None, name = None, dtype = None):
    name = util.get_name("pack", name)

    if factor is None:
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

    def assign_val(index):
        temp = local(0, dtype = dtype)
        with for_(0, factor) as i:
            temp[0][bitwidth*(i+1) : bitwidth*i] = tensor[index*factor + i]
        return temp[0]

    return compute(tuple(new_shape), lambda x: assign_val(x), name, dtype)

def cast(dtype, expr):
    dtype = util.get_dtype(dtype)
    return _make.Cast(dtype, expr)

def create_schedule(inputs, f=None):
    if f is not None:
        Schedule.stage_ops = []
        Schedule.last_stages = set([])
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
            out = copy_from(init, name)
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

