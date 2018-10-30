import inspect
import numbers
from .tvm.api import _IterVar, decl_buffer, convert, min_value, select as _select
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

def init():
    Schedule.stage_ops = []
    Schedule.last_stages = set([])

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

    # This should replace the old one!!
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

def block(fblock, name = None):
    raise DeprecationWarning("block is deprecated")

def mut_compute(shape, fcompute, name = None):
    code = fcompute.__code__
    args = code.co_varnames
    nargs = code.co_argcount

    name = util.get_name("vector", name)
    tensor = Tensor((), "int32", name)

    assert (len(shape) == nargs), "fcompute does not match output dimension"

    indices = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, len(shape))]
    var_list = [i.var for i in indices]

    with Stage(name) as stage:
        fcompute(*var_list)
        ret = stage.pop_stmt()
        stage.get_stmt_stack().append([])
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

"""
def function(shapes, fkernel, ret_void = True, dtypes = [], ret_dtype = None, name = None):
    code = fkernel.__code__
    names = code.co_varnames
    nargs = code.co_argcount
    assert len(shapes) == nargs, "The number of shapes must be the same as the number of arguments"
    assert len(dtypes) <= nargs, "The number of dtypes should not be greater than the number of arguments"

    name = util.get_name("kernel", name)
    #name = "kernel" + str(util.KID) if name is None else name
    #util.KID += 1

    inputs = []
    args = []
    arg_type = []
    for i in range(nargs):
        dtype = config.init_dtype
        if i <= len(dtypes) - 1:
            dtype = util.get_dtype(dtypes[i])
        if isinstance(shapes[i], tuple):
            p = placeholder(shapes[i], names[i], dtype)
            inputs.append(p)
            args.append(p.buf.data)
            arg_type.append(1)
        elif isinstance(shapes[i], int):
            assert shapes[i] == 1, "A var must be a scalar"
            v = var(names[i], dtype)
            inputs.append(v)
            args.append(v.var)
            arg_type.append(0)
        else:
            raise ValueError("Unknown shape" + str(shape[i]))

    with CodeBuilder() as cb:
        fkernel(*inputs)
        #ts = cb.tensors
        inputs = list(cb.last_stages.union(cb.tensors))
    ret_dtype = config.init_dtype if ret_dtype is None else ret_dtype
    ret_dtype = util.get_dtype(ret_dtype)

    _ret_void = _make.UIntImm("uint1", 1) if ret_void else _make.UIntImm("uint1", 0)

    var_dict = cb.var_dict
    axis = cb.axis_list
    body = _make.KernelDef(args, CodeBuilder.get(), _ret_void, ret_dtype, name)
    tensor = _kernel.KernelTensor(arg_type, name, ret_void, ret_dtype, body)
    tensor.var_dict = var_dict

    api_util.make_extern_op(inputs, tensor, axis, body)

    return p
"""

def cast(dtype, expr):
    dtype = util.get_dtype(dtype)
    return _make.Cast(dtype, expr)

def resize(inputs, dtype):
    raise DeprecationWarning("resize is deprecated")

def downsize(inputs, dtype):
    raise DeprecationWarning("resize is deprecated")

def quantize(inputs, dtype):
    raise DeprecationWarning("resize is deprecated")

def simdtype(inputs, dt_var):
    """
    from_vars = []
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    for i in inputs:
        if isinstance(i, tensor.Var):
            from_vars.append(i.var)
        else:
            from_vars.append(i.buf.data)
    op_list = tensor.Operation.op_list
    assert len(op_list) > 0, "Downsize must be used before create_schedule!!"
    bodies = Downsizer(from_vars, dt_var.var).enter(op_list)
    for i in range(len(op_list)):
        op_list[i].body = bodies[i]
    """

def create_schedule(t):
    t = Schedule.last_stages
    #if not isinstance(t, list):
    #    t = [t]
    ops = [t_._op.op for t_ in t] #TODO
    return Schedule(_schedule.create_schedule(ops))

def make_schedule(inputs, f):
    Schedule.stage_ops = []
    Schedule.last_stages = set([])
    ret_sch = f(*inputs)
    Function.current = None
    for op in Schedule.stage_ops:
        f.__setattr__(op.name, op)
    return create_schedule(ret_sch)

def make_schedule_from_scheme(func):
    Function.current = func
    for i in func.inputs:
        if isinstance(i, Tensor):
            i.var_dict = {}
            i.last_update = i.first_update
    return make_schedule(func.inputs, func.func)

def make_scheme(inputs, f):
    f(*inputs)
    func = Function(inputs, f)
    for op in Schedule.stage_ops:
        f.__setattr__(op.name, op)
    return func

def lower(schedule, inputs):
    new_inputs = []
    for i in inputs:
        if isinstance(i, Tensor):
            new_inputs.append(i.tensor)
        elif isinstance(i, Stage):
            new_inputs.append(i._op)
        else:
            new_inputs.append(i.var)
    return _lower(schedule.sch, new_inputs, simple_mode = True)

def build(schedule, inputs, target=None):
    new_inputs = []
    for i in inputs:
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
        cb = Stage.get_cb()
        out = None
        name = util.get_name("reducer", name)
        if isinstance(init, (_expr.Expr, numbers.Number)):
            out = local(init, name, dtype)
            def reduce_body():
                with if_(where):
                    out[0] = freduce(expr, out[0])
                return out[0]
            with Stage() as s:
                ret = reduce_body()
        else: # a list or tensor
            out = copy_from(init, name)
            def reduce_body():
                with if_(where):
                    new_out = freduce(expr, out)
                if not new_out is None:
                    copy_inplace(out, new_out)
                return out
            with Stage() as s:
                ret = reduce_body()
        body = s._op.op.body
        cb.last_substages.remove(s)
        cb.input_stages.add(out.last_update)
        body = util.make_for(axis, body, 0)
        cb.axis_list += axis
        cb.emit(body)
        return ret

    return make_reduce

def asarray(arr, dtype = None, ctx = cpu(0)):
    #if dtype is None:
    #  dtype = arr.dtype
    dtype = util.get_dtype(dtype)
    return array(arr, dtype, ctx)

def cast(dtype, val):
    dtype = util.get_dtype(dtype)
    return _make.Cast(dtype, val)

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

