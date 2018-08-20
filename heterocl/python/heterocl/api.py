from . import kernel as _kernel
from . import util
from . import types
from . import config
from . import api_util
from .tensor import Var, Tensor, TensorSlice, Operation
from .code_builder import CodeBuilder
from .resizer import Resizer, Downsizer, CastRemover
from .schedule import Schedule
from .dsl import *
from .util import HCLError
from tvm.api import _IterVar, decl_buffer, convert, min_value, select
from tvm.build_module import build as _build, lower as _lower
from tvm.ndarray import array, cpu
from tvm import var as _var
from tvm import schedule as _schedule
from tvm import _api_internal
from tvm import make as _make
from tvm import expr as _expr
from tvm import stmt as _stmt
import inspect
import numbers

def var(name = None, dtype = None):
  name = util.set_name("var", name)
  dtype = util.convert_dtype(dtype)

  return Var(_var(name = name, dtype = dtype))

def placeholder(shape, name = None, dtype = None):
  name = util.set_name("placeholder", name)
  dtype = util.convert_dtype(dtype)

  tensor = Tensor(shape, dtype, name)
  op = Operation(None, tensor, None)
  Operation.op_list.append(op)
  if len(CodeBuilder.current) != 0:
    raise HCLError("placeholder can only be used at the top level", inspect.stack()[1])

  return tensor

def compute(shape, fcompute, name = None, dtype = None):
  args = fcompute.__code__.co_varnames
  nargs = fcompute.__code__.co_argcount
  shape = CastRemover().mutate(shape)

  if not isinstance(shape, tuple):
    raise HCLError("The shape must be a tuple", inspect.stack()[1])

  # if nargs != len(shape):
  #   raise HCLError("The length of shape and the number of lambda args do not match", inspect.stack()[1])

  # create the returned tensor
  name = util.set_name("compute", name)
  dtype = util.convert_dtype(dtype)
  tensor = Tensor(shape, dtype, name)

  # get the used inputs and all indices
  lambda_ivs = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, len(shape))]
  inputs, indices, lhs, axis = api_util.compute_body(tensor, tensor, lambda_ivs, fcompute)

  # make the body
  body = util.make_for(indices, CodeBuilder.get(), 0)

  # additional process if this API is inside another CodeBuilder
  if len(CodeBuilder.current) != 0:
    api_util.in_builder_process(tensor, inputs, lhs)
  else:
    Schedule.stage_ops.append(tensor)

  Operation.op_list.append(Operation(inputs, tensor, body, indices + axis))

  return tensor

def local(init = 0, name = None, dtype = None):
  name = util.set_name("local", name)
  return compute((1,), lambda x: init, name, dtype)

def update(_tensor, fcompute, name = None):
  args = fcompute.__code__.co_varnames
  nargs = fcompute.__code__.co_argcount
  shape = _tensor.shape

  if not isinstance(shape, tuple):
    raise HCLError("The shape must be a tuple", inspect.stack()[1])
  if nargs != len(shape):
    raise HCLError("The length of shape and the number of lambda args do not match", inspect.stack()[1])

  # create the returned tensor
  name = util.set_name("update", name)
  tensor = Tensor((1,), "int32", name)

  # get the used inputs and all indices
  lambda_ivs = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]
  inputs, indices, lhs, axis = api_util.compute_body(_tensor, tensor, lambda_ivs, fcompute)
  inputs.append(_tensor)

  # make the body
  body = util.make_for(indices, CodeBuilder.get(), 0)

  # additional process if this API is inside another CodeBuilder
  if len(CodeBuilder.current) != 0:
    api_util.in_builder_process(tensor, inputs, lhs)
  else:
    Schedule.stage_ops.append(tensor)

  Operation.op_list.append(Operation(inputs, tensor, body, indices + axis))

  return tensor

# copy a tensor
def copy_from(_tensor, name = None):
  name = util.set_name("copy", name)

  indices = [_IterVar((0, _tensor.shape[n]), "copy_i" + str(n), 0) for n in range(0, len(_tensor.shape))]
  tensor = Tensor(_tensor.shape, _tensor.dtype, name)

  index, _, _ = util.get_index(_tensor.shape, indices, 0)
  body = _make.Store(tensor.buf.data, _make.Cast(_tensor.dtype, _tensor[tuple(indices)]), index)
  body = util.make_for(indices, body, 0)

  if len(CodeBuilder.current) != 0:
    api_util.in_builder_process(tensor, [_tensor], [])
  else:
    Schedule.stage_ops.append(tensor)

  Operation.op_list.append(Operation([_tensor], tensor, body, indices))

  return tensor

def update_from(_tensor, _from, name = None):
  name = util.set_name("update", name)

  indices = [_IterVar((0, _tensor.shape[n]), "update_i" + str(n), 0) for n in range(0, len(_tensor.shape))]
  tensor = Tensor((1,), "int32", name)
  _tensor.last_update = tensor

  index, _, _ = util.get_index(_tensor.shape, indices, 0)
  body = _make.Store(_tensor.buf.data, _make.Cast(_tensor.dtype, _from[tuple(indices)]), index)
  body = util.make_for(indices, body, 0)

  if len(CodeBuilder.current) != 0:
    api_util.in_builder_process(tensor, [_tensor, _from], [])
  else:
    Schedule.stage_ops.append(tensor)

  Operation.op_list.append(Operation([_tensor, _from], tensor, body, indices))

  return tensor

def block(fblock, name = None):
  raise DeprecationWarning("block is deprecated")

class stage():

  def __init__(self, name = None):
    self.name = util.set_name("stage", name)
    self.cb = None
    self.tensor = Tensor((1,), "int32", self.name)

  def __enter__(self):
    self.cb = CodeBuilder(self.name)
    self.cb.__enter__()
    return self.tensor

  def __exit__(self, etype, val, tb):
    inputs = list(self.cb.last_stages.union(self.cb.tensors))
    lhs = self.cb.lhs
    for t in lhs:
      t.last_update = self.tensor
    self.cb.__exit__(etype, val, tb)
    self.tensor.var_dict = CodeBuilder.var_dict[-1]
    axis = CodeBuilder.axis_list[-1]
    body = CodeBuilder.get()

    if len(CodeBuilder.current) != 0:
      api_util.in_builder_process(self.tensor, inputs, lhs)
    else:
      Schedule.stage_ops.append(self.tensor)

    Operation.op_list.append(Operation(inputs, self.tensor, body, axis))

def mut_compute(shape, fcompute, name = None):
  code = fcompute.__code__
  args = code.co_varnames
  nargs = code.co_argcount

  name = util.set_name("vector", name)
  tensor = Tensor((1,), "int32", name)

  assert (len(shape) == nargs), "fcompute does not match output dimension"

  indices = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, len(shape))]
  var_list = [i.var for i in indices]

  with CodeBuilder(name) as cb:
    fcompute(*var_list)
    for t in cb.lhs:
      t.last_update = tensor
    inputs = list(cb.last_stages.union(cb.tensors))
  tensor.var_dict = CodeBuilder.get_var_dict()
  axis = CodeBuilder.get_axis()
  ret = CodeBuilder.get()
  body = util.make_for(indices, ret, 0)

  if len(CodeBuilder.current) != 0:
    api_util.in_builder_process(tensor, inputs, cb.lhs)
  else:
    Schedule.stage_ops.append(tensor)

  Operation.op_list.append(Operation(inputs, tensor, body, indices + axis))

  return tensor

def function(shapes, fkernel, ret_void = True, dtypes = [], ret_dtype = None, name = None):
  code = fkernel.__code__
  names = code.co_varnames
  nargs = code.co_argcount
  assert len(shapes) == nargs, "The number of shapes must be the same as the number of arguments"
  assert len(dtypes) <= nargs, "The number of dtypes should not be greater than the number of arguments"

  name = "kernel" + str(util.KID) if name is None else name
  util.KID += 1

  inputs = []
  args = []
  arg_type = []
  for i in range(nargs):
    dtype = config.init_dtype
    if i <= len(dtypes) - 1:
      dtype = util.convert_dtype(dtypes[i])
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
    ts = cb.tensors
  ret_dtype = config.init_dtype if ret_dtype is None else ret_dtype
  ret_dtype = util.convert_dtype(ret_dtype)

  _ret_void = _make.UIntImm("uint1", 1) if ret_void else _make.UIntImm("uint1", 0)

  var_dict = CodeBuilder.var_dict[-1]
  axis = CodeBuilder.axis_list[-1]
  body = _make.KernelDef(args, CodeBuilder.get(), _ret_void, ret_dtype, name)
  p = _kernel.KernelTensor(arg_type, name, ret_void, ret_dtype, body)
  p.var_dict = var_dict

  op = Operation(ts, p, body, axis)
  Operation.op_list.append(op)

  return p

def cast(dtype, expr):
  dtype = util.convert_dtype(dtype)
  return _make.Cast(dtype, expr)

def resize(inputs, dtype):
  from_vars = []
  to_vars = []
  assert isinstance(dtype, (str, types.Type)), "Wrong input to resize data type"
  dtype = util.convert_dtype(dtype)
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  for i in inputs:
    if isinstance(i, Var):
      from_vars.append(i.var)
      new_var = _var(i.name, dtype)
      i.var = new_var
      to_vars.append(new_var)
    else:
      from_vars.append(i.buf.data)
      from_vars.append(i.buf)
      new_buf = decl_buffer(i.shape, dtype, i.name)
      i.buf = new_buf
      i.dtype = dtype
      to_vars.append(new_buf.data)
      to_vars.append(new_buf)
  op_list = Operation.op_list
  assert len(op_list) > 0, "Resize must be used before create_schedule!!"
  bodies = Resizer(from_vars, to_vars, dtype).enter(op_list)
  for i in range(len(op_list)):
    op_list[i].body = bodies[i]
  builders = CodeBuilder.current
  if len(builders) > 0:
    Resizer(from_vars, to_vars, dtype).enter_cb(CodeBuilder)

def downsize(inputs, dtype):
  assert isinstance(dtype, (types.Int, types.UInt))
  resize(inputs, dtype)

def quantize(inputs, dtype):
  assert isinstance(dtype, (types.Fixed, types.UFixed))
  resize(inputs, dtype)

def simdtype(inputs, dt_var):
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

def create_schedule(t):
  for op in Operation.op_list:
    if op.inputs is None:
      if op.body is None: #placeholder
        p = op.output
        p.tensor = _api_internal._Placeholder(p.buf.shape, p.dtype, p.name)
      else: #kernel
        p = op.output
        o_buf = [p.buf]
        p.tensor = _api_internal._ExternOp(p.name, "", op.axis, [], [], o_buf, op.body).output(0)
    else:
      i = op.inputs
      p = op.output
      for _i in i:
        assert not _i.tensor is None
      i_tensor = [_i.tensor for _i in i]
      i_buf = [_i.buf for _i in i]
      o_buf = [p.buf]
      p.tensor = _api_internal._ExternOp(p.name, "", op.axis, i_tensor, i_buf, o_buf, op.body).output(0)

  Operation.op_list = []

  if not isinstance(t, list):
    t = [t]
  ops = [t_.op for t_ in t]
  return Schedule(_schedule.create_schedule(ops))

def make_schedule(inputs, f):
  ret_sch = f(*inputs)
  print Schedule.stage_ops
  for op in Schedule.stage_ops:
    f.__setattr__(op.name, op)
  return create_schedule(ret_sch)

def lower(schedule, inputs):
  new_inputs = []
  for i in inputs:
    if isinstance(i, Tensor):
      new_inputs.append(i.tensor)
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
  def make_reduce(expr, axis, where = True):
    if not isinstance(axis, (tuple, list)):
      axis = [axis]
    cb = CodeBuilder.current[-1]
    out = None
    name = util.set_name("reducer", None)
    if isinstance(init, (_expr.Expr, numbers.Number)):
      out = local(init, name, dtype)
      def reduce_body():
        with if_(where):
          out[0] = freduce(expr, out[0])
        return out[0]
      with CodeBuilder():
        ret = reduce_body()
    else: # a list or tensor
      out = copy_from(init, name)
      def reduce_body():
        with if_(where):
          new_out = freduce(expr, out)
        if not new_out is None:
          copy_inplace(out, new_out)
        return out
      with CodeBuilder():
        ret = reduce_body()
    body = CodeBuilder.get()
    body = util.make_for(axis, body, 0)
    CodeBuilder.axis_list[-1] += axis
    cb.emit(body)
    cb.tensors.add(out)
    return ret

  return make_reduce

def asarray(arr, dtype = None, ctx = cpu(0)):
  #if dtype is None:
  #  dtype = arr.dtype
  dtype = util.convert_dtype(dtype)
  return array(arr, dtype, ctx)

sum = reducer(0, lambda x, y: x + y)
max = reducer(min_value("float"), lambda x, y: _make.Max(x, y))
