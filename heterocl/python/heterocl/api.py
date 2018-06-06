from . import tensor
from . import kernel as _kernel
from . import util
from . import types
from . import config
from .code_builder import CodeBuilder
from .resizer import Resizer, Downsizer
from .schedule import Schedule
from .dsl import *
from tvm.api import _IterVar, decl_buffer, convert
from tvm.build_module import build as _build, lower as _lower
from tvm.ndarray import array, cpu
from tvm import var as _var
from tvm import schedule as _schedule
from tvm import placeholder as _placeholder
from tvm import _api_internal
from tvm import make as _make
from tvm import expr as _expr
from tvm import stmt as _stmt
import inspect
import ast
import numbers


def var(name = None, dtype = None):
  name = "var" + str(util.VID) if name is None else name
  util.VID += 1

  dtype = config.init_dtype if dtype is None else dtype
  dtype = util.convert_dtype(dtype)
  return tensor.Var(_var(name = name, dtype = dtype))

def placeholder(shape, name = None, dtype = None):
  name = "placeholder" + str(util.PID) if name is None else name
  util.PID += 1

  dtype = config.init_dtype if dtype is None else dtype
  dtype = util.convert_dtype(dtype)
  builder = CodeBuilder.current
  p = tensor.Tensor(shape, dtype, name)
  op = tensor.Operation(None, p, None)
  tensor.Operation.op_list.append(op)
  if len(builder) == 0:
    return p
  else:
    builder[-1].emit(lambda x: _make.Allocate(p.buf.data, dtype, shape, util.true(), x))
    CodeBuilder.var_dict[-1][name] = p
    return p

def local(init = 0, name = None, dtype = None):
  name = "local" + str(util.LID) if name is None else name
  util.LID += 1

  dtype = config.init_dtype if dtype is None else dtype
  dtype = util.convert_dtype(dtype)
  builder = CodeBuilder.current
  assert len(builder) != 0, "hcl.local must be used inside a code builder"
  p = tensor.Tensor((1,), dtype, name)
  builder[-1].emit(lambda x: _make.Allocate(p.buf.data, dtype, (1,), util.true(), x))
  CodeBuilder.var_dict[-1][name] = p
  p[0] = init
  op = tensor.Operation(None, p, None)
  tensor.Operation.op_list.append(op)
  return p

def compute(shape, inputs, fcompute, name = None, dtype = None):
  code = fcompute.__code__
  args = code.co_varnames
  nargs = code.co_argcount
  shape = list(shape)

  assert(nargs == len(shape)), "The shape and the number of lambda args do not match"

  name = "compute" + str(util.CID) if name is None else name
  util.CID += 1

  dtype = config.init_dtype if dtype is None else dtype
  dtype = util.convert_dtype(dtype)

  lambda_args = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]
  var_list = [i.var for i in lambda_args]
  body = None
  p = tensor.Tensor(shape, dtype, name)

  with CodeBuilder():
    ret = fcompute(*var_list)
  p.var_dict = CodeBuilder.var_dict[-1]
  axis = CodeBuilder.axis_list[-1]

  if isinstance(ret, tensor.TensorSlice):
    ret = ret.asnode()
    indices = lambda_args
    index, _, _ = util.get_index(shape, indices, 0)
    body = _make.Store(p.buf.data, _make.Cast(dtype, ret), index)
  elif isinstance(ret, tensor.Tensor):
    ivs = [_IterVar((0, ret.shape[i]), "red_var" + str(i), 0) for i in range(0, len(ret.shape))]
    indices = []
    rid = 0
    for iv in lambda_args:
      if iv.var.name == "_":
        indices.append(ivs[rid])
        rid += 1
      else:
        indices.append(iv)
    index, _, _ = util.get_index(shape, indices, 0)
    inner, _, _ = util.get_index(ret.shape, ivs, 0)
    body = _make.Store(p.buf.data, _make.Cast(dtype, ret[tuple(ivs)]), index)
  elif isinstance(ret, (_expr.Expr, numbers.Number)):
    indices = lambda_args
    index, _, _ = util.get_index(shape, indices, 0)
    body = _make.Store(p.buf.data, _make.Cast(dtype, ret), index)
  else:
    raise ValueError("Unrecognized return value in hcl.compute")
  body = _make.Block(CodeBuilder.get(), body)
  body = util.make_for(indices, body, 0)

  builders = CodeBuilder.current
  if len(builders) != 0:
    builder = builders[-1]
    builder.emit(lambda x: _make.Allocate(p.buf.data, dtype, shape, util.true(), x))
    builder.emit(body)
    CodeBuilder.var_dict[-1][name] = p
    CodeBuilder.axis_list[-1] += (indices + axis)

  op = tensor.Operation(inputs, p, body, indices + axis)
  tensor.Operation.op_list.append(op)

  return p

def update(_tensor, inputs, fcompute, name = None):
  code = fcompute.__code__
  args = code.co_varnames
  nargs = code.co_argcount
  shape = _tensor.shape
  dtype = _tensor.dtype

  name = "update" + str(util.UID) if name is None else name
  util.UID += 1

  lambda_args = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]
  var_list = [i.var for i in lambda_args]
  body = None
  p = tensor.Tensor((1,), "int32", name)

  with CodeBuilder():
    ret = fcompute(*var_list)
  p.var_dict = CodeBuilder.var_dict[-1]
  axis = CodeBuilder.axis_list[-1]

  if isinstance(ret, tensor.TensorSlice):
    ret = ret.asnode()
    indices = lambda_args
    index, _, _ = util.get_index(shape, indices, 0)
    body = _make.Store(_tensor.buf.data, _make.Cast(dtype, ret), index)
  elif isinstance(ret, tensor.Tensor):
    ivs = [_IterVar((0, ret.shape[i]), "red_var" + str(i), 0) for i in range(0, len(ret.shape))]
    indices = []
    rid = 0
    for iv in lambda_args:
      if iv.var.name == "_":
        indices.append(ivs[rid])
        rid += 1
      else:
        indices.append(iv)
    index, _, _ = util.get_index(shape, indices, 0)
    inner, _, _ = util.get_index(ret.shape, ivs, 0)
    body = _make.Store(_tensor.buf.data, _make.Cast(dtype, ret[tuple(ivs)]), index)
  elif isinstance(ret, (_expr.Expr, numbers.Number)):
    indices = lambda_args
    index, _, _ = util.get_index(shape, indices, 0)
    body = _make.Store(_tensor.buf.data, _make.Cast(dtype, ret), index)
  else:
    raise ValueError("Unrecognized return value in hcl.compute")
  body = _make.Block(CodeBuilder.get(), body)
  body = util.make_for(indices, body, 0)

  builders = CodeBuilder.current
  if len(builders) != 0:
    builder = builders[-1]
    builder.emit(body)
    CodeBuilder.var_dict[-1][name] = p
    CodeBuilder.axis_list[-1] += (indices + axis)

  op = tensor.Operation(inputs, p, body, indices + axis)
  tensor.Operation.op_list.append(op)

  return p

def copy(_tensor, name = None):
  name = "compute" + str(util.CID) if name is None else name
  util.CID += 1

  indices = [_IterVar((0, _tensor.shape[n]), "copy_i" + str(n), 0) for n in range(0, len(_tensor.shape))]
  p = tensor.Tensor(_tensor.shape, _tensor.dtype, name)

  index, _, _ = util.get_index(_tensor.shape, indices, 0)
  body = _make.Store(p.buf.data, _make.Cast(_tensor.dtype, _tensor[tuple(indices)]), index)
  body = util.make_for(indices, body, 0)

  builders = CodeBuilder.current
  if len(builders) != 0:
    builder = builders[-1]
    builder.emit(lambda x: _make.Allocate(p.buf.data, _tensor.dtype, _tensor.shape, util.true(), x))
    builder.emit(body)
    CodeBuilder.var_dict[-1][name] = p
    CodeBuilder.axis_list[-1] += indices

  op = tensor.Operation([_tensor], p, body, indices)
  tensor.Operation.op_list.append(op)

  return p

def copy_inplace(_tensor, _from, name = None):
  name = "update" + str(util.UID) if name is None else name
  util.UID += 1

  indices = [_IterVar((0, _tensor.shape[n]), "copyip_i" + str(n), 0) for n in range(0, len(_tensor.shape))]
  p = tensor.Tensor((1,), "int32", name)

  index, _, _ = util.get_index(_tensor.shape, indices, 0)
  body = _make.Store(_tensor.buf.data, _make.Cast(_tensor.dtype, _from[tuple(indices)]), index)
  body = util.make_for(indices, body, 0)

  builders = CodeBuilder.current
  if len(builders) != 0:
    builder = builders[-1]
    builder.emit(body)
    CodeBuilder.var_dict[-1][name] = p
    CodeBuilder.axis_list[-1] += indices

  op = tensor.Operation([_tensor, _from], p, body, indices)
  tensor.Operation.op_list.append(op)

  return p

def block(inputs, fblock, name = None):
  name = "block" + str(util.BID) if name is None else name
  util.BID += 1

  p = tensor.Tensor((1,), "int32", name)

  with CodeBuilder():
    fblock()
  p.var_dict = CodeBuilder.var_dict[-1]
  axis = CodeBuilder.axis_list[-1]
  body = CodeBuilder.get()

  op = tensor.Operation(inputs, p, body, axis)
  tensor.Operation.op_list.append(op)

  return p

def mut_compute(shape, inputs, fcompute, name = None):
  code = fcompute.__code__
  args = code.co_varnames
  nargs = code.co_argcount

  name = "mut_compute" + str(util.MID) if name is None else name
  util.MID += 1

  p = tensor.Tensor((1,), "int32", name)

  assert (len(shape) == nargs), "fcompute does not match output dimension"

  indices = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]
  var_list = [i.var for i in indices]

  with CodeBuilder():
    fcompute(*var_list)
  p.var_dict = CodeBuilder.var_dict[-1]
  axis = CodeBuilder.axis_list[-1]
  ret = CodeBuilder.get()
  body = util.make_for(indices, ret, 0)

  builders = CodeBuilder.current
  if len(builders) != 0:
    builder = builders[-1]
    builder.emit(body)
    CodeBuilder.var_dict[-1][name] = p
    CodeBuilder.axis_list[-1] += (indices + axis)

  op = tensor.Operation(inputs, p, body, indices + axis)
  tensor.Operation.op_list.append(op)

  return p

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

  with CodeBuilder():
    ret_val = fkernel(*inputs)
  ret_val = 0 if ret_val is None else ret_val
  ret_dtype = config.init_dtype if ret_dtype is None else ret_dtype
  ret_dtype = util.convert_dtype(ret_dtype)

  _ret_void = _make.UIntImm("uint1", 1) if ret_void else _make.UIntImm("uint1", 0)

  var_dict = CodeBuilder.var_dict[-1]
  axis = CodeBuilder.axis_list[-1]
  body = _make.KernelDef(args, CodeBuilder.get(), _ret_void, ret_val, name)
  p = _kernel.KernelTensor(arg_type, name, ret_void, ret_dtype, body)
  p.var_dict = var_dict

  op = tensor.Operation(None, p, body, axis)
  tensor.Operation.op_list.append(op)

  return p

def resize(inputs, dtype):
  from_vars = []
  to_vars = []
  assert isinstance(dtype, (str, types.Type)), "Wrong input to resize data type"
  dtype = util.convert_dtype(dtype)
  if not isinstance(inputs, (list, tuple)):
    inputs = [inputs]
  for i in inputs:
    if isinstance(i, tensor.Var):
      from_vars.append(i.var)
      new_var = _var(i.name, dtype)
      i.var = new_var
      to_vars.append(new_var)
    else:
      from_vars.append(i.buf.data)
      new_buf = decl_buffer(i.shape, dtype, i.name)
      i.buf = new_buf
      i.dtype = dtype
      to_vars.append(new_buf.data)
  op_list = tensor.Operation.op_list
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
  for op in tensor.Operation.op_list:
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

  tensor.Operation.op_list = []

  if not isinstance(t, list):
    t = [t]
  ops = [t_.op for t_ in t]
  return Schedule(_schedule.create_schedule(ops))

def lower(schedule, inputs):
  new_inputs = []
  for i in inputs:
    if isinstance(i, tensor.Tensor):
      new_inputs.append(i.tensor)
    else:
      new_inputs.append(i.var)
  return _lower(schedule.sch, new_inputs, simple_mode = True)

def build(schedule, inputs, target=None):
  new_inputs = []
  for i in inputs:
    if isinstance(i, tensor.Tensor):
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
    if isinstance(init, (_expr.Expr, numbers.Number)):
      out = local(init, "reducer", dtype)
      def reduce_body():
        with if_(where):
          out[0] = freduce(expr, out[0])
        return out[0]
      with CodeBuilder():
        ret = reduce_body()
    else: # a list or tensor
      out = copy(init, "reducer")
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
    cb.emit(body)
    return ret

  return make_reduce

def asarray(arr, dtype = "int32", ctx = cpu(0)):
  dtype = util.convert_dtype(dtype)
  return array(arr, dtype, ctx)

sum = reducer(0, lambda x, y: x + y)
