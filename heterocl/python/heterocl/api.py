from . import tensor
from . import util
from .code_builder import CodeBuilder
from .resizer import Resizer
from tvm.api import _IterVar, decl_buffer, convert
from tvm.build_module import build as _build
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

def var(name = "var", dtype = "int32"):
  return tensor.Var(_var(name = name, dtype = dtype))

def placeholder(shape, name = "placeholder", dtype = "int32"):
  builder = CodeBuilder.current
  p = tensor.Tensor(shape, dtype, name)
  op = tensor.Operation(None, p, None)
  tensor.Operation.op_list.append(op)
  if builder is None:
    return p
  else:
    builder.emit(lambda x: _make.Allocate(p.buf.data, dtype, shape, util.true, x))
    return p

def local(init = 0, name = "local", dtype = "int32"):
  builder = CodeBuilder.current
  assert builder is not None, "hcl.local must be used inside a code builder"
  p = tensor.Tensor((1,), dtype, name)
  builder.emit(lambda x: _make.Allocate(p.buf.data, dtype, (1,), util.true, x))
  p[0] = init
  op = tensor.Operation(None, p, None)
  tensor.Operation.op_list.append(op)
  return p

# TODO: record the index of all calls and loops
def compute(shape, inputs, fcompute, name = "compute", dtype = "int32"):
  """
  A function that performs tensor computation 'fcompute' and returns a new tensor.

  Parameters
  ----------
  shape: tuple of integers
    the shape of the output tensor

  inputs: list of Tensor and/or Var
    the tensors or variables used inside fcompute

  fcompute: lambda function
    the function that performs the computation, the number of indices must match output dimension

  dtpye: string
    the output data type

  inline: boolean
    whether to inline fcompute or not, default is True

  Output
  ------
  Tensor
  """
  code = fcompute.__code__
  args = code.co_varnames
  nargs = code.co_argcount

  assert (len(shape) == nargs), "fcompute does not match output dimension"

  indices = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]
  body = None
  p = tensor.Tensor(shape, dtype, name)

  ret = fcompute(*indices)
  index, _, _ = util.get_index(shape, indices, 0)
  if isinstance(ret, tensor.TensorSlice):
    ret = ret.asnode()
  assert isinstance(ret, (_expr.Expr, numbers.Number)), "The returned value must be an expression"
  store = _make.Store(p.buf.data, _make.Cast(dtype, ret), index, util.true)
  if CodeBuilder.stmt_stack is None:
    body = util.make_for(indices, store, 0)
  else:
    body =  _make.Block(CodeBuilder.get(), store)
    body = util.make_for(indices, body, 0)


  builder = CodeBuilder.current
  if builder is not None:
    builder.emit(lambda x: _make.Allocate(p.buf.data, dtype, shape, util.true, x))
    builder.emit(body)

  op = tensor.Operation(inputs, p, body)
  tensor.Operation.op_list.append(op)

  return p

#TODO: incorrect
def update(tensor, inputs, fcompute, name = "update", extern = []):
  code = fcompute.__code__
  args = code.co_varnames

  # collect input placeholders
  input_tensors = []
  input_vars = []
  for i in inputs:
    input_tensors.append(i) if isinstance(i, tvm.tensor.Tensor) else input_vars.append(i)
  input_placeholders = [decl_buffer(i.shape, i.dtype, i.op.name) for i in input_tensors]
  update_placeholder = decl_buffer(tensor.shape, tensor.dtype, tensor.op.name)
  if tensor not in inputs:
    input_tensors.append(tensor)
    input_placeholders.append(update_placeholder)
  # infer output dtype
  output_placeholders = [decl_buffer((1,), "int32", name)]
  # collect body
  if len(args) == 0: #TODO debug message
    print "WRONG NUMBER OF ARGS!!"
  lambda_root = visitor.LambdaVisitor().enter(inspect.getsource(code)) # extract the lambda function AST
  body = visitor.HalideIRVisitor().compile_lambda(lambda_root, input_tensors, input_placeholders, input_vars, update_placeholder, extern) # compile Python AST to Halide IR
  op = _tvm_api._ExternOp(name, "", input_tensors, input_placeholders, output_placeholders, body)
  op = op.output(0)

  return op

#TODO: incorrect
def block(fblock, inputs, args = [], name = "block", extern = []):
  input_placeholders = [decl_buffer(i.shape, i.dtype, i.op.name) for i in inputs]
  output_placeholders = [decl_buffer((1,), "int32", name)]
  # compile fblock to Halide IR
  if len(args) == 0:
    args = inputs
  body = visitor.HalideIRVisitor().compile_block(fblock, inputs, input_placeholders, args, extern)
  op = _tvm_api._ExternOp(name, "", inputs, input_placeholders, output_placeholders, body)
  op = op.output(0)

  return op

def mut_compute(shape, inputs, fcompute, name = "mut_compute", dtype = "int32"):
  code = fcompute.__code__
  args = code.co_varnames
  nargs = code.co_argcount
  p = tensor.Tensor((1,), "int32", name)

  assert (len(shape) == nargs), "fcompute does not match output dimension"

  indices = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]

  fcompute(*indices)
  assert not CodeBuilder.stmt_stack is None
  ret = CodeBuilder.get()
  body = util.make_for(indices, ret, 0)

  op = tensor.Operation(inputs, p, body)
  tensor.Operation.op_list.append(op)

  return p

def resize(inputs, dtype):
  from_vars = []
  to_vars = []
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

def create_schedule(t):
  for op in tensor.Operation.op_list:
    if op.inputs is None: #placeholder
      p = op.output
      p.tensor = _api_internal._Placeholder(p.buf.shape, p.dtype, p.name)
    else:
      i = op.inputs
      p = op.output
      for _i in i:
        assert not _i.tensor is None
      i_tensor = [_i.tensor for _i in i]
      i_buf = [_i.buf for _i in i]
      o_buf = [p.buf]
      p.tensor = _api_internal._ExternOp(p.name, "", i_tensor, i_buf, o_buf, op.body).output(0)

  tensor.Operation.op_list = []
  return _schedule.create_schedule(t.op)

def build(schedule, inputs):
  new_inputs = []
  for i in inputs:
    if isinstance(i, tensor.Tensor):
      new_inputs.append(i.tensor)
    else:
      new_inputs.append(i.var)

  return _build(schedule, new_inputs)

def comm_reducer(init, freduce):

  def make_reduce(expr, axis, where = True):
    with CodeBuilder() as cb:
      out = local(init, expr.dtype) # TODO: expr may not have dtype??
      cb.emit(_make.For(axis.var, axis.dom.min, axis.dom.extent, 0, 0,
        _make.IfThenElse(where,
          _make.Store(out.buf.data, freduce(expr, out[0]), 0, util.true), None)))
      return out[0]

  return make_reduce

sum = comm_reducer(0, lambda x, y: x + y)
