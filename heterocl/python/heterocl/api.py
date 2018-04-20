from . import tensor
from . import util
from .code_builder import CodeBuilder
from tvm.api import _IterVar, decl_buffer, convert
from tvm.build_module import build as _build
from tvm import var as _var
from tvm import placeholder as _placeholder
from tvm import _api_internal as _tvm_api
from tvm import make as _make
from tvm import expr as _expr
from tvm import stmt as _stmt
import inspect
import ast

def var(name = "var", dtype = "int32"):
  return _var(name = name, dtype = dtype)

def placeholder(shape, name = "placeholder", dtype = "int32"):
  builder = CodeBuilder.current
  p = _placeholder(shape, name = name, dtype = dtype)
  p = tensor.Tensor(p)
  if builder is None:
    return p
  else:
    builder.emit(lambda x: _make.Allocate(p.buf.data, dtype, shape, util.true, x))
    return p

def local(init = 0, name = "local", dtype = "int32"):
  builder = CodeBuilder.current
  assert builder is not None, "hcl.local must be used inside a code builder"
  p = _placeholder((1,), name = name, dtype = dtype)
  p = tensor.Tensor(p)
  builder.emit(lambda x: _make.Allocate(p.buf.data, dtype, (1,), util.true, x))
  p[0] = init
  return p

# TODO: record the index of all calls and loops
def compute(shape, inputs, fcompute, name = "compute", dtype = "int32", inline = False):
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
  op = None
  body = None

  if inline:
    body = fcompute(*indices)
    body = convert([body])
    op = _tvm_api._ComputeOp(name, "", indices, body)
    op = op.output(0)
  else:
    input_tensors = []
    input_placeholders = []
    for i in inputs:
      input_tensors.append(i.tensor)
      input_placeholders.append(i.buf)
    # infer output dtype
    output_placeholder = decl_buffer(shape, dtype, name)
    # collect body
    ret = fcompute(*indices)
    index, _, _ = util.get_index(shape, indices, 0)
    if isinstance(ret, tuple):
      ret = list(ret)
      if isinstance(ret[0], tensor.TensorSlice):
        ret[0] = ret[0].asnode()
      assert isinstance(ret[0], _expr.Expr), "The returned value must be an expression"
      assert isinstance(ret[1], _stmt.Stmt), "The returned body must be a statement"
      store = _make.Store(output_placeholder.data, _make.Cast(dtype, ret[0]), index, util.true)
      body = _make.Block(ret[1], store)
      body = util.make_for(indices, body, 0)
    else:
      store = _make.Store(output_placeholder.data, _make.Cast(dtype, ret), index, util.true)
      body = util.make_for(indices, store, 0)
    op = _tvm_api._ExternOp(name, "", input_tensors, input_placeholders, [output_placeholder], body)
    op = op.output(0)

  p = tensor.Tensor(op, output_placeholder)

  builder = CodeBuilder.current
  if builder is not None:
    builder.emit(lambda x: _make.Allocate(p.buf.data, dtype, shape, util.true, x))
    builder.emit(op.op.body)

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

  assert (len(shape) == nargs), "fcompute does not match output dimension"

  indices = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]

  input_tensors = []
  input_placeholders = []
  for i in inputs:
    input_tensors.append(i.tensor)
    input_placeholders.append(i.buf)
  # infer output dtype
  output_placeholder = decl_buffer((1,), dtype, name)
  # collect body
  ret = fcompute(*indices)
  body = util.make_for(indices, ret, 0)
  op = _tvm_api._ExternOp(name, "", input_tensors, input_placeholders, [output_placeholder], body)
  op = op.output(0)

  return op

def build(schedule, inputs):

  new_inputs = []
  for i in inputs:
    if isinstance(i, tensor.Tensor):
      new_inputs.append(i.tensor)
    else:
      new_inputs.append(i)

  return _build(schedule, new_inputs)

