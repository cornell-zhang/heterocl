from . import tensor
from . import visitor
from tvm.api import _IterVar, decl_buffer, convert
from tvm import var as _var
from tvm import placeholder as _placeholder
from tvm import _api_internal as _tvm_api
import inspect
import ast

import tvm

def var(name = "var", dtype = "int32"):
  return _var(name = name, dtype = dtype)

def placeholder(shape, name = "placeholder", dtype = "float32"):
  p = _placeholder(shape, name = name, dtype = dtype)
  #return tensor.Tensor(p, dtype = dtype)
  return p

# DOES NOT SUPPORT A[x, y] WRITE A[x][y] INSTEAD
# TODO: replace tvm.tensor.Tensor with hcl.tensor.Tensor
# TODO: record the index of all calls and loops
def compute(shape, inputs, fcompute, name = "compute", dtype = "float32", inline = True, extern_funcs = []):
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

  extern_funcs: list of functions
    functions that are used inside fcompute

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

  if inline:
    body = fcompute(*indices)
    body = convert([body])
    op = _tvm_api._ComputeOp(name, "", indices, body)
    op = op.output(0)
    print op.dtype
  else:
    input_tensors = []
    input_vars = []
    for i in inputs:
      input_tensors.append(i) if isinstance(i, tvm.tensor.Tensor) else input_vars.append(i)
    # collect input placeholders
    input_placeholders = [decl_buffer(i.shape, i.dtype, i.op.name) for i in input_tensors]
    # infer output dtype
    output_placeholder = decl_buffer(shape, dtype, name)
    # collect body
    if len(args) == 0: #TODO debug message
      print "WRONG NUMBER OF ARGS!!"
    lambda_root = visitor.LambdaVisitor().enter(inspect.getsource(code)) # extract the lambda function AST
    body = visitor.HalideIRVisitor().compile_lambda(lambda_root, input_tensors, input_placeholders, input_vars, output_placeholder, extern_funcs) # compile Python AST to Halide IR
    op = _tvm_api._ExternOp(name, "", input_tensors, input_placeholders, [output_placeholder], body)
    op = op.output(0)

  return op

def update(tensor, inputs, fcompute, name = "update", extern_funcs = []):
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
  body = visitor.HalideIRVisitor().compile_lambda(lambda_root, input_tensors, input_placeholders, input_vars, update_placeholder, extern_funcs) # compile Python AST to Halide IR
  op = _tvm_api._ExternOp(name, "", input_tensors, input_placeholders, output_placeholders, body)
  op = op.output(0)

  return op

def block(fblock, inputs, args = [], name = "block", extern_funcs = []):
  input_placeholders = [decl_buffer(i.shape, i.dtype, i.op.name) for i in inputs]
  output_placeholders = [decl_buffer((1,), "int32", name)]
  # compile fblock to Halide IR
  if len(args) == 0:
    args = inputs
  body = visitor.HalideIRVisitor().compile_block(fblock, inputs, input_placeholders, args, extern_funcs)
  op = _tvm_api._ExternOp(name, "", inputs, input_placeholders, output_placeholders, body)
  op = op.output(0)

  return op
