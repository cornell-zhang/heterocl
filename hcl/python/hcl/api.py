from . import tensor
from . import visitor
from tvm.api import _IterVar, decl_buffer
from tvm import var as _var
from tvm import placeholder as _placeholder
from tvm import _api_internal as _tvm_api
import inspect
import ast

import tvm

def var(name = "var", dtype = "int32"):
  return _var(name = name, dtype = dtype)

def placeholder(shape, name = "placeholder", dtype = "float32"):
  p = _placeholder(shape, name = name, dtype = "float32")
  #return tensor.Tensor(p, dtype = dtype)
  return p

# DOES NOT SUPPORT A[x, y] WRITE A[x][y] INSTEAD
def compute(shape, inputs, fcompute, name = "compute", dtype = "float32", inline = True, extern_funcs = []):
  """
  The vector code version for a computation. Currently we only support one output.

  Inputs
  ------
  shape: tuple of integers, the shape of the output tensor
  inputs: list of input Tensors
  fcompute: lambda function that performs the computation
  dtpye: string, the output data type
  inline: boolean, default is True
  extern_funcs: list of external functions that are used inside the lambda function

  Output
  ------
  Tensor
  """
  code = fcompute.__code__
  args = code.co_varnames
  nargs = code.co_argcount
  indices = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]
  op = None

  if inline:
    body = fcompute(*indices)
    op = _tvm_api._ComputeOp(name, "", indices, [body])
    op = op.output(0)
  else:
    # collect input placeholders
    input_placeholders = [decl_buffer(i.shape, i.dtype, i.op.name) for i in inputs]
    # infer output dtype
    output_placeholders = [decl_buffer(shape, dtype, name)]
    # collect body
    if len(args) == 0: #TODO debug message
      print "WRONG NUMBER OF ARGS!!"
    lambda_root = visitor.LambdaVisitor().enter(inspect.getsource(code)) # extract the lambda function AST
    body = visitor.HalideIRVisitor().compile(lambda_root, inputs, input_placeholders, output_placeholders[0], extern_funcs) # compile Python AST to Halide IR
    op = _tvm_api._ExternOp(name, "", inputs, input_placeholders, output_placeholders, body)
    op = op.output(0)

  return op

def update(tensor, inputs, fcompute, name = "update", extern_funcs = []):
  code = fcompute.__code__
  args = code.co_varnames

  # collect input placeholders
  assert (tensor not in inputs)
  input_placeholders = [decl_buffer(i.shape, i.dtype, i.op.name) for i in inputs]
  update_placeholder = decl_buffer(tensor.shape, tensor.dtype, tensor.op.name)
  inputs.append(tensor)
  input_placeholders.append(update_placeholder)
  # infer output dtype
  output_placeholders = [decl_buffer((1,), "int32", name)]
  # collect body
  if len(args) == 0: #TODO debug message
    print "WRONG NUMBER OF ARGS!!"
  lambda_root = visitor.LambdaVisitor().enter(inspect.getsource(code)) # extract the lambda function AST
  body = visitor.HalideIRVisitor().compile(lambda_root, inputs, input_placeholders, update_placeholder, extern_funcs) # compile Python AST to Halide IR
  op = _tvm_api._ExternOp(name, "", inputs, input_placeholders, output_placeholders, body)
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
