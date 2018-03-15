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

def placeholder(shape, name = "placeholder", dtype = "int32"):
  p = _placeholder(shape, name = name, dtype = "int32")
  return tensor.Tensor(p, dtype = dtype)

# DOES NOT SUPPORT A[x, y] WRITE A[x][y] INSTEAD
def compute(shape, inputs, fcompute, name = "compute", dtype = "int32", inline = True, extern_funcs = []):
  code = fcompute.__code__
  args = code.co_varnames
  nargs = code.co_argcount
  indices = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]
  op = None

  if inline:
    body = fcompute(*indices)
    op = _tvm_api._ComputeOp(name, "", indices, [body])
  else:
    # collect input placeholders
    for i in inputs:
      assert isinstance(i.tensor, tvm.tensor.Tensor)
    inputs = [i.tensor for i in inputs]
    input_placeholders = [decl_buffer(i.shape, i.dtype, i.op.name) for i in inputs]
    # infer output dtype
    output_placeholders = [decl_buffer(shape, dtype, name)]
    # collect body
    if len(args) == 0: #TODO debug message
      print "WRONG NUMBER OF ARGS!!"
    lambda_root = visitor.LambdaVisitor().enter(inspect.getsource(code)) # extract the lambda function AST
    body = visitor.HalideIRVisitor().compile(lambda_root, inputs, input_placeholders, output_placeholders[0], None) # compile Python AST to Halide IR
    print body
    op = _tvm_api._ExternOp(name, "", inputs, input_placeholders, output_placeholders, tvm.make.Evaluate(1))

  return op
