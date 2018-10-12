from .util import get_index
from .tensor import Tensor, TensorSlice
from .code_builder import CodeBuilder
from .resizer import CastRemover
from .debug import APIError, HCLError
from tvm import expr as _expr, stmt as _stmt, make as _make
from tvm import _api_internal
from tvm.api import _IterVar
from numbers import Number
import inspect

"""
Auxiliary function for API with return values
"""
def compute_body(tensor, stage, lambda_ivs, fcompute):
  shape = tensor.shape
  dtype = tensor.dtype
  buffer_var = tensor.buf.data
  var_list = [i.var for i in lambda_ivs]

  with CodeBuilder(stage.name) as cb:
    ret = fcompute(*var_list)

    cb.lhs.add(tensor)
    for t in cb.lhs:
      t.last_update = stage

    inputs = list(cb.last_stages.union(cb.tensors))
    if isinstance(ret, (TensorSlice, _expr.Expr, Number)):
      indices = lambda_ivs
      index, _, _ = get_index(shape, indices, 0)
      cb.emit(_make.Store(buffer_var, _make.Cast(dtype, ret), index))
    elif isinstance(ret, Tensor): # reduction
      ret_ivs = [_IterVar((0, ret.shape[i]), ret.name + "_i" + str(i), 0) for i in range(0, len(ret.shape))]
      indices = []
      rid = 0
      for iv in lambda_ivs:
        if iv.var.name[0] == "_":
          indices.append(ret_ivs[rid])
          rid += 1
        else:
          indices.append(iv)
      if len(indices) != len(shape):
        raise HCLError("Incorrect number of lambda arguments", inspect.stack()[2])
      index, _, _ = get_index(shape, indices, 0)
      cb.emit(_make.Store(buffer_var, _make.Cast(dtype, ret[tuple(ret_ivs)]), index))
    else:
      raise HCLError("Unrecognized return type", inspect.stack()[2])

  tensor.var_dict = CodeBuilder.get_var_dict()
  axis = CodeBuilder.get_axis()
  cb.lhs.add(tensor)

  return inputs, indices, cb.lhs, axis

#def stage_body(tensor)

def in_builder_process(tensor, inputs, lhs):
  builder = CodeBuilder.get_cb()
  builder.emit(lambda x: _make.AttrStmt(tensor.buf, "attach_scope", _make.StringImm(builder.name), x))
  builder.last_stages.add(tensor)
  builder.last_stages.difference_update(set(inputs))
  builder.lhs.update(lhs)
  CodeBuilder.get_var_dict()[tensor.name] = tensor

def make_extern_op(inputs, output, axis, body):
  i_tensor = [_i.tensor for _i in inputs]
  i_buf = [_i.buf for _i in inputs]
  o_buf = [output.buf]
  output.tensor = _api_internal._ExternOp(output.name, "", axis, i_tensor, i_buf, o_buf, body).output(0)

