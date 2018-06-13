import .util
from .tensor import Tensor, TensorSlice
from tvm import expr as _expr, stmt as _stmt, make as _make
from numbers import Number

"""
Auxiliary function for API with return values
"""
def compute_body(ret_tensor, lambda_ivs, fcompute):
  indices = None
  var_list = [i.var for i in lambda_ivs]

  with CodeBuilder() as cb:
    ret = fcompute(*var_list)

    inputs = list(cb.tensors)
    if isinstance(ret, (TensorSlice, _expr.Expr, Number)):
      indices = lambda_ivs
      index, _, _ = util.get_index(shape, indices, 0)
      cb.emit(_make.Store(p.buf.data, _make.Cast(dtype, ret), index))
    elif isinstance(ret, Tensor):
      ivs = [_IterVar((0, ret.shape[i]), ret.name + "_i" + str(i), 0) for i in range(0, len(ret.shape))]
      indices = []
      rid = 0
      for iv in lambda_ivs:
        if iv.var.name[0] == "_":
          indices.append(ivs[rid])
          rid += 1
        else:
          indices.append(iv)
      if len(indices) != len(shape):
        raise HCLError("Incorrect number of lambda arguments", inspect.stack()[1])
      index, _, _ = util.get_index(shape, indices, 0)
      inner, _, _ = util.get_index(ret.shape, ivs, 0)
      cb.emit(_make.Store(p.buf.data, _make.Cast(dtype, ret[tuple(ivs)]), index))
    else:
      raise ValueError("Unrecognized return value in hcl.compute")

  return inputs, indices
