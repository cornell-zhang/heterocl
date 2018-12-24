from .util import get_index, make_for, get_dtype, CastRemover
from .tensor import Tensor, TensorSlice
from .schedule import Stage
from .debug import APIError, HCLError
from .mutator import Mutator
from tvm import expr as _expr, stmt as _stmt, make as _make
from tvm import _api_internal
from tvm.api import _IterVar
from numbers import Number
import inspect

class ReplaceReturn(Mutator):

    def __init__(self, buffer_var, dtype, index):
        self.buffer_var = buffer_var
        self.dtype = dtype
        self.index = index

    def mutate_KerenlDef(self, node):
        return node

    def mutate_Return(self, node):
        value = self.mutate(node.value)
        return _make.Store(self.buffer_var, _make.Cast(self.dtype, value), self.index)

def compute_body(name, lambda_ivs, fcompute, shape=(), dtype=None, tensor=None):
    var_list = [i.var for i in lambda_ivs]
    return_tensor = True if tensor is None else False

    with Stage(name, dtype, shape) as stage:
        dtype = stage._dtype
        if not return_tensor:
            stage.input_stages.add(tensor.last_update)
        else:
            tensor = Tensor(shape, dtype, name, stage._buf)
        buffer_var = tensor._buf.data
        dtype = tensor.dtype
        shape = tensor.shape

        stage.stmt_stack.append([])
        ret = fcompute(*var_list)

        stage.lhs_tensors.add(tensor)
        for t in stage.lhs_tensors:
            t.last_update = stage

        if ret is None:
            # replace all hcl.return_ with Store stmt
            indices = lambda_ivs
            index, _, _ = get_index(shape, indices, 0)
            stmt = stage.pop_stmt()
            stmt = ReplaceReturn(buffer_var, dtype, index).mutate(stmt)
            stage.emit(make_for(indices, stmt, 0))
        elif isinstance(ret, (TensorSlice, _expr.Expr, Number)):
            indices = lambda_ivs
            index, _, _ = get_index(shape, indices, 0)
            stage.emit(_make.Store(buffer_var, _make.Cast(dtype, ret), index))
            stmt = stage.pop_stmt()
            stage.emit(make_for(indices, stmt, 0))
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
            stage.emit(_make.Store(buffer_var, _make.Cast(dtype, ret[tuple(ret_ivs)]), index))
            stmt = stage.pop_stmt()
            stage.emit(make_for(indices, stmt, 0))
        else:
            print ret
            #raise ValueError("Unrecognized return type")

        stage.axis_list = indices + stage.axis_list

    if return_tensor:
        tensor._tensor = stage._op
        return tensor
