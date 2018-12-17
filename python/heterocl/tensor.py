from . import util
from . import debug
from .resizer import CastRemover
from .schedule import Stage
from tvm import make as _make
from tvm import expr as _expr
from tvm.api import decl_buffer
from tvm._ffi.node import NodeGeneric

class Scalar(NodeGeneric, _expr.ExprOp):
    def __init__(self, var):
        self.var = var

    def __getitem__(self, indices):
        if type(indices) == slice:
            return _make.GetSlice(self.var, indices.start, indices.stop)
        elif isinstance(indices, (int, _expr.Expr)):
            return _make.GetBit(self.var, indices)
        else:
            raise ValueError("Invalid indices to Var")

    def same_as(self, var):
        if isinstance(var, Scalar):
            return self.var.same_as(var.var)
        elif isinstance(var, _expr.Expr):
            return self.var.same_as(var)
        else:
            return False

    @property
    def name(self):
        return self.var.name

    @property
    def dtype(self):
        return self.var.dtype

    def asnode(self):
        return self.var

class TensorSlice(NodeGeneric, _expr.ExprOp):
    def __init__(self, tensor, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self.tensor = tensor
        self.indices = indices

    def __getitem__(self, indices):
        indices = CastRemover().mutate(indices)
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorSlice(self.tensor, self.indices + indices)

    def __setitem__(self, indices, expr):
        if not isinstance(indices, tuple):
            indices = (indices,)
        indices = self.indices + indices
        indices = CastRemover().mutate(indices)
        index, bit, _ = util.get_index(self.tensor.shape, indices, 0)
        assert Stage.get_len() != 0
        builder = Stage.get_current()
        if bit is None:
            builder.emit(_make.Store(self.tensor.buf.data, _make.Cast(self.tensor.dtype, expr), index))
        elif type(bit) == slice:
            load = _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
            expr = _make.SetSlice(load, expr, bit.start, bit.stop)
            builder.emit(_make.Store(self.tensor.buf.data, _make.Cast(self.tensor.dtype, expr), index))
        else:
            load = _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
            expr = _make.SetBit(load, expr, bit)
            builder.emit(_make.Store(self.tensor.buf.data, _make.Cast(self.tensor.dtype, expr), index))

    def asnode(self):
        if len(self.indices) < len(self.tensor.shape):
            raise ValueError("Inconsistant tensor slice dimension with tensor dimension")
        index, bit, _ = util.get_index(self.tensor.shape, self.indices, 0)
        if bit is None:
            return _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
        elif type(bit) == slice:
            return _make.GetSlice(_make.Load(self.tensor.dtype, self.tensor.buf.data, index), bit.start, bit.stop)
        else:
            return _make.GetBit(_make.Load(self.tensor.dtype, self.tensor.buf.data, index), bit)

    @property
    def dtype(self):
        return self.tensor.dtype


class Tensor(NodeGeneric, _expr.ExprOp):
    """A wrapper class for TVM intrinsic Tensor class"""

    def __init__(self, shape, dtype = "int32", name = "hcl.tensor", buf = None):
        self._tensor = None
        self._buf = buf
        self.dtype = dtype
        self.shape = shape
        self.name = name
        self.var_dict = {}
        self.first_update = None
        self.last_update = None
        if buf is None:
            self._buf = decl_buffer(shape, dtype, name)

    def __repr__(self):
        return "Tensor('" + self.name + "', " + str(self.shape) + ", " + str(self.dtype) + ")"

    # A[x, y] RHS
    def __getitem__(self, indices):
        if Stage.get_len():
            Stage.get_current().input_stages.add(self.last_update)
        #indices = CastRemover().mutate(indices)
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorSlice(self, indices)

    # A[x, y] LHS
    def __setitem__(self, indices, expr):
        Stage.get_current().input_stages.add(self.last_update)
        Stage.get_current().lhs_tensors.add(self)
        if not isinstance(indices, tuple):
            indices = (indices,)
        indices = CastRemover().mutate(indices)
        if len(indices) < len(self.shape):
            raise NotImplementedError()
        else:
            index, bit, _ = util.get_index(self.shape, indices, 0)
            assert Stage.get_len() != 0
            if bit is None:
                Stage.get_current().emit(_make.Store(self.buf.data, _make.Cast(self.dtype, expr), index))
            else:
                raise NotImplementedError()

    def asnode(self):
        if len(self.shape) == 1 and self.shape[0] == 1:
            return TensorSlice(self, 0).asnode()
        else:
            raise ValueError("Cannot perform expression on Tensor")

    def same_as(self, tensor):
        if isinstance(tensor, Tensor):
            return self._tensor.same_as(tensor.tensor)
        else:
            return False

    @property
    def tensor(self):
        return self._tensor

    @property
    def buf(self):
        return self._buf

    @property
    def type(self):
        return util.convert2hcl_dtype(self.dtype)

    @property
    def op(self):
        return self.tensor.op

    @property
    def axis(self):
        return self.tensor.op.axis

    @property
    def v(self):
        if len(self.shape) == 1 and self.shape[0] == 1:
            return self.__getitem__(0)
        else:
            raise debug.APIError(".v can only be used on mutable scalars")

    @buf.setter
    def buf(self, buf):
        self._buf = buf
        Tensor.tensor_map[self._tensor] = buf

    @tensor.setter
    def tensor(self, tensor):
        self._tensor = tensor

    @v.setter
    def v(self, value):
        self.__setitem__(0, value)
