"""HeteroCL tensors and scalars."""
#pylint: disable=missing-docstring, too-many-instance-attributes
from .tvm import make as _make
from .tvm import expr as _expr
from .tvm.api import decl_buffer
from .tvm._ffi.node import NodeGeneric
from .debug import TensorError
from .schedule import Stage
from . import util
from . import debug
from . import types

class Scalar(NodeGeneric, _expr.ExprOp):
    """A non-mutable scalar.

    This should be used by `heterocl.placeholder` only. Valid usages of
    accessing a scalar include direct access and bit operations.

    Parameters
    ----------
    var : Var
        A TVM variable

    Attributes
    ----------
    var : Var
        The wrapped TVM variable

    dtype : Type
        The data type of the scalar

    See Also
    --------
    heterocl.placeholder

    Examples
    --------
    .. code-block:: python

        # use () to specify it is a non-mutable scalar
        a = hcl.placeholder((), "a")
        # direct access
        b = a + 5
        # bit operations
        c = a[2] # the third bit of a
        d = a[3:5] # get a slice of a
    """
    def __init__(self, var):
        self.var = var

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            return _make.GetSlice(self.var, indices.start, indices.stop)
        elif isinstance(indices, (int, _expr.Expr)):
            return _make.GetBit(self.var, indices)
        else:
            raise TensorError("Invalid index")

    @property
    def name(self):
        return self.var.name

    @property
    def dtype(self):
        return self.var.dtype

    def same_as(self, var):
        if isinstance(var, Scalar):
            return self.var.same_as(var.var)
        elif isinstance(var, _expr.Expr):
            return self.var.same_as(var)
        return False

    def asnode(self):
        return self.var

class TensorSlice(NodeGeneric, _expr.ExprOp):
    """A helper class for tensor operations.

    Valid tensor accesses include: 1. getting an element from a tensor 2. bit
    operations on the element. We **do not** support operations on a slice of
    tensor.

    Parameters
    ----------
    tensor : Tensor
        The target tensor

    indices : int or tuple of int
        The indices to access the tensor

    Attributes
    ----------
    tensor : Tensor
        The target tensor

    indices : int or tuple of int
        The indices to access the tensor

    dtype : Type
        The data type of the tensor

    Examples
    --------
    .. code-block:: python

        A = hcl.placeholder((10,), "A")
        # get a single element
        a = A[5]
        # bit operations on a single element
        b = A[5][2]
        c = A[5][3:7]

        # not allowed: A[5:7]
    """
    def __init__(self, tensor, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self.tensor = tensor
        self.indices = indices

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorSlice(self.tensor, self.indices + indices)

    def __setitem__(self, indices, expr):
        if not isinstance(indices, tuple):
            indices = (indices,)
        indices = self.indices + indices
        index, bit, _ = util.get_index(self.tensor.shape, indices, 0)
        if not Stage.get_len():
            raise TensorError("Cannot set tensor elements without compute APIs")
        builder = Stage.get_current()
        if bit is None:
            builder.emit(_make.Store(self.tensor.buf.data,
                                     _make.Cast(self.tensor.dtype, expr),
                                     index))
        elif isinstance(bit, slice):
            load = _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
            expr = _make.SetSlice(load, expr, bit.start, bit.stop)
            builder.emit(_make.Store(self.tensor.buf.data,
                                     _make.Cast(self.tensor.dtype, expr),
                                     index))
        else:
            load = _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
            expr = _make.SetBit(load, expr, bit)
            builder.emit(_make.Store(self.tensor.buf.data,
                                     _make.Cast(self.tensor.dtype, expr),
                                     index))

    @property
    def dtype(self):
        return self.tensor.dtype

    def asnode(self):
        if len(self.indices) < len(self.tensor.shape):
            raise TensorError("Accessing a slice of tensor is not allowed")
        index, bit, _ = util.get_index(self.tensor.shape, self.indices, 0)
        if bit is None:
            return _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
        elif isinstance(bit, slice):
            return _make.GetSlice(_make.Load(self.tensor.dtype, self.tensor.buf.data, index),
                                  bit.start,
                                  bit.stop)
        return _make.GetBit(_make.Load(self.tensor.dtype, self.tensor.buf.data, index), bit)

class Tensor(NodeGeneric, _expr.ExprOp):
    """A HeteroCL tensor.

    This is a wrapper for a TVM tensor. It should be generated from HeteroCL
    compute APIs.

    Parameters
    ----------
    shape : tuple of int
        The shape of the tensor

    dtype : Type, optional
        The data type of the tensor

    name : str, optional
        The name of the tensor

    buf : Buffer, optional
        The TVM buffer of the tensor

    Attributes
    ----------
    dtype : Type
        The data type of the tensor

    name : str
        The name of the tensor

    var_dict : dict(str, Var)
        A dictionary that maps between a name and a variable

    first_update : Stage
        The first stage that updates the tensor

    last_update : Stage
        The last stage that updates the tensor

    tensor : Operation
        The TVM tensor

    buf : Buffer
        The TVM buffer

    type : Type
        The data type in HeteroCL format

    op : Stmt
        The operation statement

    axis : list of IterVar
        A list of axes of the tensor

    v : Expr
        Syntactic sugar to access the element of an single-element tensor

    See Also
    --------
    heterocl.placeholder, heterocl.compute
    """

    __hash__ = NodeGeneric.__hash__

    def __init__(self, shape, dtype="int32", name="tensor", buf=None):
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

    def __getitem__(self, indices):
        indices = util.CastRemover().mutate(indices)
        if Stage.get_len():
            Stage.get_current().input_stages.add(self.last_update)
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorSlice(self, indices)

    def __setitem__(self, indices, expr):
        indices = util.CastRemover().mutate(indices)
        Stage.get_current().input_stages.add(self.last_update)
        Stage.get_current().lhs_tensors.add(self)
        if not isinstance(indices, tuple):
            indices = (indices,)
        indices = util.CastRemover().mutate(indices)
        if len(indices) < len(self.shape):
            raise TensorError("Accessing a slice of tensor is not allowed")
        else:
            index, bit, _ = util.get_index(self.shape, indices, 0)
            if not Stage.get_len():
                raise TensorError("Cannot set tensor elements without compute APIs")
            builder = Stage.get_current()
            if bit is None:
                builder.emit(_make.Store(self.buf.data,
                                         _make.Cast(self.dtype, expr),
                                         index))
            elif isinstance(bit, slice):
                load = _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
                expr = _make.SetSlice(load, expr, bit.start, bit.stop)
                builder.emit(_make.Store(self.tensor.buf.data,
                                         _make.Cast(self.tensor.dtype, expr),
                                         index))
            else:
                load = _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
                expr = _make.SetBit(load, expr, bit)
                builder.emit(_make.Store(self.tensor.buf.data,
                                         _make.Cast(self.tensor.dtype, expr),
                                         index))

    @property
    def tensor(self):
        return self._tensor

    @property
    def buf(self):
        return self._buf

    @property
    def type(self):
        return types.dtype_to_hcl(self.dtype)

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
        """Set the TVM buffer.

        Parameters
        ----------
        buf : Buffer
        """
        self._buf = buf
        Tensor.tensor_map[self._tensor] = buf

    @tensor.setter
    def tensor(self, tensor):
        """Set the TVM tensor.

        Parameters
        ----------
        tensor : Tensor
        """
        self._tensor = tensor

    @v.setter
    def v(self, value):
        """A syntactic sugar for setting the value of a single-element tensor.

        This is the same as using `a[0]=value`, where a is a single-element tensor.

        Parameters
        ----------
        value : Expr
            The value to be set
        """
        self.__setitem__(0, value)

    def asnode(self):
        if len(self.shape) == 1 and self.shape[0] == 1:
            return TensorSlice(self, 0).asnode()
        else:
            raise ValueError("Cannot perform expression on Tensor")

    def same_as(self, tensor):
        if isinstance(tensor, Tensor):
            return self._tensor.same_as(tensor.tensor)
        return False
