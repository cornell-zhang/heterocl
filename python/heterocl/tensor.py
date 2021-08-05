"""HeteroCL tensors and scalars."""
#pylint: disable=missing-docstring, too-many-instance-attributes
from .tvm import make as _make
from .tvm import expr as _expr
from .tvm import ir_pass as _pass
from .tvm.api import decl_buffer
from .tvm._ffi.node import NodeGeneric
from .debug import APIError, TensorError
from .schedule import Stage
from . import util
from . import debug
from . import types
from .tvm.api import _IterVar

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
        # get a slice of the tensor
        a = A[1:6]
        # bit operations on a single element
        b = A[5][2]
        c = A[5][3:7]
    """
    def __init__(self, tensor, indices, dtype=None):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self.tensor = tensor
        self.indices = indices
        self._dtype = dtype if dtype is not None else self.tensor.dtype
        nshape = len(self.tensor.shape)
        self._shape = list(self.tensor.shape)
        indices_slice = []
        shape_dict = {}
        # the key is the tensor dimension being altered by the slice 
        # the value is a list of indices of the slices in self.indices
        # corresponding to the tensor dimension
        shape_idx = 0
        x = 0
        while shape_idx < nshape and x < len(self.indices):
            i = self.indices[x]
            if isinstance(i, slice):
                if shape_idx not in shape_dict:
                    shape_dict[shape_idx] = []
                shape_dict[shape_idx].append(x)
                indices_slice.append(i)
                diff = i.stop - i.start
                self._shape[shape_idx] = diff
                if diff <= 0:
                    raise TensorError("Invalid index range")
            else:
                shape_idx += 1
            x += 1
        self._shape = tuple(self._shape)
        # is true when individual elements of the tensor are being accessed
        self.slice_offset = len(self.indices) - nshape - len(indices_slice) == 0

        # offsets the dimensions that are sliced
        if len(indices_slice) > 0 and self.slice_offset:
            i = 0
            for key, value in shape_dict.items():
                for x in range(0, len(value)):
                    index_slice = value[x] + len(value) - x 
                    tmp = list(self.indices)
                    tmp[index_slice] = self.indices[index_slice] + indices_slice[i].start
                    self.indices = tuple(tmp)
                    i += 1
        index, bit, _ = util.get_index(self.tensor.shape, self.indices, 0)
        # check if we have bit slicing
        if isinstance(bit, slice) and not isinstance(self.tensor.type, types.Struct):
            diff = bit.start - bit.stop
            if not isinstance(diff, int):
                diff = util.CastRemover().mutate(diff)
                diff = _pass.Simplify(diff)
            try:
                diff = int(diff)
                if diff < 0:
                    diff = -diff
                self._dtype = "uint" + str(diff)
            except:
                if isinstance(diff, (_expr.IntImm, _expr.UIntImm)):
                    diff = diff.value
                    if diff < 0:
                        diff = -diff
                    self._dtype = "uint" + str(diff)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        return TensorSlice(self.tensor, self.indices + indices)

    def __setitem__(self, indices, expr):
        if not isinstance(indices, tuple):
            indices = (indices,)
        indices = util.CastRemover().mutate(indices)
        indices = self.indices + indices
        off_shift = 0
        num_slice = 0
        # if the indices have not yet been shifted
        if not self.slice_offset:
            shift_index = False
            idx = 0
            tmp = list(indices)
            for i in indices:
                if shift_index and not isinstance(i, slice):
                    tmp[idx] = indices[idx] + off_shift
                    shift_index = False
                elif shift_index:
                    off_shift += i.start
                    num_slice += 1
                elif isinstance(i, slice):
                    shift_index = True
                    off_shift = i.start
                    num_slice += 1
                idx += 1
            indices = tuple(tmp)
        index, bit, index_acc = util.get_index(self.tensor.shape, indices, 0)
        if not Stage.get_len():
            raise TensorError("Cannot set tensor elements without compute APIs")
        builder = Stage.get_current()
        emit_dtype = None
        if bit is None:
            emit_dtype = self._dtype
        elif isinstance(bit, slice):
            load = _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
            # special handle for struct: we need to make sure the bitwidths
            # are the same before and after bitcast
            if (isinstance(self.tensor.type, types.Struct)
                    and util.get_type(self._dtype)[0] != "uint"):
                ty = "uint" + str(util.get_type(self._dtype)[1])
                expr = _make.Call(ty, "bitcast",
                                  [expr], _expr.Call.PureIntrinsic, None, 0)
            expr = _make.SetSlice(load, expr, bit.start, bit.stop)
            emit_dtype = self.tensor.dtype
        else:
            load = _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
            expr = _make.SetBit(load, expr, bit)
            emit_dtype = self._dtype

        set_dim = len(indices) - num_slice
        if isinstance(index, slice):
            index = 0
        if isinstance(indices[-1], slice) and set_dim < len(self.tensor.shape)\
                        and isinstance(expr, (TensorSlice, Tensor)):
            st = _make.Add(_make.Mul(index, index_acc, False),
                        _make.Div(_make.Mul(off_shift, index_acc, False),
                        self.tensor.shape[set_dim], False), False)
            acc = 1
            for x in self.tensor.shape[(set_dim + 1):]:
                acc *= x
            iv = [0 for n in range(set_dim, len(self.tensor.shape) - 1)]
            iv.append(_IterVar((0, (indices[-1].stop - indices[-1].start) * acc), "set_array_var", 0))
            stmt = _make.Store(self.tensor._buf.data,
                                _make.Cast(emit_dtype, expr[tuple(iv)]),
                                iv[-1] + st)
            for_stmt = util.make_for([iv[-1]], stmt, 0, "tensorslice_set_array")
            builder.emit(for_stmt)
        elif isinstance(indices[-1], slice) and set_dim < len(self.tensor.shape):
            st = _make.Add(_make.Mul(index, index_acc, False),
                        _make.Div(_make.Mul(off_shift, index_acc, False),
                        self.tensor.shape[set_dim], False), False)
            acc = 1
            for x in self.tensor.shape[(set_dim + 1):]:
                acc *= x
            iv = [_IterVar((0, (indices[-1].stop - indices[-1].start) * acc), "broadcast_var", 0)]
            stmt = _make.Store(self.tensor._buf.data,
                                _make.Cast(emit_dtype, expr),
                                iv[0] + st)
            for_stmt = util.make_for(iv, stmt, 0, "tensorslice_broadcast")
            builder.emit(for_stmt)
        else:
            builder.emit(_make.Store(self.tensor.buf.data,
                                     _make.Cast(emit_dtype, expr),
                                     index))

    def __getattr__(self, key):
        if key in ('__array_priority__', '__array_struct__'):
            raise APIError(
                    "Cannot use NumPy numbers as left-hand-side operand")
        hcl_dtype = self.tensor.hcl_dtype
        if not isinstance(hcl_dtype, types.Struct):
            raise TensorError(
                    "Cannot access attribute if type is not struct")
        start = 0
        end = 0
        dtype = None
        for dkey, dval in hcl_dtype.dtype_dict.items():
            if dkey == key:
                end = start + dval.bits
                dtype = types.dtype_to_str(dval)
                break
            else:
                start += dval.bits
        if dtype is None:
            raise DTypeError("Field " + key
                             + " is not in struct " + str(hcl_dtype))
        indices = (slice(end, start),)
        return TensorSlice(self.tensor, self.indices + indices, dtype)

    def __setattr__(self, key, expr):
        if key in ("tensor", "indices", "_dtype"):
            super().__setattr__(key, expr)
        elif key == "_shape" or key == "slice_offset":
            self.__dict__[key] = expr
        else:
            hcl_dtype = self.tensor.hcl_dtype
            if not isinstance(hcl_dtype, types.Struct):
                raise TensorError(
                        "Cannot access attribute if type is not struct")
            start = 0
            end = 0
            for dkey, dval in hcl_dtype.dtype_dict.items():
                if dkey == key:
                    end = start + dval.bits
                    self._dtype = types.dtype_to_str(dval)
                    break
                else:
                    start += dval.bits
            if start == end:
                raise DTypeError("Field " + key
                                 + " is not in struct " + str(hcl_dtype))
            indices = (slice(end, start),)
            self.__setitem__(indices, expr)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        num_slice = sum(isinstance(x, slice) for x in self.indices)
        idx = len(self.indices) - num_slice
        if idx > len(self.tensor.shape):
            raise TensorError("Shape is not defined when the length of indices"
                              + " is greater than the number of dimensions")
        return self._shape[idx:]

    def asnode(self):
        self.indices = util.CastRemover().mutate(self.indices)
        index, bit, _ = util.get_index(self.tensor.shape, self.indices, 0)
        if bit is None:
            return _make.Load(self._dtype, self.tensor.buf.data, index)
        elif isinstance(bit, slice):
            load = _make.GetSlice(_make.Load(self.tensor.dtype,
                                             self.tensor.buf.data, index),
                                  bit.start,
                                  bit.stop)
            if self.tensor.dtype != self._dtype:
                if (isinstance(self.tensor.type, types.Struct)
                        and util.get_type(self._dtype)[0] != "uint"):
                    bw_from = types.get_bitwidth(self.tensor.dtype)
                    bw_to = types.get_bitwidth(self._dtype)
                    if bw_from != bw_to:
                        ty = util.get_type(self.tensor.dtype)[0] + str(bw_to)
                        load = _make.Cast(ty, load)
                    load = _make.Call(self._dtype, "bitcast",
                                  [load], _expr.Call.PureIntrinsic, None, 0)
                else:
                    load = _make.Cast(self._dtype, load)
            return load
        return _make.GetBit(_make.Load(self._dtype,
                                       self.tensor.buf.data,
                                       index), bit)

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
        self.hcl_dtype = types.dtype_to_hcl(dtype)
        self.dtype = types.dtype_to_str(dtype)
        self.shape = shape
        self.name = name
        self.var_dict = {}
        self.first_update = None
        self.last_update = None
        if buf is None:
            self._buf = decl_buffer(shape, self.dtype, name)

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
        index, bit, index_acc = util.get_index(self.shape, indices, 0)
        if not Stage.get_len():
                raise TensorError("Cannot set tensor elements without compute APIs")
        builder = Stage.get_current()
        if isinstance(index, slice):
            r_acc = _make.Div(index_acc, self.shape[0], False)
            iv_bound = _make.Mul((index.stop-index.start), r_acc, False)
            ivs = [_IterVar((0, iv_bound), "tensor_broadcast", 0)]
            stmt_index = _make.Add(ivs[0], _make.Mul(index.start, r_acc, False), False)
            if isinstance(expr, (TensorSlice, Tensor)):
                stmt = _make.Store(self._buf.data, expr[ivs[0]], stmt_index)
            else:
                stmt = _make.Store(self._buf.data, expr, stmt_index)
            for_stmt = util.make_for(ivs, stmt, 0, "tensor_set_array_loop")
            builder.emit(for_stmt)
        elif bit is None:
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
        return self.hcl_dtype

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
