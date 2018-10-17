"""Functions defined in TVM."""
# pylint: disable=invalid-name,unused-import,redefined-builtin
from __future__ import absolute_import as _abs

from numbers import Integral as _Integral

from ._ffi.base import string_types
from ._ffi.node import register_node, NodeBase
from ._ffi.node import convert_to_node as _convert_to_node
from ._ffi.function import Function
from ._ffi.function import _init_api, register_func, get_global_func, extract_ext_funcs
from ._ffi.function import convert_to_tvm_func as _convert_tvm_func
from ._ffi.runtime_ctypes import TVMType
from . import _api_internal
from . import make as _make
from . import expr as _expr
from . import tensor as _tensor
from . import schedule as _schedule
from . import container as _container
from . import tag as _tag

int8 = "int8"
int32 = "int32"
float32 = "float32"
handle = "handle"


def min_value(dtype):
    """minimum value of dtype"""
    return _api_internal._min_value(dtype)


def max_value(dtype):
    """maximum value of dtype"""
    return _api_internal._max_value(dtype)


def const(value, dtype=None):
    """construct a constant"""
    if dtype is None:
        if isinstance(value, _Integral):
            dtype = 'int32'
        else:
            dtype = 'float32'
    return _api_internal._const(value, dtype)


def convert(value):
    """Convert value to TVM node or function.

    Parameters
    ----------
    value : python value

    Returns
    -------
    tvm_val : Node or Function
        Converted value in TVM
    """
    if isinstance(value, (Function, NodeBase)):
        return value

    if callable(value):
        return _convert_tvm_func(value)

    return _convert_to_node(value)


def load_json(json_str):
    """Load tvm object from json_str.

    Parameters
    ----------
    json_str : str
        The json string

    Returns
    -------
    node : Node
        The loaded tvm node.
    """
    return _api_internal._load_json(json_str)


def save_json(node):
    """Load tvm object as json string.

    Parameters
    ----------
    node : Node
        A TVM Node object to be saved.

    Returns
    -------
    json_str : str
        Saved json string.
    """
    return _api_internal._save_json(node)

def any(*args):
    """Create a new experssion of the union of all conditions in the arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    ret = _make.Or(args[0], args[1])
    for i in range(2, len(args)):
        ret = _make.Or(ret, args[i])
    return ret


def all(*args):
    """Create a new experssion of the intersection of all conditions in the
      arguments

    Parameters
    ----------
    args : list
        List of symbolic boolean expressions

    Returns
    -------
    expr: Expr
        Expression
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    ret = _make.And(args[0], args[1])
    for i in range(2, len(args)):
        ret = _make.And(ret, args[i])
    return ret

def decl_buffer(shape,
                dtype=None,
                name="buffer",
                data=None,
                strides=None,
                elem_offset=None,
                scope="",
                data_alignment=-1,
                offset_factor=0):
    """Decleare a new symbolic buffer.

    Normally buffer is created automatically during lower and build.
    This is only needed if user want to specify their own buffer layout.

    See the note below for detailed discussion on usage of buffer.

    Parameters
    ----------
    shape : tuple of Expr
        The shape of the buffer.

    dtype : str, optional
        The data type of the buffer.

    name : str, optional
        The name of the buffer.

    data : Var, optional
        The data pointer in the buffer.

    strides: array of Expr
        The stride of the buffer.

    elem_offset: Expr, optional
        The beginning offset of the array to data.
        In terms of number of elements of dtype.

    scope: str, optional
        The storage scope of the buffer, if not global.
        If scope equals empty string, it means it is global memory.

    data_alignment: int, optional
        The alignment of data pointer in bytes.
        If -1 is passed, the alignment will be set to TVM's internal default.

    offset_factor: int, optional
        The factor of elem_offset field, when set,
        elem_offset is required to be multiple of offset_factor.
        If 0 is pssed, the alignment will be set to 1.
        if non-zero is passed, we will created a Var for elem_offset if elem_offset is not None.

    Returns
    -------
    buffer : Buffer
        The created buffer

    Note
    ----
    Buffer data structure reflects the DLTensor structure in dlpack.
    While DLTensor data structure is very general, it is usually helpful
    to create function that only handles specific case of data structure
    and make compiled function benefit from it.

    If user pass strides and elem_offset is passed as None
    when constructing the function, then the function will be specialized
    for the DLTensor that is compact and aligned.
    If user pass a fully generic symbolic array to the strides,
    then the resulting function becomes fully generic.
    """
    shape = (shape,) if isinstance(shape, (_expr.Expr, _Integral)) else shape
    dtype = float32 if dtype is None else dtype
    strides = () if strides is None else strides
    if offset_factor != 0 and elem_offset is None:
        elem_offset = _api_internal._Var('%s_elem_offset' % name, shape[0].dtype)
    if data is None:
        data = _api_internal._Var(name, "handle")
    return _api_internal._Buffer(
        data, dtype, shape, strides, elem_offset, name, scope,
        data_alignment, offset_factor)


def _IterVar(dom, name, iter_type, thread_tag=''):
    """Internal function to create IterVar

    Parameters
    ----------
    dom : Range
        The domain of iteration.

    name : str
        The name of iteration variable.

    iter_type : int
        The type of iteration.

    thread_tag : str
        The thread tag of the iteration variable.

    Returns
    -------
    iter_var : IterVar
       The result itervar
    """
    if dom is not None:
        if isinstance(dom, (list, tuple)):
            if len(dom) != 2:
                raise TypeError("need to be list of ranges")
            dom = Range(dom[0], dom[1])

        if not isinstance(dom, _container.Range):
            raise TypeError("dom need to be Range")
    name = name if name else 'iter'
    v = _api_internal._Var(name, "int32")
    return _api_internal._IterVar(dom, v, iter_type, thread_tag)




def select(cond, t, f):
    """Construct a select branch
    Parameters
    ----------
    cond : Expr
        The condition
    t : Expr
        The result expression if cond is true.
    f : Expr
        The result expression if cond is false.

    Returns
    -------
    node : Node
        The tvm.expr.Select node
    """
    return _make.Select(convert(cond), convert(t), convert(f))


_init_api("tvm.api")
