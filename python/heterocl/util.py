"""Utility functions for HeteroCL"""
from .tvm import make as _make
from .tvm import expr as _expr
from .tvm.expr import Var, Call
from .tvm.api import _IterVar, decl_buffer
from . import types
from . import devices
from . import config
from .scheme import Scheme
from .debug import DTypeError
from .mutator import Mutator

class VarName():
    """A counter for each type of variables.

    Parameters
    ----------
    name_dict: dict
        A dictionary whose key is the variable type and whose value is
        the number of such variable.
    """
    name_dict = {}

def get_name(var_type, name=None):
    """Get the name of a given type of variable.

    If the name is not given, this function automatically generates a
    name according to the given type of variable.

    Parameters
    ----------
    var_type: str
        The type of the variable in string.

    name: str, optional
        The name specified by the user.

    Returns
    -------
    new_name: str
        The name of the variable.
    """
    if name is not None:
        return name
    else:
        if VarName.name_dict.get(var_type) is None:
            VarName.name_dict[var_type] = 0
            return var_type + "0"
        else:
            counter = VarName.name_dict[var_type] + 1
            VarName.name_dict[var_type] = counter
            return var_type + str(counter)

def get_dtype(dtype, name=None):
    """Get the data type by default or from a value.

    We first check if the data type of a variable is specified after
    the scheduling or the variable is used for the first time. After
    that, we check whether user specifies the data type or not.

    Parameters
    ----------
    dtype: Type or str or None
        The specified data type.

    name: str, optional
        The name of the variable that will be given a data type.

    Returns
    -------
    dtype: str
        A data type represented in str.
    """
    if Scheme.current is not None:
        dtype_ = Scheme.current.dtype_dict.get(name)
        dtype = dtype if dtype_ is None else dtype_
    dtype = config.init_dtype if dtype is None else dtype
    return dtype

def get_tvm_dtype(dtype, name=None):
    return types.dtype_to_str(get_dtype(dtype, name))

def true():
    return _make.UIntImm("uint1", 1)

def make_for(indices, body, level):
        iter_var = indices[level]
        if level == len(indices) - 1:
            body = _make.AttrStmt(iter_var, "loop_scope", iter_var.var, body)
            return _make.For(iter_var.var, iter_var.dom.min, iter_var.dom.extent, 0, 0, body)
        else:
            body = _make.AttrStmt(iter_var, "loop_scope", iter_var.var, make_for(indices, body, level+1))
            return _make.For(iter_var.var, iter_var.dom.min, iter_var.dom.extent, 0, 0, body)

# return (index, bit, _)
def get_index(shape, args, level):
    if level == len(args) - 1: # the last arg
        if level == len(shape): # bit-selection
            return (0, args[level], 1)
        else:
            return (args[level], None, shape[level])
    else:
        index = get_index(shape, args, level+1)
        new_arg = args[level]
        new_index = _make.Add(index[0],
                _make.Mul(new_arg, index[2], False), False)
        new_acc = _make.Mul(index[2], shape[level], False)
        return (new_index, index[1], new_acc)

def get_type(dtype):
    if dtype[0:3] == "int":
        return "int", int(dtype[3:])
    elif dtype[0:4] == "uint":
        return "uint", int(dtype[4:])
    elif dtype[0:5] == "float":
        return "float", int(dtype[5:])
    elif dtype[0:5] == "fixed":
        strs = dtype[5:].split('_')
        return "fixed", int(strs[0]), int(strs[1])
    elif dtype[0:6] == "ufixed":
        strs = dtype[6:].split('_')
        return "ufixed", int(strs[0]), int(strs[1])
    else:
        raise ValueError("Unknown data type: " + dtype)

class CastRemover(Mutator):

    def mutate_ConstExpr(self, node):
        return node.value

    def mutate_BinOp(self, binop, node):
        a = self.mutate(node.a)
        b = self.mutate(node.b)
        if isinstance(a, _expr.ConstExpr):
            a = a.value
        if isinstance(b, _expr.ConstExpr):
            b = b.value
        return binop(a, b, False)

    def mutate_Cast(self, node):
        return self.mutate(node.value)
