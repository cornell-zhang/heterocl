"""Define HeteroCL data types"""
# pylint: disable=too-few-public-methods, too-many-return-statements
import numbers
import types as python_types
from collections import OrderedDict
from hcl_mlir.ir import Type as mlir_type
from hcl_mlir import mlir_type_to_str
from hcl_mlir.exceptions import *

class Type(object):
    """The base class for all data types

    The default bitwidth is 32 and no fractional bit.

    Parameters
    ----------
    bits: int
        Number of total bits.
    fracs: int
        Number of fractional bits.
    """

    def __init__(self, bits=32, fracs=0):
        if not isinstance(bits, numbers.Integral):
            raise DTypeError("Bitwidth must be an integer.")
        if not isinstance(fracs, numbers.Integral):
            raise DTypeError("Number of fractional bits must be an integer.")
        if bits > 2047:
            raise DTypeError(
                "The maximum supported total bitwidth is 2047 bits.")
        if fracs > 255:
            raise DTypeError(
                "The maximum supported fractional bitwidth is 255 bits.")
        self.bits = bits
        self.fracs = fracs

    def __eq__(self, other):
        if other == None:
            return False
        other = dtype_to_hcl(other)
        return other.bits == self.bits and other.fracs == self.fracs


class Int(Type):
    """Arbitrary-bit signed integers"""

    def __repr__(self):
        return "Int(" + str(self.bits) + ")"


class UInt(Type):
    """Arbitrary-bit unsigned integers"""

    def __repr__(self):
        return "UInt(" + str(self.bits) + ")"


class Index(UInt):
    """Index type"""

    def __init__(self):
        super(Index, self).__init__(32)

    def __repr__(self):
        return "Index"


class Float(Type):
    """Floating points"""

    def __repr__(self):
        return "Float(" + str(self.bits) + ")"


class Fixed(Type):
    """Arbitrary-bit signed fixed points"""

    def __repr__(self):
        return "Fixed(" + str(self.bits) + ", " + str(self.fracs) + ")"


class UFixed(Type):
    """Arbitrary-bit unsigned fixed points"""

    def __repr__(self):
        return "UFixed(" + str(self.bits) + ", " + str(self.fracs) + ")"


class Struct(Type):
    """A C-like struct

    The struct members are defined with a Python dictionary
    """

    def __init__(self, dtype_dict):
        self.bits = 0
        for name, dtype in dtype_dict.items():
            dtype = dtype_to_hcl(dtype)
            dtype_dict[name] = dtype
            self.bits += dtype.bits
        self.dtype_dict = OrderedDict(dtype_dict)
        Type.__init__(self, self.bits, 0)

    def __repr__(self):
        return "Struct(" + str(self.dtype_dict) + ")"

    def __getattr__(self, key):
        try:
            return self.dtype_dict[key]
        except KeyError:
            raise DTypeError(key + " is not in struct")

    def __getitem__(self, key):
        return self.__getattr__(key)

def dtype_to_str(dtype):
    """Convert a data type to string format.

    This method is mainly for TVM APIs.

    Parameters
    ----------
    dtype : Type or str
        The data type to be converted

    Returns
    -------
    str
        The converted data type in string format.
    """
    if isinstance(dtype, mlir_type):
        return mlir_type_to_str(dtype)
    if isinstance(dtype, Type):
        if isinstance(dtype, Int):
            return "int" + str(dtype.bits)
        # struct is treated as uint
        elif isinstance(dtype, (UInt, Struct)):
            return "uint" + str(dtype.bits)
        elif isinstance(dtype, Fixed):
            bits = dtype.bits
            fracs = dtype.fracs
            if fracs == 0:
                return "int" + str(bits)
            return "fixed" + str(bits) + "_" + str(fracs)
        elif isinstance(dtype, UFixed):
            bits = dtype.bits
            fracs = dtype.fracs
            if fracs == 0:
                return "uint" + str(bits)
            return "ufixed" + str(bits) + "_" + str(fracs)
        else:  # Float
            return "float" + str(dtype.bits)
    else:
        if not isinstance(dtype, str):
            raise DTypeError("Unsupported data type format: {}".format(dtype))
        return dtype


def dtype_to_hcl(dtype):
    """Convert a data type to Heterocl type.

    Parameters
    ----------
    dtype : Type or str
        The data type to be converted

    Returns
    -------
    Type
    """
    if isinstance(dtype, mlir_type):
        # convert mlir type to string
        dtype = mlir_type_to_str(dtype)
    if isinstance(dtype, Type):
        return dtype
    elif isinstance(dtype, str):
        if dtype[0:3] == "int":
            return Int(int(dtype[3:]))
        elif dtype[0:4] == "uint":
            return UInt(int(dtype[4:]))
        elif dtype[0:5] == "float":
            return Float()
        elif dtype[0:5] == "fixed":
            strs = dtype[5:].split('_')
            return Fixed(int(strs[0]), int(strs[1]))
        elif dtype[0:6] == "ufixed":
            strs = dtype[6:].split('_')
            return UFixed(int(strs[0]), int(strs[1]))
        else:
            raise DTypeError("Unrecognized data type: {}".format(dtype))
    else:
        raise DTypeError("Unrecognized data type format: {}".format(dtype))


def get_bitwidth(dtype):
    """Get the bitwidth of a given data type.

    Parameters
    ----------
    dtype : Type or str
        The given data type

    Returns
    -------
    int
    """
    dtype = dtype_to_hcl(dtype)
    return dtype.bits


def get_fractional_bitwidth(dtype):
    """Get the fractional bitwidth of a given data type.

    Parameters
    ----------
    dtype : Type or str
        The given data type

    Returns
    -------
    int
    """
    dtype = dtype_to_hcl(dtype)
    return dtype.fracs

def sort_type_classes(types):
    """Sort the types in the order of Int, UInt, Fixed, UFixed, Float, Struct.

    Parameters
    ----------
    types : list of Type
        The list of types to be sorted.

    Returns
    -------
    list of Type
        The sorted list of types.
    """
    if isinstance(types, tuple):
        types = list(types)
    elif not isinstance(types, list):
        raise DTypeError(f"sort_type_classes input should be a list or tuple, got {type(types)}")
    for t in types:
        if not isinstance(t, type):
            raise DTypeError(f"sort_type_classes input should be a list of types, got a list of {t} : {type(t)}")
        elif not issubclass(t, Type):
            raise DTypeError(f"sort_type_classes input should be a list of Type subclass, got {t}")
    type_classes = [Int, UInt, Index, Fixed, UFixed, Float, Struct]
    type_classes = [t.__name__ for t in type_classes]
    return sorted(types, key=lambda t: type_classes.index(t.__name__))

class TypeRule(object):
    """Type inference rule for a set of operations.
    """
    def __init__(self, OpClass, inf_rules):
        """
        Parameters
        ----------
        OpClass : a class or a collection of classes
            The operation class or a list of operation classes
        
        inf_rules : a dictionary or a collection of dictionaries
            The inference rules for the operation class
            Each item should be (input types, lambda function)
        """
        # Check argument types
        if isinstance(OpClass, type):
            OpClass = [OpClass]
        elif isinstance(OpClass, tuple):
            OpClass = list(OpClass)
        elif not isinstance(OpClass, list):
            raise TypeError(f"OpClass must be a class or a collection of classes, not {type(OpClass)}")
        
        if isinstance(inf_rules, dict):
            inf_rules = [inf_rules]
        elif not isinstance(inf_rules, tuple):
            inf_rules = list(inf_rules)
        elif not isinstance(inf_rules, list):
            raise TypeError(f"inf_rules must be a dict or a collection of dict, not {type(inf_rules)}")
        
        # A collection of applicable operations
        self.OpClass = OpClass
        # Inference rules
        self.inf_rules = dict()
        # a dictionary of the form:
        # { input types (tuple) : inference function (lambda func) }
        # merge the collection of inference rules into a single dictionary
        for rule_set in inf_rules:
            for itype, inf_rule in rule_set.items():
                # check itype type
                if not isinstance(itype, tuple):
                    raise TypeError(f"itype must be a tuple, not {type(itype)}")
                for t in itype:
                    if not isinstance(t, type):
                        raise TypeError(f"itype must be a tuple of Class, not {type(t)}")
                # check inf_rule type
                if not isinstance(inf_rule, python_types.LambdaType):
                    raise TypeError(f"inf_rule must be a lambda function, not {type(inf_rule)}")
                # sort the input types
                itype = tuple(sort_type_classes(itype))
                # check if the input types are already in the dictionary
                if itype in self.inf_rules:
                    raise TypeError(f"Duplicate inference rule for input types {itype}")
                # add the rule to the dictionary
                self.inf_rules[itype] = inf_rule

    def __call__(self, *args):
        """Call the inference rule with the given input types.
        
        It automatically finds the typing rule based on the input types.
        If no rule is found, it will raise an error.

        Parameters
        ----------
        args : list of input types

        Returns
        -------
        Type
            The inferred output type
        """
        itype_classes = sort_type_classes([type(t) for t in args])
        itype_classes = tuple(itype_classes)
        if itype_classes not in self.inf_rules:
            raise APIError(f"Typing rule is not defined for {self.OpClass} with input types {itype_classes}")
        rule = self.inf_rules[itype_classes]
        res_type = rule(*args)
        return res_type

    # def __repr__(self):
        # TODO: make type rule printable
        # pass