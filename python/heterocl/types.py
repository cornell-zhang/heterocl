"""Define HeteroCL data types"""
#pylint: disable=too-few-public-methods, too-many-return-statements
import numbers
from collections import OrderedDict
from .debug import DTypeError

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
        self.bits = bits
        self.fracs = fracs

class Int(Type):
    """Arbitrary-bit signed integers"""
    def __repr__(self):
        return "Int(" + str(self.bits) + ")"

class UInt(Type):
    """Arbitrary-bit unsigned integers"""
    def __repr__(self):
        return "UInt(" + str(self.bits) + ")"

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
        self.dtype_dict = OrderedDict(dtype_dict)
        self.bits = 0
        for dtype in dtype_dict.values():
            self.bits += dtype.bits
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
        else: # Float
            return "float" + str(dtype.bits)
    else:
        if not isinstance(dtype, str):
            raise DTypeError("Unsupported data type format")
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
            raise DTypeError("Unrecognized data type")
    else:
        raise DTypeError("Unrecognized data type format")

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
