"""Define HeteroCL data types"""
import numbers
from .debug import DTypeError

#pylint: disable=too-few-public-methods
class Type(object):
    """The base class for all data types

    The defualt bitwidth is 32 and no fractional bit.

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
