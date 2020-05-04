"""Helper functions for NumPy arrays."""
#pylint: disable=missing-docstring
import numpy as np
from .tvm.ndarray import array, cpu
from .util import get_tvm_dtype
from . import types

def cast_np(np_in, dtype):
    """Cast a NumPy array to a specified data type.

    Parameters
    ----------
    np_in : ndarray
        The array to be cast

    dtype : Type
        The target data type

    Returns
    -------
    ndarray

    Examples
    --------
    .. code-block:: python

        A_float = numpy.random.rand(10)
        A_fixed = hcl.cast_np(A_float, hcl.Fixed(10, 8))
    """
    def cast(val):
        if isinstance(dtype, (types.Fixed, types.UFixed)):
            bits = dtype.bits
            fracs = dtype.fracs
            val = int(val * (1 << fracs))
            mod = val % (1 << bits)
            if isinstance(dtype, types.Fixed):
                val = mod if mod < (1 << (bits-1)) else mod - (1 << bits)
            else:
                val = mod
            val = float(val) / (1 << fracs)
            return val
        elif isinstance(dtype, (types.Int, types.UInt)):
            bits = dtype.bits
            mod = int(val) % (1 << bits)
            if isinstance(dtype, types.Int):
                val = mod if mod < (1 << (bits-1)) else mod - (1 << bits)
            else:
                val = mod
            return val
        return val

    vfunc = np.vectorize(cast)
    return vfunc(np_in)

def asarray(arr, dtype=None, ctx=cpu(0)):
    """Convert a NumPy array to a HeteroCL array.

    The converted array will be inputs to the executable.

    Parameters
    ----------
    arr : ndarray
        The array to be converted

    dtype : Type, optional
        The target data type

    ctx : TVMContext
        The target context

    Returns
    -------
    NDArray

    Examples
    --------
    .. code-block:: python

        np_A = numpy.zeros(10)
        hcl_A = np_A.asarray()
    """
    dtype = get_tvm_dtype(dtype)
    return array(arr, dtype, ctx)

def pack_np(np_in, dtype_in, dtype_out):
    """Pack a NumPy array according to the specified data types.

    Now we only support packing and unpacking for a 1-dimensional array.

    Parameters
    ----------
    np_in : ndarray
        The array to be packed

    dtype_in : Type
        The data type of the input array

    dtype_out : Type
        The target data type

    Returns
    -------
    ndarray

    Examples
    --------
    .. code-block:: python

        a = numpy.random.randint(16, size=(10,))
        packed_a = hcl.pack_np(np_in, hcl.UInt(8), hcl.UInt(32))
    """
    factor = dtype_out.bits / dtype_in.bits
    fracs = dtype_in.fracs
    shape = np_in.shape
    np_out = []
    signed = True
    if isinstance(dtype_in, (types.UInt, types.UFixed)):
        signed = False
    for i in range(0, shape[0]/factor):
        num = 0
        for j in range(0, factor):
            val = int(np_in[i*factor + j] * (1 << fracs))
            if signed:
                val = val if val >= 0 else val + (1 << dtype_in.bits)
            num += val << (j * dtype_in.bits)
        np_out.append(num)

    return np.array(np_out)

def unpack_np(np_in, dtype_in, dtype_out):
    """Unpack a NumPy array according to the specified data types.

    Now we only support packing and unpacking for a 1-dimensional array.

    Parameters
    ----------
    np_in : ndarray
        The array to be unpacked

    dtype_in : Type
        The data type of the input array

    dtype_out : Type
        The target data type

    Returns
    -------
    ndarray

    Examples
    --------
    .. code-block:: python

        a = numpy.random.randint(4, size=(10,))
        unpacked_a = hcl.pack_np(np_in, hcl.UInt(32), hcl.UInt(8))
    """
    factor = dtype_in.bits / dtype_out.bits
    fracs = dtype_out.fracs
    shape = np_in.shape
    np_out = []
    signed = True
    if isinstance(dtype_out, (types.UInt, types.UFixed)):
        signed = False
    for i in range(0, shape[0] * factor):
        num = int(np_in[i/factor]) >> (dtype_out.bits * (i%factor))
        num = num % (1 << dtype_out.bits)
        if signed:
            num = num if num < 1 << (dtype_out.bits - 1) else num - (1 << dtype_out.bits)
        np_out.append(float(num) / (1 << fracs))
    return np.array(np_out)
