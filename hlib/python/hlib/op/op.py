from collections import OrderedDict
import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
import hlib
#from .nn import *

dtype = hcl.Float()

max = hcl.reducer(-10000, lambda x, y: tvm.make.Max(x, y), dtype)
min = hcl.reducer(10000, lambda x, y: tvm.make.Min(x, y), dtype)
sum = hcl.reducer(0, lambda x, y: x + y, dtype)
prod = hcl.reducer(1, lambda x, y: x * y, dtype)

# unary operations


def abs(array, dtype=None, name='abs'):
    if dtype is None:
        dtype = array.dtype
    return hcl.compute(array.shape, lambda *x: hcl.select(
        array[x] < 0, -array[x], array[x]), dtype=dtype, name=name)


def negative(array, dtype=None, name='negative'):
    if dtype is None:
        dtype = array.dtype
    return hcl.compute(array.shape, lambda *x: -
                       array[x], dtype=dtype, name=name)


def cast(array, dtype=None, name='cast'):
    return hcl.compute(array.shape, lambda *x: hcl.cast(dtype, array[x]), dtype=dtype, name=name)


def expand_dims(data, axis, new_axis, name="expand_dims"):
    shape = []
    val_var = []
    ind_len = len(data.shape)
    if(axis > ind_len):
        axis = ind_len
    for i in range(axis):
        shape.append(data.shape[i])
        val_var.append(1)
    for i in range(new_axis):
        shape.append(1)
        val_var.append(0)
    for i in range(ind_len - axis):
        shape.append(data.shape[i + axis])
        val_var.append(1)
    shape = tuple(shape)

    def _expand_ind(val_var, *indices):
        indices = indices[0]
        new_shape = []
        for i in range(len(val_var)):
            if val_var[i]:
                new_shape.append(indices[i])
        return tuple(new_shape)
    return hcl.compute(
        shape, lambda *x: data[_expand_ind(val_var, x)], name=name)

# elemwise functions

def logical_and(input1, input2, name='logical_and'):
    return hcl.compute(
        input1.shape,
        lambda *x: input1[x] & input2[x],
        name=name,
        dtype=input1.dtype)


def logical_or(input1, input2, name='logical_or'):
    return hcl.compute(
        input1.shape,
        lambda *x: input1[x] | input2[x],
        name=name)


def logical_not(input1, name='logical_not'):
    return hcl.compute(input1.shape, lambda *x: ~input1[x], name=name)


def elemwise_add(input1, input2, name='elemwise_add'):
    return hcl.compute(
        input1.shape,
        lambda *x: input1[x] + input2[x],
        name=name)


def elemwise_sub(input1, input2, name='elemwise_sub'):
    return hcl.compute(
        input1.shape,
        lambda *x: input1[x] - input2[x],
        name=name)


def elemwise_mul(input1, input2, name='elemwise_mul'):
    return hcl.compute(
        input1.shape,
        lambda *x: input1[x] * input2[x],
        name=name)


def elemwise_div(input1, input2, name='elemwise_div'):
    return hcl.compute(
        input1.shape,
        lambda *x: input1[x] / input2[x],
        name=name)


def elemwise_mod(input1, input2, name='elemwise_mod'):
    return hcl.compute(
        input1.shape, lambda *x: input1[x] %
        input2[x], name=name)


def elemwise_pow(input1, input2, name='elemwise_pow'):
    return hcl.compute(
        input1.shape, lambda *x: hcl.power(input1[x], input2[x]), name=name)

# broadcast functions


def _broadcast(shape, *indices):
    axes = []
    indices = indices[0]
    for i in range(len(shape)):
        if(shape[i] == 1):
            axes.append(0)
        else:
            axes.append(indices[i])
    axes = tuple(axes)
    return axes


def broadcast_to(input1, out_shape, name='broadcast_to'):
    assert(len(input1.shape) == len(out_shape))
    return hcl.compute(
        out_shape, lambda *x: input1[_broadcast(out_shape, x)], name=name)


def broadcast_set(input1, input2):
    len1 = len(input1.shape)
    len2 = len(input2.shape)
    if(len1 < len2):
        return hlib.op.nn.expand_dims(input1, len1, len2 - len1), input2, False
    elif(len2 < len1):
        return input1, hlib.op.nn.expand_dims(input2, len2, len1 - len2), True
    else:
        for i in input1.shape:
            for j in input2.shape:
                if(i < j):
                    return input1, input2, False
                elif(j < i):
                    return input1, input2, True
        return input1, input2, True


def broadcast_add(input1, input2, name='broadcast_add'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(
            input1_mod.shape, lambda *x: input1_mod[x] + input2_mod[_broadcast(input2_mod.shape, x)], name=name)
    else:
        return hcl.compute(
            input2_mod.shape, lambda *x: input1_mod[_broadcast(input1_mod.shape, x)] + input2_mod[x], name=name)


def broadcast_sub(input1, input2, name='broadcast_sub'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(
            input1_mod.shape, lambda *x: input1_mod[x] - input2_mod[_broadcast(input2_mod.shape, x)], name=name)
    else:
        return hcl.compute(
            input2_mod.shape, lambda *x: input1_mod[_broadcast(input1_mod.shape, x)] - input2_mod[x], name=name)


def broadcast_mul(input1, input2, name='broadcast_mul'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(
            input1_mod.shape, lambda *x: input1_mod[x] * input2_mod[_broadcast(input2_mod.shape, x)], name=name)
    else:
        return hcl.compute(
            input2_mod.shape, lambda *x: input1_mod[_broadcast(input1_mod.shape, x)] * input2_mod[x], name=name)


def broadcast_div(input1, input2, name='broadcast_div'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(
            input1_mod.shape, lambda *x: input1_mod[x] / input2_mod[_broadcast(input2_mod.shape, x)], name=name)
    else:
        return hcl.compute(
            input2_mod.shape, lambda *x: input2_mod[_broadcast(input1_mod.shape, x)] / input2_mod[x], name=name)


def broadcast_mod(input1, input2, name='broadcast_mod'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(
            input1_mod.shape, lambda *x: input1_mod[x] % input2_mod[_broadcast(input2_mod.shape, x)], name=name)
    else:
        return hcl.compute(
            input2_mod.shape, lambda *x: input1_mod[_broadcast(input1_mod.shape, x)] % input2_mod[x], name=name)


def broadcast_pow(input1, input2, name='broadcast_pow'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(
            input1_mod.shape, lambda *x: hcl.power(input1_mod[x], input2_mod[_broadcast(input2_mod.shape, x)]), name=name)
    else:
        return hcl.compute(
            input2_mod.shape, lambda *x: hcl.power(input1_mod[_broadcast(input1_mod.shape, x)], input2_mod[x]), name=name)


def broadcast_equal(input1, input2, name='broadcast_equal'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[x] == input2_mod[_broadcast(input2_mod.shape, x)], 1, 0), name=name)
    else:
        return hcl.compute(input2_mod.shape, lambda *x: hcl.select(
            input1_mod[_broadcast(input1_mod.shape, x)] == input2_mod[x], 1, 0), name=name)


def broadcast_not_equal(input1, input2, name='broadcast_not_equal'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[x] != input2_mod[_broadcast(input2_mod.shape, x)], 1, 0), name=name)
    else:
        return hcl.compute(input2_mod.shape, lambda *x: hcl.select(
            input1_mod[_broadcast(input1_mod.shape, x)] != input2_mod[x], 1, 0), name=name)


def broadcast_greater(input1, input2, name='broadcast_greater'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[x] > input2_mod[_broadcast(input2_mod.shape, x)], 1, 0), name=name)
    else:
        return hcl.compute(input2_mod.shape, lambda *x: hcl.select(
            input1_mod[_broadcast(input2_mod.shape, x)] > input2_mod[x], 1, 0), name=name)


def broadcast_less(input1, input2, name='broadcast_less'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[x] < input2_mod[_broadcast(input2_mod.shape, x)], 1, 0), name=name)
    else:
        return hcl.compute(input2_mod.shape, lambda *x: hcl.select(
            input1_mod[_broadcast(input1_mod.shape, x)] < input2_mod[x], 1, 0), name=name)


def broadcast_greater_equal(input1, input2, name='broadcast_greater_equal'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[x] >= input2_mod[_broadcast(input2_mod.shape, x)], 1, 0), name=name)
    else:
        return hcl.compute(input2_mod.shape, lambda *x: hcl.select(
            input1_mod[_broadcast(input1_mod.shape, x)] >= input2_mod[x], 1, 0), name=name)


def broadcast_less_equal(input1, input2, name='broadcast_less_equal'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[x] <= input2_mod[_broadcast(input2_mod.shape, x)], 1, 0), name=name)
    else:
        return hcl.compute(input2_mod.shape, lambda *x: hcl.select(
            input1_mod[_broadcast(input1_mod.shape, x)] <= input2_mod[x], 1, 0), name=name)


def broadcast_right_shift(input1, input2, name='broadcast_right_shift'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[x] << input2_mod[_broadcast(input2_mod.shape, x)], 1, 0), name=name)
    else:
        return hcl.compute(input2_mod.shape, lambda *x: hcl.select(
            input1_mod[_broadcast(input1_mod.shape, x)] << input2_mod[x], 1, 0), name=name)


def broadcast_left_shift(input1, input2, name='broadcast_left_shift'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[x] >> input2_mod[_broadcast(input2_mod.shape, x)], 1, 0), name=name)
    else:
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[_broadcast(input1_mod.shape, x)] >> input2_mod[x], 1, 0), name=name)


def broadcast_max(input1, input2, name='broadcast_max'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[x] > input2_mod[_broadcast(input2_mod.shape, x)],
            input1_mod[x], input2_mod[_broadcast(input2_mod.shape, x)]), name=name)
    else:
        return hcl.compute(input2_mod.shape, lambda *x: hcl.select(
            input1_mod[_broadcast(input1_mod.shape, x)] > input2_mod[x],
            input1_mod[_broadcast(input1_mod.shape, x)], input2_mod[x]), name=name)


def broadcast_min(input1, input2, name='broadcast_min'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(input1_mod.shape, lambda *x: hcl.select(
            input1_mod[x] < input2_mod[_broadcast(input2_mod.shape, x)],
            input1_mod[x], input2_mod[_broadcast(input2_mod.shape, x)]), name=name)
    else:
        return hcl.compute(input2_mod.shape, lambda *x: hcl.select(
            input1_mod[_broadcast(input1_mod.shape, x)] < input2_mod[x],
            input1_mod[_broadcast(input1_mod.shape, x)], input2_mod[x]), name=name)


def broadcast_and(input1, input2, name='broadcast_and'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(
            input1_mod.shape, lambda *x: input1_mod[x] & input2_mod[_broadcast(input2_mod.shape, x)], name=name)
    else:
        return hcl.compute(
            input2_mod.shape, lambda *x: input1_mod[_broadcast(input1_mod.shape, x)] & input2_mod[x], name=name)


def broadcast_or(input1, input2, name='broadcast_or'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(
            input1_mod.shape, lambda *x: input1_mod[x] | input2_mod[_broadcast(input2_mod.shape, x)], name=name)
    else:
        return hcl.compute(
            input1_mod.shape, lambda *x: input1_mod[_broadcast(input1_mod.shape, x)] | input2_mod[x], name=name)


def broadcast_xor(input1, input2, name='broadcast_xor'):
    input1_mod, input2_mod, switch = broadcast_set(input1, input2)
    if(switch):
        return hcl.compute(
            input1_mod.shape, lambda *x: input1_mod[x] ^ input2_mod[_broadcast(input2_mod.shape, x)], name=name)
    else:
        return hcl.compute(
            input1_mod.shape, lambda *x: input1_mod[_broadcast(input1_mod.shape, x)] ^ input2_mod[x], name=name)

# numpy_like functions


def full(in_shape, fill_val=1, dtype=dtype, name='full'):
    return hcl.compute(
        in_shape, lambda *x: hcl.cast(dtype, fill_val), name=name)


def full_like(array, fill_val, dtype=None, name='full_like'):
    if dtype is None:
        dtype = array.dtype
    return hcl.compute(
        array.shape, lambda *x: hcl.cast(dtype, fill_val), name=name)


def ones(in_shape, dtype=dtype, name='ones'):
    return hcl.compute(in_shape, lambda *x: hcl.cast(dtype, 1), name=name)


def ones_like(array, dtype=None, name='ones_like'):
    if dtype is None:
        dtype = array.dtype
    return hcl.compute(array.shape, lambda *x: hcl.cast(dtype, 1), name=name)


def zeros(in_shape, dtype=dtype, name='zeros'):
    return hcl.compute(in_shape, lambda *x: hcl.cast(dtype, 0), name=name)


def zeros_like(array, dtype=None, name='zeros_like'):
    if dtype is None:
        dtype = array.dtype
    return hcl.compute(array.shape, lambda *x: hcl.cast(dtype, 0), name=name)
