from collections import OrderedDict
import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
from . import nn
dtype = hcl.Int()

# math functions
def exp(input1, name='exp'):
    return hcl.compute(input1.shape, lambda *x: hcl.exp(input1[x]), name=name)


def log(input1, name='log'):
    return hcl.compute(input1.shape, lambda *x: hcl.log(input1[x]), name=name)


def sqrt(input1, name='sqrt'):
    return hcl.compute(input1.shape, lambda *x: hcl.sqrt(input1[x]), name=name)


def sigmoid(input1, name='sigmoid'):
    return hcl.compute(input1.shape,
                       lambda *x: hcl.exp(input1[x]) /
                       (hcl.exp(input1[x]) + 1),
                       name=name)


def tanh(x, name="tanh"):
    return hcl.compute(x.shape,
                       lambda *args: hcl.tanh(x[args]),
                       name,
                       attrs=OrderedDict([('app_name',
                                           tvm.make.StringImm('tanh'))]))


def clip(x, a_min=0.0, a_max=1.0, name="clip"):
    lower = hcl.compute(x.shape,
                        lambda *args: hcl.select(x[args] <= a_max,
                                                 x[args],
                                                 a_max),
                        name=name + "_low")
    return hcl.compute(x.shape, lambda *args: hcl.select(
        lower[args] >= a_min, lower[args], a_min), name=name + "_high")


def sum(data, axis=None, keepdims=True):
    init_shape = data.shape
    init_dim = len(init_shape)
    new_shape = []
    new_axis = []
    if isinstance(axis, int):
        if axis < 0:
            axis = init_dim + axis
        axis = [axis]
    for i in range(len(init_shape)):
        if axis is None:
            new_axis.append(i)
        elif i in axis:
            new_axis.append(i)
        if i not in new_axis:
            new_shape.append(init_shape[i])
        else:
            if keepdims:
                new_shape.append(1)

    def _new_axes(axis, init_shape):
        new_axes = []
        for i in range(len(init_shape)):
            if i in axis:
                new_axes.append(hcl.reduce_axis(0, init_shape[i]))
        return new_axes

    def _new_inx(axis, axes, init_shape, *indices):
        indices = indices[0]
        init_dim = len(init_shape)
        new_axis = []
        inx = 0
        axis_inx = 0
        for i in range(init_dim):
            if i in axis:
                new_axis.append(axes[axis_inx])
                axis_inx = axis_inx + 1
            else:
                new_axis.append(indices[inx])
                inx = inx + 1
        return tuple(new_axis)
    axes = _new_axes(new_axis, init_shape)
    axis_len = len(axis)
    temp_transpose = []
    _new_shape = []

    for i in range(len(init_shape)):
        if i not in axis:
            _new_shape.append(init_shape[i])
            temp_transpose.append(i)
    for i in range(len(axis)):
        temp_transpose.append(axis[i])
    while len(_new_shape) < len(init_shape):
        _new_shape.append(1)
    transpose_axes = []
    for i in range(len(temp_transpose)):
        transpose_axes.append(temp_transpose.index(i))
    _sum = hcl.reducer(0, lambda x, y: x + y, data.dtype)
    out = hcl.compute(tuple(_new_shape),
                      lambda *x: _sum(data[_new_inx(new_axis,
                                                    axes,
                                                    init_shape,
                                                    x)],
                                      axis=axes))
    if keepdims:
        return nn.transpose(out, transpose_axes)
    else:
        return nn.squeeze(out)


def prod(data, axis=None, keepdims=False):
    init_shape = data.shape
    init_dim = len(init_shape)
    new_shape = []
    new_axis = []
    if isinstance(axis, int):
        if axis < 0:
            axis = init_dim + axis
        axis = [axis]
    for i in range(len(init_shape)):
        if axis is None:
            new_axis.append(i)
        elif i in axis:
            new_axis.append(i)
        if i not in new_axis:
            new_shape.append(init_shape[i])
        else:
            if keepdims:
                new_shape.append(1)

    def _new_axes(axis, init_shape):
        new_axes = []
        for i in range(len(init_shape)):
            if i in axis:
                new_axes.append(hcl.reduce_axis(0, init_shape[i]))
        return new_axes

    def _new_inx(axis, axes, init_shape, *indices):
        indices = indices[0]
        init_dim = len(init_shape)
        new_axis = []
        inx = 0
        axis_inx = 0
        for i in range(init_dim):
            if i in axis:
                new_axis.append(axes[axis_inx])
                axis_inx = axis_inx + 1
            else:
                new_axis.append(indices[inx])
                inx = inx + 1
        return tuple(new_axis)
    axes = _new_axes(new_axis, init_shape)
    axis_len = len(axis)
    temp_transpose = []
    _new_shape = []

    for i in range(len(init_shape)):
        if i not in axis:
            _new_shape.append(init_shape[i])
            temp_transpose.append(i)
    for i in range(len(axis)):
        temp_transpose.append(axis[i])
    while len(_new_shape) < len(init_shape):
        _new_shape.append(1)
    transpose_axes = []
    for i in range(len(temp_transpose)):
        transpose_axes.append(temp_transpose.index(i))
    _prod = hcl.reducer(1, lambda x, y: x * y, data.dtype)
    out = hcl.compute(tuple(_new_shape),
                      lambda *x: _prod(data[_new_inx(new_axis,
                                                     axes,
                                                     init_shape,
                                                     x)],
                                       axis=axes))
    if keepdims:
        return nn.transpose(out, transpose_axes)
    else:
        out = nn.squeeze(out)
        return out


def max(data, axis=None, keepdims=False):
    init_shape = data.shape
    init_dim = len(init_shape)
    new_shape = []
    new_axis = []
    if isinstance(axis, int):
        if axis < 0:
            axis = init_dim + axis
        axis = [axis]
    for i in range(len(init_shape)):
        if axis is None:
            new_axis.append(i)
        elif i in axis:
            new_axis.append(i)
        if i not in new_axis:
            new_shape.append(init_shape[i])
        else:
            if keepdims:
                new_shape.append(1)

    def _new_axes(axis, init_shape):
        new_axes = []
        for i in range(len(init_shape)):
            if i in axis:
                new_axes.append(hcl.reduce_axis(0, init_shape[i]))
        return new_axes

    def _new_inx(axis, axes, init_shape, *indices):
        indices = indices[0]
        init_dim = len(init_shape)
        new_axis = []
        inx = 0
        axis_inx = 0
        for i in range(init_dim):
            if i in axis:
                new_axis.append(axes[axis_inx])
                axis_inx = axis_inx + 1
            else:
                new_axis.append(indices[inx])
                inx = inx + 1
        return tuple(new_axis)
    axes = _new_axes(new_axis, init_shape)
    axis_len = len(axis)
    temp_transpose = []
    _new_shape = []

    for i in range(len(init_shape)):
        if i not in axis:
            _new_shape.append(init_shape[i])
            temp_transpose.append(i)
    for i in range(len(axis)):
        temp_transpose.append(axis[i])
    while len(_new_shape) < len(init_shape):
        _new_shape.append(1)
    transpose_axes = []
    for i in range(len(temp_transpose)):
        transpose_axes.append(temp_transpose.index(i))
    _max = hcl.reducer(-10000, lambda x, y: tvm.make.Max(x, y), data.dtype)
    out = hcl.compute(tuple(_new_shape),
                      lambda *x: _max(data[_new_inx(new_axis,
                                                    axes,
                                                    init_shape,
                                                    x)],
                                      axis=axes))
    if keepdims:
        return nn.transpose(out, transpose_axes)
    else:
        out = nn.squeeze(out)
        return out


def min(data, axis=None, keepdims=False):
    init_shape = data.shape
    init_dim = len(init_shape)
    new_shape = []
    new_axis = []
    if isinstance(axis, int):
        if axis < 0:
            axis = init_dim + axis
        axis = [axis]
    for i in range(len(init_shape)):
        if axis is None:
            new_axis.append(i)
        elif i in axis:
            new_axis.append(i)
        if i not in new_axis:
            new_shape.append(init_shape[i])
        else:
            if keepdims:
                new_shape.append(1)

    def _new_axes(axis, init_shape):
        new_axes = []
        for i in range(len(init_shape)):
            if i in axis:
                new_axes.append(hcl.reduce_axis(0, init_shape[i]))
        return new_axes

    def _new_inx(axis, axes, init_shape, *indices):
        indices = indices[0]
        init_dim = len(init_shape)
        new_axis = []
        inx = 0
        axis_inx = 0
        for i in range(init_dim):
            if i in axis:
                new_axis.append(axes[axis_inx])
                axis_inx = axis_inx + 1
            else:
                new_axis.append(indices[inx])
                inx = inx + 1
        return tuple(new_axis)
    axes = _new_axes(new_axis, init_shape)
    axis_len = len(axis)
    temp_transpose = []
    _new_shape = []

    for i in range(len(init_shape)):
        if i not in axis:
            _new_shape.append(init_shape[i])
            temp_transpose.append(i)
    for i in range(len(axis)):
        temp_transpose.append(axis[i])
    while len(_new_shape) < len(init_shape):
        _new_shape.append(1)
    transpose_axes = []
    for i in range(len(temp_transpose)):
        transpose_axes.append(temp_transpose.index(i))
    _min = hcl.reducer(10000, lambda x, y: tvm.make.Min(x, y), data.dtype)
    out = hcl.compute(tuple(_new_shape),
                      lambda *x: _min(data[_new_inx(new_axis,
                                                    axes,
                                                    init_shape,
                                                    x)],
                                      axis=axes))
    if keepdims:
        return nn.transpose(out, transpose_axes)
    else:
        out = nn.squeeze(out)
        return out

# numpy_like functions

# get rid of this


def full(shape=(1,), fill_val=1, dtype=dtype, name='full'):
    if isinstance(shape, list):
        shape = tuple(shape)
    return hcl.compute(
        shape, lambda *x: hcl.cast(dtype, fill_val), name=name, dtype=dtype)


def full_like(array, fill_val, dtype=None, name='full_like'):
    if dtype is None:
        dtype = array.dtype
    hcl.init(dtype)
    return hcl.compute(
        array.shape, lambda *x: hcl.cast(dtype, fill_val), name=name)


def ones(shape=(1,), dtype=dtype, name='ones'):
    dtype = hcl.dtype_to_hcl(dtype)
    return hcl.compute(
        tuple(shape), lambda *x: hcl.cast(dtype, 1), name=name, dtype=dtype)


def ones_like(array, dtype=None, name='ones_like'):
    if dtype is None:
        dtype = array.dtype
    hcl.init(dtype)
    return hcl.compute(array.shape, lambda *x: hcl.cast(dtype, 1), name=name)


def zeros(shape=(1,), dtype=dtype, name='zeros'):
    dtype = hcl.dtype_to_hcl(dtype)
    shape = list(shape)
    for i in range(len(shape)):
        if hasattr(shape[i], 'value'):
            shape[i] = shape[i].value
    return hcl.compute(
        tuple(shape), lambda *x: hcl.cast(dtype, 0), name=name, dtype=dtype)


def zeros_like(array, dtype=None, name='zeros_like'):
    if dtype is None:
        dtype = array.dtype
    return hcl.compute(array.shape, lambda *x: hcl.cast(dtype, 0), name=name)
