import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
import numpy.testing as tst
import hlib

dtype = hcl.Float(64)
hcl.init(hcl.Float(64))

_sum = hcl.reducer(0,lambda x,y: x + y, dtype)
_max = hcl.reducer(-100000,lambda x,y: tvm.make.Max(x,y), dtype)
_min = hcl.reducer(100000,lambda x,y: tvm.make.Min(x,y), dtype)
_prod= hcl.reducer(1,lambda x,y: x * y, dtype)

def sum(data,axis=None,keepdims=False):
    init_shape = data.shape
    new_shape = []
    new_axis = []
    if axis==None:
        new_shape=[1]
    if isinstance(axis,int):
        if axis<0:
            axis = init_dim+axis
        axis = [axis]
    for i in range(len(init_shape)):
        if axis==None:
            new_axis.append(i)
        elif i in axis:
            new_axis.append(i)
        if not i in new_axis:
            new_shape.append(init_shape[i])
        else:
            if keepdims:
                new_shape.append(1)
    def _new_axes(axis,init_shape):
        new_axes=[]
        for i in range(len(init_shape)):
            if i in axis:
                new_axes.append(hcl.reduce_axis(0,init_shape[i]))
        return new_axes
    def _new_inx(axis,axes,init_shape,*indices):
        indices = indices[0]
        init_dim=len(init_shape)
        new_axis=[]
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
    axes = _new_axes(new_axis,init_shape)
    return hcl.compute(tuple(init_shape),lambda *x: _sum(data[_new_inx(new_axis,axes,init_shape,x)],axis=axes))

def prod(data,axis=None,keepdims=False):
    init_shape = data.shape
    new_shape = []
    new_axis = []
    if axis==None:
        new_shape=[1]
    if isinstance(axis,int):
        if axis<0:
            axis = init_dim+axis
        axis = [axis]
    for i in range(len(init_shape)):
        if axis==None:
            new_axis.append(i)
        elif i in axis:
            new_axis.append(i)
        if not i in new_axis:
            new_shape.append(init_shape[i])
        else:
            if keepdims:
                new_shape.append(1)
    def _new_axes(axis,init_shape):
        new_axes=[]
        for i in range(len(init_shape)):
            if i in axis:
                new_axes.append(hcl.reduce_axis(0,init_shape[i]))
        return new_axes
    def _new_inx(axis,axes,init_shape,*indices):
        indices = indices[0]
        init_dim=len(init_shape)
        new_axis=[]
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
    axes = _new_axes(new_axis,init_shape)
    return hcl.compute(tuple(init_shape),lambda *x: _prod(data[_new_inx(new_axis,axes,init_shape,x)],axis=axes))

def max(data,axis=None,keepdims=False):
    init_shape = data.shape
    new_shape = []
    new_axis = []
    if axis==None:
        new_shape=[1]
    if isinstance(axis,int):
        if axis<0:
            axis = init_dim+axis
        axis = [axis]
    for i in range(len(init_shape)):
        if axis==None:
            new_axis.append(i)
        elif i in axis:
            new_axis.append(i)
        if not i in new_axis:
            new_shape.append(init_shape[i])
        else:
            if keepdims:
                new_shape.append(1)
    def _new_axes(axis,init_shape):
        new_axes=[]
        for i in range(len(init_shape)):
            if i in axis:
                new_axes.append(hcl.reduce_axis(0,init_shape[i]))
        return new_axes
    def _new_inx(axis,axes,init_shape,*indices):
        indices = indices[0]
        init_dim=len(init_shape)
        new_axis=[]
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
    axes = _new_axes(new_axis,init_shape)
    return hcl.compute(tuple(init_shape),lambda *x: _max(data[_new_inx(new_axis,axes,init_shape,x)],axis=axes))

def min(data,axis=None,keepdims=False):
    init_shape = data.shape
    new_shape = []
    new_axis = []
    if axis==None:
        new_shape=[1]
    if isinstance(axis,int):
        if axis<0:
            axis = init_dim+axis
        axis = [axis]
    for i in range(len(init_shape)):
        if axis==None:
            new_axis.append(i)
        elif i in axis:
            new_axis.append(i)
        if not i in new_axis:
            new_shape.append(init_shape[i])
        else:
            if keepdims:
                new_shape.append(1)
    def _new_axes(axis,init_shape):
        new_axes=[]
        for i in range(len(init_shape)):
            if i in axis:
                new_axes.append(hcl.reduce_axis(0,init_shape[i]))
        return new_axes
    def _new_inx(axis,axes,init_shape,*indices):
        indices = indices[0]
        init_dim=len(init_shape)
        new_axis=[]
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
    axes = _new_axes(new_axis,init_shape)
    return hcl.compute(tuple(init_shape),lambda *x: _min(data[_new_inx(new_axis,axes,init_shape,x)],axis=axes))

def exp_test(in_shape):
    data = hcl.placeholder(in_shape)
    def math_func(data):
        return hlib.math.exp(data)
    s = hcl.create_schedule(data,math_func)
    f = hcl.build(s)
    _in = 10*np.random.random(in_shape)-5
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))
    real_out = np.exp(_in)
    f(hcl.asarray(_in),out)
    tst.assert_almost_equal(out.asnumpy(),real_out)

def log_test(in_shape):
    data = hcl.placeholder(in_shape)
    def math_func(data):
        return hlib.math.log(data)
    s = hcl.create_schedule(data,math_func)
    f = hcl.build(s)
    _in = 100*np.random.random(in_shape)+1
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))
    real_out = np.log(_in)
    f(hcl.asarray(_in),out)
    tst.assert_almost_equal(out.asnumpy(),real_out)

def sigmoid_test(in_shape):
    data = hcl.placeholder(in_shape)
    def math_func(data):
        return hlib.math.sigmoid(data)
    s = hcl.create_schedule(data,math_func)
    f = hcl.build(s)
    _in = 10*np.random.random(in_shape)-5
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))
    def sigmoid(data):
        return 1/(1+np.exp(-data))
    real_out = sigmoid(_in)
    f(hcl.asarray(_in),out)
    tst.assert_almost_equal(out.asnumpy(),real_out)

def sqrt_test(in_shape):
    data = hcl.placeholder(in_shape)
    def math_func(data):
        return hlib.math.sqrt(data)
    s = hcl.create_schedule(data,math_func)
    f = hcl.build(s)
    _in = 100*np.random.random(in_shape)+1
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))
    real_out = np.sqrt(_in)
    f(hcl.asarray(_in),out)
    tst.assert_almost_equal(out.asnumpy(),real_out)

def tanh_test(in_shape):
    data = hcl.placeholder(in_shape)
    def math_func(data):
        return hlib.math.tanh(data)
    s = hcl.create_schedule(data,math_func)
    f = hcl.build(s)
    _in = 100*np.random.random(in_shape)-50
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))
    real_out = np.tanh(_in)
    f(hcl.asarray(_in),out)
    tst.assert_almost_equal(out.asnumpy(),real_out)

def sum_test(in_shape,axis=None,keepdims=False):
    data = hcl.placeholder(in_shape)
    def math_func(data,axis=axis,keepdims=keepdims):
        return sum(data,axis,keepdims)
    s = hcl.create_schedule(data,math_func)
    f = hcl.build(s)
    _in = np.random.randint(10,size=in_shape)
    out = hcl.asarray(np.zeros(in_shape))
    f(hcl.asarray(_in),out)
    return _in, out.asnumpy()
 
def max_test(in_shape,axis=None,keepdims=False):
    data = hcl.placeholder(in_shape)
    def math_func(data,axis=axis,keepdims=keepdims):
        return max(data,axis,keepdims)
    s = hcl.create_schedule(data,math_func)
    f = hcl.build(s)
    _in = np.random.randint(10,size=in_shape)
    out = hcl.asarray(np.zeros(in_shape))
    f(hcl.asarray(_in),out)
    return _in, out.asnumpy()

def prod_test(in_shape,axis=None,keepdims=False):
    data = hcl.placeholder(in_shape)
    def math_func(data,axis=axis,keepdims=keepdims):
        return prod(data,axis,keepdims)
    s = hcl.create_schedule(data,math_func)
    f = hcl.build(s)
    _in = np.random.random(size=in_shape)
    out = hcl.asarray(np.zeros(in_shape))
    f(hcl.asarray(_in),out)
    return _in, out.asnumpy()
 
def min_test(in_shape,axis=None,keepdims=False):
    data = hcl.placeholder(in_shape)
    def math_func(data,axis=axis,keepdims=keepdims):
        return min(data,axis,keepdims)
    s = hcl.create_schedule(data,math_func)
    f = hcl.build(s)
    _in = np.random.randint(10,size=in_shape)
    out = hcl.asarray(np.zeros(in_shape))
    f(hcl.asarray(_in),out)
    return _in, out.asnumpy()
 
exp_test((1,3))
exp_test((3,3,3))
exp_test((5,5,3,2))

log_test((1,3))
log_test((3,3,3))
log_test((5,5,3,2))

sigmoid_test((1,3))
sigmoid_test((3,3,3))
sigmoid_test((5,5,3,2))

sqrt_test((1,3))
sqrt_test((3,3,3))
sqrt_test((5,5,3,2))

tanh_test((1,3))
tanh_test((3,3,3))
tanh_test((5,5,3,2))

print(sum_test((3,3)))
print(sum_test((2,2,2),axis=(0,)))

print(max_test((3,3)))
print(max_test((2,2,2),axis=(0,)))

print(prod_test((3,3)))
print(prod_test((2,2,2),axis=(0,)))

print(min_test((3,3)))
print(min_test((2,2,2),axis=(0,)))
