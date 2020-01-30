import heterocl as hcl
import heterocl.tvm as tvm
from numbers import Integral
from . import nn
from collections import OrderedDict
import types

def change_func_args(function, new_args):
    """ Create a new function with its arguments renamed to new_args. """
    code_obj = function.__code__
    assert(0 <= len(new_args) <= code_obj.co_argcount)
    new_varnames = tuple(list(new_args[:code_obj.co_argcount]) +
                         list(code_obj.co_varnames[code_obj.co_argcount:]))
    new_code_obj = types.CodeType(code_obj.co_argcount,
                                  code_obj.co_kwonlyargcount,
                                  code_obj.co_nlocals,
                                  code_obj.co_stacksize,
                                  code_obj.co_flags,
                                  code_obj.co_code,
                                  code_obj.co_consts,
                                  code_obj.co_names,
                                  new_varnames,
                                  code_obj.co_filename,
                                  code_obj.co_name,
                                  code_obj.co_firstlineno,
                                  code_obj.co_lnotab,
                                  code_obj.co_freevars,
                                  code_obj.co_cellvars)
    modified = types.FunctionType(new_code_obj, function.__globals__, 
                                  name=function.__name__,
                                  argdefs=function.__defaults__,
                                  closure=function.__closure__)
    function.__code__ = modified.__code__  # replace code portion of original

# out = hlib.function.conv2d_nchw(_input, _kernel)
def conv2d_nchw(_input, _kernel, _output, stride=[1,1], name="conv",
                padding=[[0,0],[0,0]], activation="relu"):
    
    out_dtype = _input.dtype
    batch, in_channel, in_height, in_width = _input.shape
    num_filter, channel, kernel_h, kernel_w = _kernel.shape
    stride_h, stride_w = stride
    [pad_top, pad_left], [pad_down, pad_right] = padding
    # compute the output shape
    out_channel = num_filter
    out_height = nn.simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width  = nn.simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    # activation function 
    actv = lambda x: x
    if activation == "relu":
      actv = lambda x: hcl.select(x > 0, x, 0)
    if activation == "tanh":
      actv = lambda x: tvm.tanh(x)
    sum = hcl.reducer(0, lambda x, y: x + y, _input.dtype)

    # wrap hcl compute apis
    def conv2d(_input, _kernel):
        # if padding != [[0,0],[0,0]]:
        #     _input = nn.pad(_input, pad_before, pad_after, name=name+"_pad")
        rc = hcl.reduce_axis(0, in_channel)
        ry = hcl.reduce_axis(0, kernel_h)
        rx = hcl.reduce_axis(0, kernel_w)

        return hcl.update(_output,
            lambda nn, ff, yy, xx: actv(sum(
                _input[nn, rc, yy * stride_h + ry, xx * stride_w + rx] *
                _kernel[ff, rc, ry, rx],
                axis=[rc, ry, rx])),
            name=name)

    # return decorated function  
    mod = hcl.def_([_input.shape, _kernel.shape, _output.shape], name=name)(conv2d)
    return mod

# hlib.function.sort(a, b)
def sort(a, b, axis=1, init_value=1e3, name="sort"):
    """ sort in axis with ascending order """ 
    assert axis<len(a.shape) and len(a.shape)<=2, "invalid axis" 
    assert a.shape == b.shape, "shape mismatch" 
    size = a.shape[axis] # 2d insert sorting  

    def sreduce(x, Y):
        with hcl.for_(0, size, name="i") as i:
          with hcl.if_(x < Y[i]):
            with hcl.for_(size-1,i,-1) as j:
              Y[j] = Y[j-1]
            Y[i] = x
            hcl.break_()

    def sort2d(A, B):
        init = hcl.compute((size,), lambda x: init_value) 
        my_sort = hcl.reducer(init, sreduce)
        r = hcl.reduce_axis(0, size, name="rdx")
        if axis == 1:
          hcl.update(B, lambda x, _y: 
              my_sort(A[x, r], axis=r), name=name)
        else: # sort in y axis
          hcl.update(B, lambda _x, y: 
              my_sort(A[r, y], axis=r), name=name)

    # return decorated function  
    # change_func_args(sort2d, [a.name, b.name])
    mod = hcl.def_([a.shape, b.shape], name=name, 
                   arg_names=[a.name, b.name])(sort2d)
    mod(a, b)

def argmax(a, b, axis=1, init_value=-1, name="argmax"):
    """ sort in axis with ascending order """ 
    assert axis<len(a.shape) and len(a.shape)<=2, "invalid axis" 
    assert b.shape[axis] == 2, "shape mismatch" 
    size = a.shape[axis] # save max arg index 

    def argmax2d(A, B):
        init = hcl.compute((2,), lambda x: init_value) 
        r = hcl.reduce_axis(0, size, name="rdx")
        # Y as reducer tensor
        def sreduce(x, Y):
            with hcl.if_(x > Y[1]):
                Y[0] = r
                Y[1] = x

        my_argmax = hcl.reducer(init, sreduce)
        if axis == 1:
          return hcl.update(B, 
              lambda x, _y: my_argmax(A[x, r], axis=r), name=name)
        else: # reduce in y axis
          return hcl.update(B, 
              lambda _x, y: my_argmax(A[r, y], axis=r), name=name)

    # return decorated function  
    mod = hcl.def_([a.shape, b.shape], name=name)(argmax2d)
    mod(a, b)
