import heterocl as hcl
import heterocl.tvm as tvm
from numbers import Integral

def sort(a, axis=1, init_value=1e3, name="sort"):
    """ sort in axis with ascending order """ 
    assert axis<len(a.shape) and len(a.shape)<=2, "invalid axis" 
    size = a.shape[axis] # 2d insert sorting  
    init = hcl.compute((size,), lambda x: init_value) 

    def sreduce(x, Y):
      with hcl.for_(0,size) as i:
        with hcl.if_(x < Y[i]):
          with hcl.for_(size-1,i,-1) as j:
            Y[j] = Y[j-1]
          Y[i] = x
          hcl.break_()

    my_sort = hcl.reducer(init, sreduce)
    r = hcl.reduce_axis(0, size, name="rdx")
    if axis == 1:
      return hcl.compute(a.shape, 
          lambda x, _y: my_sort(a[x, r], axis=r), name=name)
    else: # sort in y axis
      return hcl.compute(a.shape, 
          lambda _x, y: my_sort(a[r, y], axis=r), name=name)

def argmax(a, axis=1, init_value=-1, name="argmax"):
    """ sort in axis with ascending order """ 
    assert axis<len(a.shape) and len(a.shape)<=2, "invalid axis" 
    size = a.shape[axis] # save max arg index 
    init = hcl.compute((2,), lambda x: init_value) 

    # Y as reducer tensor
    def sreduce(x, Y):
      with hcl.if_(x > Y[1]):
        Y[0] = r
        Y[1] = x

    my_argmax = hcl.reducer(init, sreduce)
    r = hcl.reduce_axis(0, size, name="rdx")
    if axis == 1:
      return hcl.compute((a.shape[0],2), 
          lambda x, _y: my_argmax(a[x, r], axis=r), name=name)
    else: # reduce in y axis
      return hcl.compute((2,a.shape[1]), 
          lambda _x, y: my_argmax(a[r, y], axis=r), name=name)

