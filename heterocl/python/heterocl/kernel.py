from tvm import expr as _expr, stmt as _stmt
from tvm import make as _make
from .tensor import Var, Tensor, TensorSlice
from .code_builder import CodeBuilder

class Kernel():

  # arg: 0 is var, 1 is placeholder

  def __init__(self, args, name, ret_void, body, dtype = None):
    self.args = args
    self.name = name
    self.ret_void = ret_void
    self.dtype = dtype
    self.body = body

  def __call__(self, *args):
    builders = CodeBuilder.current
    assert len(builders) != 0, "Kernel call must be used inside a CodeBuilder"
    new_args = []
    for (arg_type, arg) in zip(self.args, args):
      if arg_type == 0:
        if isinstance(arg, Var):
          new_args.append(arg.var)
        elif isinstance(arg, (_expr.Expr, TensorSlice)):
          new_args.append(arg)
      else:
        new_args.append(arg.buf.data)
    if self.ret_void:
      builders[-1].emit(_make.KernelStmt(new_args, self.name))
    else:
      return _make.KernelExpr(self.dtype, new_args, self.name)

class KernelTensor(Kernel, Tensor):

  def __init__(self, args, name, ret_void, body, dtype = None):
    Kernel.__init__(self, args, name, ret_void, body, dtype)
    Tensor.__init__(self, (1,), "int32", name)
