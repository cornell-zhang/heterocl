from .tvm import expr as _expr, stmt as _stmt
from .tvm import make as _make
from .tensor import Var, Tensor, TensorSlice
from .schedule import Stage

class Module(object):

    # arg: 0 is var, 1 is placeholder

    def __init__(self, args, name, ret_void, lhs, dtype=None):
        self.args = args
        self.name = name
        self.ret_void = ret_void
        self.dtype = dtype
        self.lhs = lhs

    def __call__(self, *args):
        #stage = Stage.get_current()
        print self.ret_void
        new_args = []
        for (arg_type, arg) in zip(self.args, args):
            if arg_type == ():
                if isinstance(arg, Var):
                    new_args.append(arg.var)
                elif isinstance(arg, (_expr.Expr, TensorSlice)):
                    new_args.append(arg)
            else:
                new_args.append(arg.buf.data)
        assert(Stage.get_len() > 0)
        stage = Stage.get_current()
        for arg in args:
            if isinstance(arg, Tensor):
                stage.input_stages.add(arg.last_update)
        for l in self.lhs:
            stage.lhs_tensors.add(args[l])
        if self.ret_void:
            stage.emit(_make.KernelStmt(new_args, self.name))
        else:
            return _make.KernelExpr(self.dtype, new_args, self.name)

"""
class KernelTensor(Kernel, Tensor):

    def __init__(self, args, name, ret_void, body, dtype = None):
        Kernel.__init__(self, args, name, ret_void, body, dtype)
        Tensor.__init__(self, (1,), "int32", name)

    def __call__(self, *args):
        CodeBuilder.current[-1].tensors.add(self)
        return Kernel.__call__(self, *args)
"""
