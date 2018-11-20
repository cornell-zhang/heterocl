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
        self.call_num = 0

    def __call__(self, *args):
        #stage = Stage.get_current()
        new_args = []
        for (arg_type, arg) in zip(self.args, args):
            if arg_type == ():
                if isinstance(arg, Var):
                    new_args.append(arg.var)
                elif isinstance(arg, (_expr.Expr, TensorSlice)):
                    new_args.append(arg)
            else:
                new_args.append(arg.buf.data)
        input_stages = set([])
        lhs_tensors = set([])
        for arg in args:
            if isinstance(arg, Tensor):
                input_stages.add(arg.last_update)
        for l in self.lhs:
            lhs_tensors.add(args[l])
        if self.ret_void:
            with Stage(self.name+str(self.call_num)) as stage:
                stage.input_stages.update(input_stages)
                stage.lhs_tensors.update(lhs_tensors)
                stage.emit(_make.KernelStmt(new_args, self.name))
            self.call_num += 1
        else:
            assert(Stage.get_len() > 0)
            stage = Stage.get_current()
            stage.input_stages.update(input_stages)
            stage.lhs_tensors.update(lhs_tensors)
            return _make.KernelExpr(self.dtype, new_args, self.name)
