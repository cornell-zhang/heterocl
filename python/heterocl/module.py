from .tvm import expr as _expr, stmt as _stmt
from .tvm import make as _make
from .api import placeholder
from .tensor import Scalar, Tensor, TensorSlice
from .schedule import Stage, Schedule

class Module(object):

    def __init__(self, shapes, arg_names, name, ret_void, lhs, dtype=None):
        self.shapes = shapes
        self.arg_names = arg_names
        self.name = name
        self.ret_void = ret_void
        self.dtype = dtype
        self.lhs = lhs
        self.call_num = 0

    def __call__(self, *args):
        #stage = Stage.get_current()
        new_args = []
        for (shape, arg) in zip(self.shapes, args):
            if shape == ():
                if isinstance(arg, Scalar):
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

        # register function calls
        if self.name not in Schedule.mod_calls:
            Schedule.mod_calls[self.name] = list()
        Schedule.mod_calls[self.name].append(args)
        
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
