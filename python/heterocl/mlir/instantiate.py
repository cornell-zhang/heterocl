import hcl_mlir
from hcl_mlir.dialects import std, builtin
from hcl_mlir import GlobalInsertionPoint, ASTVisitor
from hcl_mlir.ir import *
from .context import *
from .utils import hcl_dtype_to_mlir
from .schedule import Schedule

def instantiate(func, name=None, count=1):
    """Instantiate a function.

    Parameters
    ----------
    func : hcl.Function
        The function to be instantiated.
    name : str, optional
        The name of the instantiated function.
    count : int, optional
        The number of instances of the function.

    Returns
    -------
    hcl.Function
        The instantiated function.
    """
    if name is None:
        if count == 1:
            name = UniqueName.get("instance")
        else:
            names = [UniqueName.get("instance") for _ in range(count)]
    
    if count == 1:
        return Instance(func, name)
    else:
        return [Instance(func, name) for name in names]


class Instance(object):

    def __init__(self, func, name):
        self.func = func
        self.name = name

    def __call__(self, *args):
        call_arg_list = [arg.result for arg in args]
        input_types = [arg.result.type for arg in args]

        # get instance interface return type
        saved_ip_stack = [ip for ip in GlobalInsertionPoint.ip_stack]
        set_context()
        with get_context(), get_location():
            module = Module.create()
            GlobalInsertionPoint.clear()
            GlobalInsertionPoint.save(module.body)
            ret = self.func(*args)
        result_types = [ret.result.type]
        GlobalInsertionPoint.clear()
        GlobalInsertionPoint.ip_stack.extend(saved_ip_stack)
        
        # Build a FuncOp with no function body as declaration
        func_op = builtin.FuncOp(
            name=self.name,
            type=FunctionType.get(inputs=input_types, results=result_types),
            visibility="private",
            ip=GlobalInsertionPoint.get_global()
        )
        call_op = hcl_mlir.CallOp(result_types[0], self.name, call_arg_list)
        call_op.build()

        # Attach necessary information for the callOp
        # attach a 'memref_type' attribute to the call op
        call_op.memref_type = call_op.result.type
        # attach element type to the call op
        setattr(call_op.op, 'dtype', hcl_dtype_to_mlir(ret.dtype))
        # attach a name to the call op
        setattr(call_op, 'name', ret.name)

        # Set up a dataflow node for this instance
        node = Schedule._CurrentSchedule.DataflowGraph.create_node(call_op)
        Schedule._CurrentSchedule.DataflowGraph.add_edges(list(args), [node])
        
        return call_op
