import inspect
from collections import OrderedDict

import hcl_mlir
import numpy as np
import re
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import *
from hcl_mlir.exceptions import *

from . import config
from .types import Int, Type, UInt, Struct, dtype_to_hcl
from .context import NestedStageLevel, UniqueName
from .dsl import for_
from .schedule import Schedule, Stage, scope
from .tensor import Array, Tensor
from .utils import get_dtype_str, hcl_dtype_to_mlir, get_func_obj, get_src_loc
from .context import get_context, get_location
from .ir import intermediate as itmd


def init(init_dtype=Int(32), raise_assert_exception=True):
    """Initialize a HeteroCL environment with configurations."""
    config.init_dtype = init_dtype
    config.raise_assert_exception = raise_assert_exception


def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs."""
    if name is None:
        name = UniqueName.get("tensor")
    if (
        not dtype == None
        and not isinstance(dtype, (Type, str))
        and not hcl_mlir.is_hcl_mlir_type(dtype)
        and not isinstance(name, str)
    ):
        raise APIError("Input type error, got dtype={}, name={}".format(dtype, name))
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    if shape == ():
        shape = (1,)
    dtype = config.init_dtype if dtype == None else dtype
    filename, lineno = get_src_loc(frame=1)
    alloc = itmd.AllocOp(name, shape, dtype, itmd.Location(filename, lineno))
    return alloc


def asarray(np_array, dtype=None):
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype == None else dtype
    return Array(np_array, dtype)


def scalar(init, name=None, dtype=None):
    if name is None:
        name = UniqueName.get("scalar")
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype == None else dtype # dtype is HeteroCL type
    if isinstance(dtype, Struct):
        raise HCLNotImplementedError("Struct scalar is not supported yet")

    # Generate a ComputeOp
    filename, lineno = get_src_loc()
    op = itmd.ComputeOp(name, (1,), lambda _ : init, dtype, itmd.Location(filename, lineno))
    region = scope.get()
    region.append(op)
    return op.tensor

def scalar_old(init, name=None, dtype=None):
    """Syntactic sugar: single-value tensor
    - init: int, float, expr, or tuple
    """
    hcl_mlir.enable_build_inplace()
    if name is None:
        name = UniqueName.get("scalar")

    ret_tensor = placeholder((1,), name=name, dtype=dtype)
    index = hcl_mlir.ConstantOp("index", 0)
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    dtype = config.init_dtype if dtype == None else dtype # dtype is HeteroCL type
    mlir_type = hcl_dtype_to_mlir(dtype)
    if isinstance(dtype, Struct):
        if isinstance(init, tuple):
            init = hcl_mlir.StructConstructOp(list(init))
        elif isinstance(init, int):
            vals = list()
            for ftype in dtype.dtype_dict.values():
                mask = (1 << (ftype.bits+1)) - 1
                val = init & mask
                init = init >> ftype.bits
                vals.append(hcl_mlir.ConstantOp(hcl_dtype_to_mlir(ftype), val))
            init = hcl_mlir.StructConstructOp(vals)
        # TODO(Niansong): support init as a single expr
    elif isinstance(init, int) or isinstance(init, float):
        init = hcl_mlir.ConstantOp(mlir_type, init)
    elif isinstance(init, Tensor):
        init = init.op
    ret_tensor.init()  # init hcl_mlir type
    hcl_mlir.StoreOp(init, ret_tensor.op, [index])
    return ret_tensor


def reduce_axis(lower, upper, name=None):
    """Create a reduction axis for reduction operations."""
    if name is None:
        name = UniqueName.get("reduction_axis")
    # return hcl_mlir.ReduceVar(None, bound=(lower, upper), name=name)
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.ReduceVar(name, parent_loop=None, loc=loc, bound=(lower, upper))


def cast(dtype, expr):
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    if isinstance(expr, Tensor):
        raise APIError("Tensor is not supported in hcl.cast. " +
                        "If you are try to cast a hcl.scalar, please use hcl.cast(scalar.v)")
    return hcl_mlir.CastOp(expr, hcl_dtype_to_mlir(dtype))


def const_tensor(values, name=None, dtype=None):
    """Create a constant tensor"""
    if name is None:
        name = UniqueName.get("tensor")
    dtype = config.init_dtype if dtype == None else dtype
    cst = hcl_mlir.ConstantOp(hcl_dtype_to_mlir(dtype), values, name)
    return cst.tensor


def copy(values, name=None, dtype=None):
    """A syntactic sugar for copying an existing tensor."""
    if name is None:
        name = UniqueName.get("tensor")
    dtype = config.init_dtype if dtype == None else dtype
    cst = hcl_mlir.ConstantOp(hcl_dtype_to_mlir(dtype), values, name)
    return cst.tensor


def select(cond, true_val, false_val):
    # return hcl_mlir.SelectOp(cond, true_val, false_val)
    raise HCLNotImplementedError("select is not implemented yet")


def sum(expr, axis=None, dtype=None, name=None):
    if name is None:
        name = UniqueName.get("op")
    if axis is None:
        raise HCLNotImplementedError("sum with axis=None is not supported")
    if isinstance(axis, tuple):
        axis = list(axis)
    elif isinstance(axis, itmd.ReduceVar):
        axis = [axis]
    elif not isinstance(axis, list):
        raise APIError("axis must be a list of reduction axis")
    dtype = config.init_dtype if dtype == None else dtype
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.SumOp(name, expr, axis, dtype, loc)


def max(data, axis=None, dtype=None, name=""):
    raise HCLNotImplementedError("max is not implemented yet")
    dtype = config.init_dtype if dtype == None else dtype
    return hcl_mlir.MaxOp(data, axis, get_dtype_str(dtype))


def min(data, axis=None, dtype=None, name=""):
    raise HCLNotImplementedError("min is not implemented yet")
    dtype = config.init_dtype if dtype == None else dtype
    return hcl_mlir.MinOp(data, axis, get_dtype_str(dtype))


def reduce(data, init_val, reduce_op, axis=None, dtype=None, name=""):
    raise HCLNotImplementedError("reduce is not implemented yet")
    return hcl_mlir.ReduceOp(data, axis, get_dtype_str(dtype), prefix=name, init_val=init_val, reduce_op={"si": reduce_op})


def pack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Pack a tensor with smaller bitwidth to a tensor with larger bitwidth."""
    if factor is None and dtype is not None:
        factor = dtype.bits // tensor.dtype.bits
    if factor is None or not isinstance(factor, int):
        raise APIError("Should specify factor")
    if not isinstance(tensor.dtype, (Int, UInt)):
        raise APIError("Only support integer packing")
    if name == None or name == "":
        name = UniqueName.get("tensor")
    bitwidth = tensor.dtype.bits
    if isinstance(tensor.dtype, Int):
        new_type = Int(bitwidth * factor)
    else:
        new_type = UInt(bitwidth * factor)
    new_shape = [
        size // factor if i == axis else size for i, size in enumerate(tensor.shape)
    ]

    def assign_val(*indices):
        result = scalar(0, name="packed_" + name, dtype=new_type)
        with for_(0, factor) as i:
            new_indices = [
                (index * factor + i) if j == axis else index
                for j, index in enumerate(indices)
            ]
            val = tensor[tuple(new_indices)]
            result[0][bitwidth * i: bitwidth *
                      (i + 1)] = tensor[tuple(new_indices)]
        return result[0]

    return compute(tuple(new_shape), assign_val, name, new_type)


def unpack(tensor, axis=0, factor=None, name=None, dtype=None):
    """Unpack a tensor with larger bitwidth to a tensor with smaller bitwidth."""
    if factor is None and dtype is not None:
        factor = tensor.dtype.bits // dtype.bits
    if factor is None or not isinstance(factor, int):
        raise APIError("Should specify factor")
    if not isinstance(tensor.dtype, (Int, UInt)):
        raise APIError("Only support integer packing")
    if name == None or name == "":
        name = UniqueName.get("tensor")
    bitwidth = tensor.dtype.bits
    if isinstance(tensor.dtype, Int):
        new_type = Int(bitwidth // factor)
    else:
        new_type = UInt(bitwidth // factor)
    new_shape = [
        size * factor if i == axis else size for i, size in enumerate(tensor.shape)
    ]

    def assign_val(*indices):
        result = scalar(0, name="unpacked_" + name, dtype=new_type)
        new_indices = [
            (index // factor) if j == axis else index for j, index in enumerate(indices)
        ]
        lower = (indices[axis] % factor) * (bitwidth // factor)
        upper = lower + bitwidth // factor
        val = tensor[tuple(new_indices)][lower:upper]
        if val.dtype.width != bitwidth // factor:
            # cast val to the same width as bitwidth // factor
            val = hcl_mlir.CastOp(val, hcl_dtype_to_mlir(new_type))
        result[0][0: bitwidth // factor] = val
        return result[0]

    return compute(tuple(new_shape), assign_val, name, new_type)
    

def compute_body(name, shape, fcompute, dtype, loc, tensor):
    """ Create an itmd.ComputeOp and its body operations

    Parameters
    ----------
    name : str
        The name of the compute op
    shape: tuple
        The shape of the compute op
    fcompute: function
        The compute function
    dtype: hcl.dtype
        The data type of the compute op
    tensor: itmd.AllocOp, None, or "no_alloc"
        The tensor to store the result of the compute op
        - itmd.AllocOp: hcl.update
        - None: hcl.compute, ComputeOp will allocate new tensor
        - "no_alloc": hcl.mutate, no tensor will be allocated

    Returns
    -------
    itmd.ComputeOp
        The compute op
    """
    # Generate a ComputeOp
    compute_op = itmd.ComputeOp(name, shape, fcompute, dtype, loc, tensor)
    region = scope.get()
    region.append(compute_op)
    # Analyze input tensors, and update uses for those tensors
    closure_var = inspect.getclosurevars(fcompute).nonlocals
    input_tensors = [v for v in closure_var.values() if isinstance(v, itmd.AllocOp)]
    reduce_vars = [v for v in closure_var.values() if isinstance(v, itmd.ReduceVar)]
    for t in input_tensors:
        t.uses.append(compute_op)

    # Build AST for fcompute body
    argspec = inspect.getfullargspec(fcompute)
    axis_names = argspec.args
    iter_vars = [itmd.IterVar(name, None, loc) for name in axis_names]
    # attach iter_vars to the compute op
    # iter_var's parent_loop will be set in ir.ir_builder.build_compute
    compute_op.iter_vars.extend(iter_vars)
    compute_op.reduce_vars.extend(reduce_vars)
    scope.push(compute_op.body)
    if tensor is None:
        # hcl.compute
        res_expr = fcompute(*iter_vars)
        res_expr = itmd.immediate_to_constant(res_expr, loc)
        store_op = itmd.StoreOp(compute_op.tensor, compute_op.iter_vars, res_expr, loc)
        compute_op.body.append(store_op)
        scope.pop()
    elif isinstance(tensor, itmd.AllocOp):
        # hcl.update
        res_expr = fcompute(*iter_vars)
        res_expr = itmd.immediate_to_constant(res_expr, loc)
        store_op = itmd.StoreOp(tensor, iter_vars, res_expr, loc)
        compute_op.body.append(store_op)
        scope.pop()
    elif isinstance(tensor, str) and tensor == "no_alloc":
        # hcl.mutate
        res_expr = fcompute(*iter_vars)
        if res_expr is not None:
            raise APIError("hcl.mutate does not support return value")
        scope.pop()
    else:
        raise APIError("Invalid tensor type")

    return compute_op

def compute(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
    if not isinstance(shape, tuple):
        raise APIError("The shape of compute API must be a tuple")
    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    if name is None:
        name = UniqueName.get("tensor")
    if not dtype == None and not isinstance(dtype, (Type, str)):
        raise APIError("Type error")
    dtype = config.init_dtype if dtype == None else dtype
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)

    # Generate a ComputeOp
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    compute_op = compute_body(name, shape, fcompute, dtype, loc, None)
    return compute_op.tensor

def old_compute(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
    """
    This function call does not directly build IR, it only creates a node
    """
    # check API correctness
    if not isinstance(shape, tuple):
        raise APIError("The shape of compute API must be a tuple")
    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    if name is None:
        name = UniqueName.get("tensor")
    if not dtype == None and not isinstance(dtype, (Type, str)):
        raise APIError("Type error")
    dtype = config.init_dtype if dtype == None else dtype
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)

    ret_tensor = Tensor(shape, dtype, name=name,
                        fcompute=fcompute, impl="compute")
    for tensor in ret_tensor.op.inputs:
        tensor.add_use(ret_tensor)


    # Check the caller function
    caller_func_name = inspect.stack()[1].function
    if Schedule._TopFunction is None:
        called_from_top = False
    else:
        called_from_top = caller_func_name == Schedule._TopFunction.__name__
    caller_func = Schedule._TopFunction if called_from_top else get_func_obj(caller_func_name)
    if not called_from_top:
        # Caller function up one level
        caller_parent_func_name = inspect.stack()[2].function
        caller_parent_func = get_func_obj(caller_parent_func_name)
        # If haven't already, attach the caller function
        # to its parent function as an attribute
        if not hasattr(caller_parent_func, caller_func_name):
            caller_parent_func.__setattr__(caller_func_name, caller_func)
    stage = ret_tensor.op.stage
    if Schedule._TopFunction != None:
        if NestedStageLevel.get() > 0:
            # Attach the stage to the parent stage
            # TODO(Niansong): test this
            Schedule._CurrentStage[-1].__setattr__(name, stage)
            Schedule._CurrentStage[-1]._sub_stages.append(stage)
        else:
            caller_func.__setattr__(stage.name, stage.op)
            # Set up a list of stages for the caller function
            if not hasattr(caller_func, "_stages"):
                caller_func.__setattr__("_stages", [stage])
            else:
                caller_func._stages.append(stage)    
    
    return ret_tensor

def update(tensor, fcompute, name=None):
    if not isinstance(tensor, itmd.AllocOp):
        raise APIError("The input of update API must be an allocated tensor")
    if name is None:
        name = tensor.name + "_updated"
    
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    compute_body(name, tensor.shape, fcompute, tensor.dtype, loc, tensor)
    return tensor

def old_update(tensor: Tensor, fcompute, name=None):
    """
    fcompute: function, callable
    name: str
    """
    # Check the caller function
    caller_func_name = inspect.stack()[1].function
    if Schedule._TopFunction is None:
        called_from_top = False
    else:
        called_from_top = caller_func_name == Schedule._TopFunction.__name__
    caller_func = get_func_obj(caller_func_name)
    if not called_from_top:
        # Caller function up one level
        caller_parent_func_name = inspect.stack()[2].function
        caller_parent_func = get_func_obj(caller_parent_func_name)
        # If haven't already, attach the caller function
        # to its parent function as an attribute
        if not hasattr(caller_parent_func, caller_func_name):
            caller_parent_func.__setattr__(caller_func_name, caller_func)

    # Check tensor type
    if not isinstance(tensor, Tensor):
        raise APIError(
            "Unexpected argument type of the "
            + "first argument: {}, update API expects tensor as input.".format(
                type(tensor)
            )
        )
    if name is None:
        name = tensor.name + "_updated"
    # Create a new Tensor, along with its stage
    new_tensor = Tensor(
        tensor.shape,
        tensor.dtype,
        fcompute=fcompute,
        name=name,
        impl="compute",
        output=tensor if isinstance(
            tensor.op, hcl_mlir.TensorOp) else tensor.op.output,
    )
    tensor.add_use(new_tensor)
    Schedule._CurrentSchedule.DataflowGraph.add_edge(
        tensor, new_tensor, stateful=True)

    if Schedule._TopFunction != None:
        stage = new_tensor.op.stage
        stage.__setattr__(tensor.name, new_tensor)
        with get_context() as ctx, get_location() as loc:
            stage.stage_handle = hcl_d.CreateOpHandleOp(
                StringAttr.get(name), ip=hcl_mlir.GlobalInsertionPoint.get()
            )
        Schedule._CurrentStage.append(stage)
        if NestedStageLevel.get() > 0:
            # Attach the stage to the parent stage
            Schedule._CurrentStage[-2].__setattr__(name, stage)
            Schedule._CurrentStage[-2]._sub_stages.append(stage)
        else:
            # Attach the stage to the caller function as an attribute
            caller_func.__setattr__(name, stage)
            # Set up a list of stages for the caller function
            if not hasattr(caller_func, "_stages"):
                caller_func.__setattr__("_stages", [stage])
            else:
                caller_func._stages.append(stage)


def mutate(domain, fcompute, name=None):
    if not isinstance(domain, tuple):
        raise APIError("The domain of mutate API must be a tuple")
    if name is None:
        name = UniqueName.get("tensor")
    
    # Generate a ComputeOp
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    compute_body(name, domain, fcompute, None, loc, "no_alloc")
    return

def old_mutate(domain, fcompute, name=None):
    """
    For now, assume no return value
    """
    # check API correctness
    if not isinstance(domain, tuple):
        raise APIError("The domain of mutate API must be a tuple")
    if name is None:
        name = UniqueName.get("tensor")
    ret_tensor = Tensor(domain, None, name=name,
                        fcompute=fcompute, impl="compute")

    # Check the caller function
    caller_func_name = inspect.stack()[1].function
    if Schedule._TopFunction is None:
        called_from_top = False
    else:
        called_from_top = caller_func_name == Schedule._TopFunction.__name__
    caller_func = get_func_obj(caller_func_name)
    if not called_from_top:
        # Caller function up one level
        caller_parent_func_name = inspect.stack()[2].function
        caller_parent_func = get_func_obj(caller_parent_func_name)
        # If haven't already, attach the caller function
        # to its parent function as an attribute
        if not hasattr(caller_parent_func, caller_func_name):
            caller_parent_func.__setattr__(caller_func_name, caller_func)
    stage = ret_tensor.op.stage
    if Schedule._TopFunction != None:
        if NestedStageLevel.get() > 0:
            # Attach the stage to the parent stage
            # TODO(Niansong): test this
            Schedule._CurrentStage[-1].__setattr__(name, stage)
            Schedule._CurrentStage[-1]._sub_stages.append(stage)
        else:
            caller_func.__setattr__(stage.name, stage.op)
            # Set up a list of stages for the caller function
            if not hasattr(caller_func, "_stages"):
                caller_func.__setattr__("_stages", [stage])
            else:
                caller_func._stages.append(stage)    
    return ret_tensor


def bitcast(tensor, dst_dtype, name=None):
    """Bitcast a HeteroCL tensor or expression to the destination data type of the same bitwidth.
    This API **bitcast** the input tensor from its own data type (source dtype)
    to the destination data type (dst_dtype). The destination data type must have
    the same bitwidth with the source datatype.
    """
    if not isinstance(tensor, Tensor) and not isinstance(tensor, hcl_mlir.ExprOp):
        raise APIError("bitcast input must be HeteroCL Tensor or ExprOp.")

    # check type
    if not isinstance(dst_dtype, Type):
        raise APIError("dst_dtype should be HeteroCL data type.")

    # check bitwidth
    if isinstance(tensor, Tensor):
        src_bitwidth = tensor.dtype.bits
    else:  # ExprOp
        src_bitwidth = hcl_mlir.get_bitwidth(tensor.dtype)
    dst_bitwidth = dst_dtype.bits
    if src_bitwidth != dst_bitwidth:
        raise APIError(
            "Destination datatype bitwidth does not match source bitwidth:"
            + f"source bitwidth: {src_bitwidth} , destination bitwidth {dst_bitwidth}."
        )

    # set up name, shape, and fcompute
    dst_dtype_str = get_dtype_str(dst_dtype)
    if isinstance(tensor, Tensor):
        name = tensor.name + "_" + dst_dtype_str if name is None else name
        shape = tensor.shape
        fcompute = lambda *args: hcl_mlir.BitCastOp(
            hcl_dtype_to_mlir(dst_dtype), tensor[args]
        )
        return compute(shape, fcompute, name=name, dtype=dst_dtype)
    else:
        bitcast = hcl_mlir.BitCastOp(hcl_dtype_to_mlir(dst_dtype), tensor)
        builder = hcl_mlir.ASTVisitor(mode="build")
        builder.visit(bitcast)
        # return an expression
        return bitcast


def cast_np(np_array, dtype):
    """
    Cast a numpy array to a HeteroCL data type.
    """
    if not isinstance(np_array, np.ndarray):
        raise APIError("cast_np input must be numpy array.")
    if isinstance(dtype, str):
        dtype = dtype_to_hcl(dtype)
    elif not isinstance(dtype, Type):
        raise APIError("dtype should be HeteroCL data type.")
    return asarray(np_array, dtype).asnumpy()


def match(scope, pattern):
    """Match the pattern in the given scope.
    Parameters
    ----------
    scope : Scope
        The scope to be matched. Either a function or a stage.
    pattern : Pattern
        The pattern to be matched. Python regular expression.
    Returns
    -------
    matched : list
        A list of matched stages.
    """
    # Check if scope is a function or a stage
    if not inspect.isfunction(scope) and not isinstance(scope, Stage):
        raise APIError("The scope of match API must be a function or a stage.")
    if not isinstance(pattern, str) and not inspect.isfunction(pattern):
        raise APIError("The pattern of match API must be a string or a lambda function.")
    
    matched = []
    if isinstance(pattern, str):
        # Check if pattern is a valid regular expression
        try:
            re.compile(pattern)
        except re.error:
            raise APIError("The pattern of match API must be a valid regular expression.")

    def _ismatch(pattern, stage):
        if isinstance(pattern, str):
            return re.match(pattern, stage.name)
        else:
            return pattern(stage)

    # Check if scope is the top function
    if inspect.isfunction(scope):
        if scope == Schedule._TopFunction:
            # search in the top function
            for _, stage in Stage._mapping:
                if _ismatch(pattern, stage):
                    if stage not in matched:
                        matched.append(stage)
        else: # search in local function
            for stage in scope._stages:
                if _ismatch(pattern, stage):
                    if stage not in matched:
                        matched.append(stage)
    else: # search in stage
        for stage in scope._sub_stages:
            if _ismatch(pattern, stage):
                if stage not in matched:
                    matched.append(stage)
    return matched