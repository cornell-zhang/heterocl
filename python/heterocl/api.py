"""This module contains all HeteroCL APIs"""
#pylint: disable=no-member
import numbers
from ordered_set import OrderedSet
from .tvm.build_module import build as _build, lower as _lower
from .tvm.api import convert, _IterVar
from .tvm import _api_internal as tvm_api
from .tvm import schedule as _schedule
from .tvm import call_intrin
from .tvm import expr as _expr, stmt as _stmt, make as _make
from .tensor import Scalar, Tensor, TensorSlice
from .schedule import Stage, Schedule
from .scheme import Scheme
from . import util
from . import types
from . import config

def init(init_dtype="int32", raise_assert_exception=True):
    """Initialize a HeteroCL environment with configurations.

    This API must be called each time the users write an application.
    Within the same HeteroCL environment, users can try different
    combinations of customization primitives.

    Parameters
    ----------
    init_dtype : Type, optional
        The default data type for each variables

    Examples
    --------
    .. code-block:: python

        # app 1
        hcl.init()
        A = hcl.placeholder(...)
        B = hcl.placeholder(...)
        def app1(A, B):
            # define the algorithm for app 1
        s = hcl.create_scheme([A, B], app1)
        # apply customization primitives
        f1 = hcl.build(s)
        # execute f1

        # app 2 - initialize again with a different data type
        hcl.init(hcl.Float())
        A = hcl.placeholder(...)
        B = hcl.placeholder(...)
        C = hcl.placeholder(...)
        def app2(A, B, C):
            # define the algorithm for app 2
        s = hcl.create_scheme([A, B, C], app2)
        f2 = hcl.build(s)
        # execute f2
    """
    # set the configurations
    config.init_dtype  = init_dtype
    config.raise_assert_exception = raise_assert_exception
    # initialize global variables
    Schedule.stage_ops = []
    Schedule.stage_names = set()
    Schedule.mod_calls = dict()
    Schedule.last_stages = OrderedSet([])
    Scheme.current = None

def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs.

    If the shape is an empty tuple, the returned value is a scalar.

    Parameters
    ----------
    shape : tuple
        The shape of the placeholder

    name : str, optional
        The name of the placeholder

    dtype : Type, optional
        The data type of the placeholder

    Returns
    -------
    Scalar or Tensor

    Examples
    --------
    .. code-block:: python

        # scalar - cannot be updated
        a = hcl.placeholder((), "a")
        # 1-dimensional tensor - can be updated
        A = hcl.placeholder((1,), "A")
    """
    name = util.get_name("placeholder", name)
    name = util.legalize_name(name)
    dtype = util.get_dtype(dtype)
    tvm_dtype = types.dtype_to_str(dtype)

    if shape == ():
        return Scalar(tvm_api._Var(name, tvm_dtype))
    tensor = Tensor(shape, dtype, name)
    tensor.tensor = tvm_api._Placeholder(tensor.buf.shape, tvm_dtype, name)

    # placeholder is also a stage
    stage = Stage(name)
    stage._op = tensor.tensor
    stage._buf = tensor._buf
    tensor.first_update = stage
    tensor.last_update = stage
    return tensor

def create_scheme(inputs, func):
    """Create a quantization scheme.

    The first argument is a list of inputs to the second argument, which is a
    function the defines the algorithm. The numbers of arguments should match.
    The function will be set with attributes for later optimizations. This
    API returns an object that has two methods: `quantize` and `downsize`.

    Parameters
    ----------
    inputs : Tensor or list of Tensor
        A list of placeholders that are inputs to the algorithm. It can be a
        single tensor

    func : callable
        A function that defines the algorithm

    Returns
    -------
    Scheme

    See Also
    --------
    scheme.Scheme.downsize, scheme.Scheme.quantize

    Examples
    --------
    .. code-block:: python

        A = hcl.placeholder((10,))
        def algo(A):
            return hcl.compute(A.shape, lambda x: A[x]+1, "B")
        s = hcl.create_scheme(A, algo)
        s.downsize(algo.B, hcl.Int(8))
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    # reset the global variables
    Schedule.stage_ops = []
    Schedule.mod_calls = dict()
    Schedule.stage_names = set()
    Schedule.last_stages = OrderedSet([])
    with Stage("_top") as top:
        func(*inputs)
    for op in top.substages:
        func.__setattr__(op.name, op)
    return Scheme(inputs, func)

def create_schedule(inputs, func=None, name=""):
    """Create a schedule for compute optimizations.

    The first argument is a list of inputs to the second argument, which is a
    function the defines the algorithm. The numbers of arguments should match.
    The function will be set with attributes for later optimizations.

    Parameters
    ----------
    inputs : Tensor or list of Tensor
        A list of placeholders that are inputs to the algorithm. It can be a
        single tensor

    func : callable, optional
        A function that defines the algorithm

    Returns
    -------
    Schedule

    See Also
    --------
    create_scheme, create_schedule_from_scheme

    Notes
    -----
    If the function is not provided, we can also create a schedule. However,
    users cannot create a quantization scheme anymore. We strongly recommend
    users to provide the function.

    Examples
    --------
    .. code-block:: python

        # example 1 - normal usage
        A = hcl.placeholder((10,))
        def algo(A):
            return hcl.compute(A.shape, lambda x: A[x]+1, "B")
        s = hcl.create_schedule(A, algo)
        s[algo.B].unroll(algo.B.axis[0])

        # example 2 - no function is provided
        A = hcl.placeholder((10,))
        B = hcl.compute(A.shape, lambda x: A[x]+1)
        s = hcl.create_schedule(A)
        s[B].unroll(B.axis[0])
    """
    name = util.legalize_name(name)
    outputs = [ ]
    if not isinstance(inputs, list):
        inputs = [inputs]
    if func is not None:
        # reset the global variables
        Schedule.stage_ops = []
        Schedule.mod_calls = dict()
        Schedule.stage_names = set()
        Schedule.last_stages = OrderedSet([])
        # execute the algorithm
        with Stage("_top") as top:
            ret = func(*inputs)
        # append the output tensors to the input list
        if ret is not None:
            if isinstance(ret, tuple):
                outputs = list(ret)
            else:
                outputs.append(ret)
        # let each stage be an attribute of the function
        for op in top.substages:
            #op = stage._op
            func.__setattr__(op.name, op)
    t = Schedule.last_stages
    ops = [t_._op.op for t_ in t]
    s = Schedule(_schedule.create_schedule(ops), inputs, outputs, name)
    return s

def create_schedule_from_scheme(scheme, name=""):
    """Create a schedule from a scheme.

    Parameters
    ----------
    scheme : Scheme
        The quantization scheme that will be applied to the compute schedule

    Returns
    -------
    Schedule

    See Also
    --------
    create_schedule, create_scheme

    Examples
    --------
    .. code-block:: python

        A = hcl.placeholder((10,))
        def algo(A):
            return hcl.compute(A.shape, lambda x: A[x]+1, "B")
        sm = hcl.create_scheme(A, algo)
        sl = hcl.create_schedule_from_scheme(sm)
    """
    name = util.legalize_name(name)
    Scheme.current = scheme
    # reset the values of each tensor
    for i in scheme.inputs:
        if isinstance(i, Tensor):
            i.var_dict = {}
            i.last_update = i.first_update
    return create_schedule(scheme.inputs, scheme.func, name=name)

def lower(schedule):
    """Get the generated IR of a given schedule.

    Parameters
    ----------
    schedule : Schedule
        The schedule that will be used to generate the IR

    Returns
    -------
    Stmt
    """
    new_inputs = []
    for i in schedule.inputs:
        if isinstance(i, Tensor):
            new_inputs.append(i.tensor.op.output(0))
        elif isinstance(i, Stage):
            new_inputs.append(i._op.op.output(0))
        else:
            new_inputs.append(i.var)
    return _lower(schedule.sch, new_inputs, simple_mode=True)

def build(schedule, target=None, name="default_function", stmt=None):
    """Build the executable according to the schedule and target.

    The default target is `llvm` (i.e., CPU execution). If stmt is specified,
    the statements created by HeteroCL APIs will be ignored.

    Parameters
    ----------
    schedule : Schedule
        The schedule to be built

    target : str, optional
        The target of the executable

    name : str, optional
        The name of the generated function

    stmt : Stmt, optional
        The built statement

    Returns
    -------
    tvm.module.Module
    """
    name = util.legalize_name(name)
    new_inputs = []
    for i in schedule.inputs:
        if isinstance(i, Tensor):
            new_inputs.append(i.tensor.op.output(0))
        else:
            new_inputs.append(i.var)

    # auto data moving to dev
    if len(schedule.placement) == 0 and (target is not None):
        if not isinstance(target, str):
            # TODO: print clean info for auto placement
            # import builtins as __builtin__
            # __builtin__.print("[HCL] Auto data placement...")
            inputs = [_ for _ in schedule.inputs if _ not in schedule.outputs]
            for _ in inputs:
                schedule.to(_, target.xcel)
            for _ in schedule.outputs:
                schedule.to(_, target.host)

    if stmt is not None:
        for i in schedule.inputs:
            if isinstance(i, Tensor):
                shapes = []
                for s in i.shape:
                    shapes.append(0)
                    shapes.append(s)
                tpl = tuple(shapes)
                stmt = _make.AttrStmt([i.buf, i.tensor], "buffer_bind_scope",
                        call_intrin('handle', 'tvm_tuple', *tpl), stmt)
    return _build(schedule.sch, new_inputs, target=target, name=name, stmt=stmt, schedule_name=schedule.name)

##############################################################################
# Other useful APIs
##############################################################################

def cast(dtype, expr):
    """Cast an expression to specified data type.

    Parameters
    ----------
    dtype : Type
        The target data type

    expr : Expr
        The expression to be cast

    Returns
    -------
    Expr
    """
    dtype = types.dtype_to_str(dtype)
    return _make.Cast(dtype, expr)


def select(cond, true, false):
    """Construct a select branch with the given condition.

    It is similar to the following Python expression.

    .. code-block:: python

        ret = true if cond else false

    Parameters
    ----------
    cond : Expr
        The condition

    true : Expr
        The true branch

    false : Expr
        The false branch

    Returns
    -------
    Expr
    """
    return _make.Select(convert(cond), convert(true), convert(false))

def print(vals, format=""):
    """Print a HeteroCL object.

    Parameters
    ----------
    vals : Expr or list of Expr
        The values to be printed

    format : string, optional
        The printing format similar to printf

    Returns
    -------
    None
    """
    if not isinstance(vals, (tuple, list)):
        vals = [vals]

    def get_format(val):
        if isinstance(val, (TensorSlice, Scalar, _expr.Expr)):
            if (util.get_type(val.dtype)[0] == "int"
                    or util.get_type(val.dtype)[0] == "uint"):
                return "%lld"
            else:
                return "%f"
        elif isinstance(val, int):
            return "%d"
        elif isinstance(val, float):
            return "%f"

    def print_tensor(val, ivs, i, ndim):
        if i == 0: #inner-most
            iv = ivs[ndim-1]
            stmt = _make.Print([], "[")
            value = val[tuple(ivs)]
            body = _make.Print([value], get_format(value))
            ite = _make.IfThenElse(iv < iv.dom.extent-1,
                                   _make.Print([], ", "),
                                   _make.Evaluate(0))
            body = _make.Block(body, ite)
            loop = _make.For(iv.var, iv.dom.min, iv.dom.extent, 0, 0, body)
            stmt = _make.Block(stmt, loop)
            stmt = _make.Block(stmt, _make.Print([], "]"))
            return stmt
        else:
            iv = ivs[ndim-1-i]
            stmt = _make.Print([], "[")
            body = print_tensor(val, ivs, i-1, ndim)
            ite = _make.IfThenElse(iv < iv.dom.extent-1,
                                   _make.Print([], ",\n"),
                                   _make.Evaluate(0))
            body = _make.Block(body, ite)
            loop = _make.For(iv.var, iv.dom.min, iv.dom.extent, 0, 0, body)
            stmt = _make.Block(stmt, loop)
            stmt = _make.Block(stmt, _make.Print([], "]"))
            return stmt

    def print_val(val):
        stage = Stage.get_current()
        if isinstance(val, (Scalar, _expr.Expr, numbers.Number)):
            stage.emit(_make.Print([val], get_format(val) + "\n"))
        elif isinstance(val, TensorSlice) \
                and len(val.indices) == len(val.tensor.shape):
            stage.emit(_make.Print([val], get_format(val) + "\n"))
        else: # we are dealing with tensors
            nshape = len(val.tensor.shape)
            ndim = nshape
            if isinstance(val, TensorSlice):
                ndim = nshape - len(val.indices)
            args = ["print_"+str(n) for n in range(0, ndim)]
            ivs = [_IterVar((0, val.tensor.shape[n]), args[n], 0) \
                    for n in range(0, ndim)]
            import builtins
            stage.emit(print_tensor(val, ivs, ndim-1, ndim))
            stage.emit(_make.Print([], "\n"))

    if format == "":
        for val in vals:
            print_val(val)
    else:
        stage = Stage.get_current()
        stage.emit(_make.Print(vals, format))

def assert_(cond, message="assert error\n", vals=0):
    """assert a condition in HeteroCL.

    Parameters
    ----------
    cond : boolean
    the condition to be tested

    message : string, optional
        message to be printed when condition is false

    vals: number or array, optional
       message to be printed when condition is false

    Returns
    -------
    None
    """
    if "\n" not in message:
        message = message + "\n"

    if not isinstance(vals, (tuple, list)):
        vals = [vals]
    stage = Stage.get_current()
    stage.emit(_make.Assert(cond, vals, message))
