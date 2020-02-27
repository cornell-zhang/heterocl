"""This module contains all HeteroCL APIs"""
#pylint: disable=no-member
import networkx as nx
from ordered_set import OrderedSet
from .tvm.build_module import build as _build, lower as _lower
from .tvm.api import convert
from .tvm import _api_internal as tvm_api
from .tvm import schedule as _schedule
from .tvm import make as _make
from .tvm import call_intrin
from .tensor import Scalar, Tensor
from .schedule import Stage, Schedule
from .scheme import Scheme
from . import util
from . import types
from . import config

def init(init_dtype="int32", tuning=False):
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
    # initialize global variables
    Schedule.stage_ops = []
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
    func(*inputs)
    for op in Schedule.stage_ops:
        func.__setattr__(op.name, op)
    return Scheme(inputs, func)

def create_schedule(inputs, func=None):
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
    if not isinstance(inputs, list):
        inputs = [inputs]
    if func is not None:
        # reset the global variables
        Schedule.stage_ops = []
        Schedule.last_stages = OrderedSet([])
        # execute the algorithm
        ret = func(*inputs)
        # append the output tensors to the input list
        if ret is not None:
            if isinstance(ret, tuple):
                inputs += list(ret)
            else:
                inputs.append(ret)
        # let each stage be an attribute of the function
        for op in Schedule.stage_ops:
            func.__setattr__(op.name, op)
    t = Schedule.last_stages
    ops = [t_._op.op for t_ in t]
    return Schedule(_schedule.create_schedule(ops), inputs)

def create_schedule_from_scheme(scheme):
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
    Scheme.current = scheme
    # reset the values of each tensor
    for i in scheme.inputs:
        if isinstance(i, Tensor):
            i.var_dict = {}
            i.last_update = i.first_update
    return create_schedule(scheme.inputs, scheme.func)

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
    new_inputs = []
    for i in schedule.inputs:
        if isinstance(i, Tensor):
            new_inputs.append(i.tensor.op.output(0))
        else:
            new_inputs.append(i.var)
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
    return _build(schedule.sch, new_inputs, target=target, name=name, stmt=stmt)

def tune(s, func, target, workload=None):
    """ Extract auto-tuning tasks and apply schedules"""
    g = s.dataflow_graph()
    s_dataflow, s_compute = set(), dict()
    for name in nx.topological_sort(g):
      try: # get hcl stage ops 
        stage = getattr(func, name)
        # declarative op applys
        if len(stage.axis) > 0:
          s_compute[s[stage]] = stage.axis
        # imperative externop 
        if len(stage.var_dict) > 0:  
          for k, v in stage.var_dict.items(): 
            if isinstance(v, Tensor):
              s_dataflow.add(v)
      except Exception as e: # placeholder
        pass

    try: # import uptune
        import uptune as ut
        from uptune.tuners import bandit
        ut.init(apply_best=False)
    except ImportError as e:
        assert False, "uptune not installed" 

    # apply schedule
    def sch_apply(s, sch, varlist, name):
        try: # tolerate fault
            prmtv = getattr(s, sch) 
            val = ut.tune(0, list(range(len(varlist))), 
                          name=name) 
            prmtv(varlist[val])
        except Exception as e: 
          print("Failed to apply {}".format(sch))
          print("Error: ".format(str(e)))

    # design space (pipeline, split)
    for k, v in s_compute.items():
      # apply parallel, pipeline & reorder 
      name = str(k).split(",")[0].split("(")[-1]
      itervars = [ v[_] for _ in range(len(v)) ]
      if ut.tune(True, (), name=name+"_gate"):
        sch_apply(k, "parallel", itervars, name=name+"_parallel")
        sch_apply(k, "pipeline", itervars, name=name+"_pipeline")

    # infer tuning mode
    def get_qor(tool):
        if "vivado" in str(tool): 
          path = "__tmp__/out.prj/solution1/syn/report/top_csynth.xml"
          return ut.vhls, path
        elif "quartus" in str(tool):
          return ut.quartus

    f = build(s, target)
    metrics, path = get_qor(target.tool)
    assert callable(metrics), "metrics not callble"

    if not workload: # optimize for specific workload
        worload = 1 

    f(*workload)
    del f, s, target
    qor = float(metrics(path)['avg'])
    ut.target(qor)

    ut.config["gpu-num"] = 0
    ut.config["test-limit"] = 5
    ut.tune(tuner=bandit(ut.config))

def autosch(sch, func, target, plot=False):
    # assert len(sch.placement) == 0, "placement not empty"
    from .auto_sch import auto_sch
    auto_sch(sch, func, target, plot)
    print(sch); print(lower(sch)); import sys; sys.exit()

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
