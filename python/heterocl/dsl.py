"""HeteroCL imperative DSL."""
#pylint: disable=too-many-arguments,missing-docstring
from .tvm import make as _make
from .tvm import stmt as _stmt
from .tvm import ir_pass as _pass
from .tvm._api_internal import _IterVar, _Var
from .tvm.ir_builder import WithScope
from .api import placeholder
from .debug import DSLError, APIError
from .schedule import Stage
from .module import Module
from . import util

def and_(*args):
    """Compute the logic AND between expressions.

    If there is only one argument, itself is returned.

    Parameters
    ----------
    args : list of Expr
        A list of expression to be computed

    Returns
    -------
    Expr

    Examples
    --------
    .. code-block:: python

        A = hcl.placeholder((3,))
        cond = hcl.and_(A[0] > 0, A[1] > 1, A[2] > 2)
    """
    ret = args[0]
    for i in range(1, len(args)):
        ret = _make.And(ret, args[i])
    return ret

def or_(*args):
    """Compute the logic OR between expressions.

    If there is only one argument, itself is returned.

    Parameters
    ----------
    args : list of Expr
        A list of expression to be computed

    Returns
    -------
    Expr

    Examples
    --------
    .. code-block:: python

        A = hcl.placeholder((3,))
        cond = hcl.or_(A[0] > 0, A[1] > 1, A[2] > 2)
    """
    ret = args[0]
    for i in range(1, len(args)):
        ret = _make.Or(ret, args[i])
    return ret

def if_(cond):
    """Construct an IF branch.

    The usage is the same as Python `if` statement. Namely, a single `if`
    statement without the `else` branch is allowed. In addition, we cannot
    use `else` and `elif` without an `if` statement. Finally, an `else`
    statement must be preceded by either an `if` or `elif` statement.

    Parameters
    ----------
    cond : Expr
        The condition of the `if` statement

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        def my_compute(x):
            with hcl.if_(A[x] < 3):
                # do something
            with hcl.elif_(A[x] < 6):
                # do something
            with hcl.else_():
                # do something
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    stage.stmt_stack.append([])
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.emit(_make.IfThenElse(cond, stmt, None))
    return WithScope(None, _exit_cb)

def else_():
    """Construct an ELSE branch.

    Parameters
    ----------

    Returns
    -------
    None

    See Also
    --------
    if_
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    prev = stage.stmt_stack[-1][-1]
    if not isinstance(prev, _stmt.IfThenElse):
        raise DSLError("There is no if_ or elif_ in front of the else_ branch")
    stage.stmt_stack[-1].pop()
    stage.stmt_stack.append([])
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.emit(stage.replace_else(prev, stmt))
    return WithScope(None, _exit_cb)

def elif_(cond):
    """Construct an ELIF branch.

    Parameters
    ----------
    cond : Expr
        The condition of the branch

    Returns
    -------
    None

    See Also
    --------
    if_
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    prev = stage.stmt_stack[-1][-1]
    if not isinstance(prev, _stmt.IfThenElse):
        raise DSLError("There is no if_ or elif_ in front of the elif_ branch")
    stage.stmt_stack[-1].pop()
    stage.stmt_stack.append([])
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.emit(stage.replace_else(prev, _make.IfThenElse(cond, stmt, None)))
    return WithScope(None, _exit_cb)

def for_(begin, end, step=1, name="i", dtype="int32", for_type="serial"):
    """Construct a FOR loop.

    Create an imperative for loop based on the given bound and step. It is
    the same as the following Python code.

    .. code-block:: python

        for i in range(begin, end, step):
            # do something

    The bound and step can be negative values. In addition, `begin` is
    inclusive while `end` is exclusive.

    Parameters
    ----------
    begin : Expr
        The starting bound of the loop

    end : Expr
        The ending bound of the loop

    step : Expr, optional
        The step of the loop

    name : str, optional
        The name of the iteration variable

    dtype : Type, optional
        The data type of the iteration variable

    for_type : str, optional
        The type of the for loop

    Returns
    -------
    Var
        The iteration variable

    See Also
    --------
    break_

    Examples
    --------
    .. code-block:: python

        # example 1 - basic usage
        with hcl.for_(0, 5) as i:
            # i = [0, 1, 2, 3, 4]

        # example 2 - negative step
        with hcl.for_(5, 0, -1) as i:
            # i = [5, 4, 3, 2, 1]

        # example 3 - larger step
        with hcl.for_(0, 5, 2) as i:
            # i = [0, 2, 4]

        # example 4 - arbitrary bound
        with hcl.for_(-4, -8, -2) as i:
            # i = [-4, -6]
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    stage.stmt_stack.append([])
    extent = (end - begin) // step
    extent = util.CastRemover().mutate(extent)
    name = "i"+str(stage.for_ID) if name is None else name
    stage.for_ID += 1
    iter_var = _IterVar(_make.range_by_min_extent(0, extent), _Var(name, dtype), 0, '')
    stage.var_dict[name] = iter_var
    stage.axis_list.append(iter_var)
    stage.for_level += 1
    def _exit_cb():
        if for_type == "serial":
            for_type_id = 0
        elif for_type == "parallel":
            for_type_id = 1
        elif for_type == "vectorize":
            for_type_id = 2
        elif for_type == "unroll":
            for_type_id = 3
        else:
            raise ValueError("Unknown for_type")
        stmt = _make.AttrStmt(iter_var, "loop_scope", iter_var.var, stage.pop_stmt())
        stage.has_break = False
        stage.for_level -= 1
        stage.emit(_make.For(iter_var.var, 0, extent, for_type_id, 0, stmt))
    ret_var = _pass.Simplify(iter_var.var * step + begin)
    return WithScope(ret_var, _exit_cb)

def while_(cond):
    """Construct a WHILE loop.

    Parameters
    ----------
    cond : Expr
        The condition of the loop

    Returns
    -------
    None

    See Also
    --------
    break_

    Examples
    --------
    .. code-block:: python

        with hcl.while_(A[x] > 5):
            # do something
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    stage.stmt_stack.append([])
    stage.for_level += 1
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.for_level -= 1
        stage.emit(_make.While(cond, stmt))
    return WithScope(None, _exit_cb)

def break_():
    """
    Construct a BREAK statement.

    This DSL can only be used inside a `while` loop or a `for loop`. Moreover,
    it is not allowed to have tracing statements after the `break`.

    Parameters
    ----------

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        # example 1 - inside a for loop
        with hcl.for_(0, 5) as i:
            with hcl.if_(A[i] > 5):
                hcl.break_()

        # example 2 - inside a while loop
        with hcl.while_(A[i] > 5):
            with hcl.if_(A[i] > 10):
                hcl.break_()
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    if not Stage.get_current().for_level:
        raise DSLError("break_ must be used inside a for/while loop")
    Stage.get_current().emit(_make.Break())
    Stage.get_current().has_break = True

def def_(shapes, dtypes=None, ret_dtype=None, name=None, arg_names=None):
    """
    Define a HeteroCL function from a Python function.

    This DSL is used as a Python decorator. The function defined with HeteroCL
    is not inlined by default. Users need to provide the shapes of the
    arguments, while the data types of the arguments and the returned data
    type are optional. This DSL helps make the algorithm more organized and
    could potentially reduce the memory usage by reusing the same
    functionality. Users can later on use compute primitives to decide whether
    to inline these functions or not.

    After specifying a Python function to be a HeteroCL function, users can
    use the function just like using a Python function. We can also apply
    optimization primitives.

    Parameters
    ----------
    shapes : list of tuple
        The shapes of the arguments

    dtypes : list of Type, optional
        The data types of the argument

    ret_dtype : Type, optional
        The data type of the returned value

    name : str, optional
        The name of the function. By default, it is the same as the Python
        function

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        # example 1 - no return
        A = hcl.placeholder((10,))
        B = hcl.placeholder((10,))
        x = hcl.placeholder(())

        @hcl.def_([A.shape, B.shape, x.shape])
        def update_B(A, B, x):
            with hcl.for_(0, 10) as i:
                B[i] = A[i] + x

        # directly call the function
        update_B(A, B, x)

        # example 2 - with return value
        @hcl.def_([(10,), (10,), ()])
        def ret_add(A, B, x):
            hcl.return_(A[x] + B[x])

        # use inside a compute API
        A = hcl.placeholder((10,))
        B = hcl.placeholder((10,))
        C = hcl.compute((10,), lambda x: ret_add(A, B, x))
        D = hcl.compute((10,), lambda x: ret_add(A, C, x))
    """
    def decorator(fmodule, shapes=shapes, dtypes=dtypes, ret_dtype=ret_dtype, name=name, arg_names=arg_names):
        name = name if name is not None else fmodule.__name__
        code = fmodule.__code__
        names = code.co_varnames
        if arg_names is not None:
          names = list(names)
          for i in range(len(arg_names)):
            names[i] = arg_names[i]
          names = tuple(names)
        nargs = code.co_argcount

        with Stage(name) as s:
            # prepare names
            new_names = [s.name_with_prefix + "." + name_ for name_ in names]
            # prepare dtypes
            hcl_dtypes = []
            if dtypes is None:
                dtypes = []
                for name_ in new_names:
                    dtypes.append(util.get_tvm_dtype(None, name_))
                    hcl_dtypes.append(util.get_dtype(None, name_))
            elif isinstance(dtypes, list):
                if len(dtypes) != nargs:
                    raise APIError("The number of data types does not match the of arguments")
                for (name_, dtype_) in zip(new_names, dtypes):
                    dtypes.append(util.get_tvm_dtype(dtype_, name_))
                    hcl_dtypes.append(util.get_dtype(dtype_, name_))
                dtypes = dtypes[int(len(dtypes)/2):]
            else:
                dtype = util.get_tvm_dtype(dtypes)
                dtypes = []
                for name_ in new_names:
                    dtypes.append(util.get_tvm_dtype(dtype, name_))
            ret_dtype = util.get_tvm_dtype(ret_dtype, s.name_with_prefix)
            # prepare inputs for IR generation
            inputs = []
            inputs_tvm = []
            arg_shapes, arg_dtypes = [], []
            for shape, name_, dtype, htype in zip(shapes, new_names, dtypes, hcl_dtypes):
                if shape == ():
                    var_ = placeholder((), name_, dtype)
                    inputs.append(var_)
                    inputs_tvm.append(var_.var)
                    arg_shapes.append([1])
                    arg_dtypes.append(dtype)
                else: # tensor inputs (new bufs)
                    placeholder_ = placeholder(shape, name_, htype)
                    inputs.append(placeholder_)
                    inputs_tvm.append(placeholder_.buf.data)
                    arg_shapes.append(list(shape))
                    arg_dtypes.append(dtype)

            s.ret_dtype = ret_dtype
            s._module = True
            s._inputs = inputs
            fmodule(*inputs)
            lhs = []
            for tensor in s.lhs_tensors:
                try:
                    lhs.append(inputs.index(tensor))
                except ValueError:
                    pass
            ret_void = _make.UIntImm("uint1", 0) if s.has_return else _make.UIntImm("uint1", 1)
            body = s.pop_stmt()

            s.stmt_stack.append([])
            s.emit(_make.KernelDef(inputs_tvm, arg_shapes, arg_dtypes,
                                   body, ret_void, ret_dtype, name, []))
            for name_, i in zip(names, inputs):
                s.var_dict[name_] = i
            s.input_stages.clear()

        return Module(shapes, names, name, not s.has_return, lhs, ret_dtype)
    return decorator

def return_(val):
    """Return an expression within a function.

    This DSL should be used within a function definition. The return type can
    only be an expression.

    Parameters
    ----------
    val : Expr
        The returned expression

    Returns
    -------
    None

    See Also
    --------
    heterocl.compute, def_

    Examples
    --------
    .. code-block:: python

        # example 1 - using with a compute API
        A = hcl.placeholder((10,))

        def compute_out(x):
            with hcl.if_(A[x]>0):
                hcl.return_(1)
            with hcl.else_():
                hcl.return_(0)

        B = hcl.compute(A.shape, compute_out)

        # example 2 - using with a HeteroCL function
        A = hcl.placeholder((10,))

        @hcl.def_([A.shape, ()])
        def compute_out(A, x):
            with hcl.if_(A[x]>0):
                hcl.return_(1)
            with hcl.else_():
                hcl.return_(0)

        B = hcl.compute(A.shape, lambda x: compute_out(A, x))
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    dtype = util.get_tvm_dtype(stage.ret_dtype)
    stage.emit(_make.Return(_make.Cast(dtype, val)))
    stage.has_return = True
