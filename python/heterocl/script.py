import heterocl._jit_internal as _jit_internal
from torch.jit.frontend import get_jit_def, get_default_args, get_jit_class_def
from torch.jit._state import (
    _try_get_jit_cached_function,
    _try_get_jit_cached_overloads,
    _set_jit_function_cache,
    _set_jit_overload_cache,
)

def _check_directly_compile_overloaded(obj):
    qual_name = _qualified_name(obj)
    if _jit_internal._get_fn_overloads(qual_name) or _try_get_jit_cached_overloads(obj):
        raise RuntimeError(
            "Function {} cannot be directly compiled because it"
            " is overloaded. It must be used in a context of a function"
            " where its inputs can determine which overload to call.".format(qual_name)
        )

def script_method(fn):
    if not _enabled:
        return fn
    # NOTE: we need to traverse two frames here because the meta-class frame
    # for ScriptModule will be present, as opposed to invoking @script on a
    # a function or invoking define() on a CompilationUnit.
    # The stack will look like:
    #
    # 0. createResolutionCallback()
    # 1. script_method()
    # 2. ScriptModule metaclass frame
    # 3. Surrounding scope
    #
    # createResolutionCallback internally adds 1 to get us to the scope of this
    # function (the calling function). Adding 2 gets us to the proper surrounding scope.
    _rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=2)
    ast = get_jit_def(fn, fn.__name__, self_name="ScriptModule")
    return ScriptMethodStub(_rcb, ast, fn)

def _check_directly_compile_overloaded(obj):
    qual_name = _qualified_name(obj)
    if _jit_internal._get_fn_overloads(qual_name) or _try_get_jit_cached_overloads(obj):
        raise RuntimeError(
            "Function {} cannot be directly compiled because it"
            " is overloaded. It must be used in a context of a function"
            " where its inputs can determine which overload to call.".format(qual_name)
        )

def script(obj, optimize=None, _frames_up=0, _rcb=None):
    if not _enabled:
        return obj
    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. Use `with torch.jit.optimized_execution() instead"
        )
    if isinstance(obj, ScriptModule):
        raise RuntimeError("HeteroCL currently does not support scripting Module right now.")
    
    if inspect.isclass(obj):
        raise RuntimeError("HeteroCL currently does not support scripting python Class right now.")

    else: 
        _check_directly_compile_overloaded(obj)
        maybe_already_compiled_fn = _try_get_jit_cached_function(obj)
        if maybe_already_compiled_fn:
            return maybe_already_compiled_fn
        ast = get_jit_def(obj, obj.__name__)
        if _rcb is None:
            _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
        fn = torch._C._jit_script_compile(
            qualified_name, ast, _rcb, get_default_args(obj)
        )
        fn.__doc__ = obj.__doc
        _set_jit_function_cache(obj, fn)
        return fn

