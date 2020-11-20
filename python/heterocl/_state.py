import torch
import os
import weakref

_python_cu = torch._C.CompilationUnit()

def _try_get_jit_cached_overloads(key):
    qual_names = _jit_function_overload_caching.get(key, None)
    if qual_names:
        return [_python_cu.find_function(qual_name) for qual_name in qual_names]
    else:
        return None

def _try_get_jit_cached_function(key):
    if getattr(key, "__disable_jit_function_caching__", False) is True:
        return None
    qual_name = _jit_caching_layer.get(key, None)
    if qual_name:
        return _python_cu.find_function(qual_name)
    else:
        return None

def _set_jit_function_cache(key, value):
    # only free functions currently supported
    assert isinstance(value, torch.jit.ScriptFunction)
    _jit_caching_layer[key] = value.qualified_name


