import contextlib
import collections
import enum
import inspect
import weakref
import warnings
import torch
import sys


def createResolutionCallbackFromEnv(lookup_base):
    """
    Creates a resolution callback that will look up qualified names in an
    environment, starting with `lookup_base` for the base of any qualified
    names, then proceeding down the lookup chain with the resolved object.

    You should not use this directly, it should only be used from the other
    createResolutionCallbackFrom* functions.
    """
    def lookupInModule(qualified_name, module):
        if '.' in qualified_name:
            parts = qualified_name.split('.')
            base = parts[0]
            remaining_pieces = '.'.join(parts[1:])
            module_value = getattr(module, base)
            return lookupInModule(remaining_pieces, module_value)
        else:
            return getattr(module, qualified_name)

    def parseNestedExpr(expr, module) -> Tuple[Any, int]:
        i = 0
        while i < len(expr) and expr[i] not in (',', '[', ']'):
            i += 1

        base = lookupInModule(expr[:i].strip(), module)
        assert base is not None, f"Unresolvable type {expr[:i]}"
        if i == len(expr) or expr[i] != '[':
            return base, i

        assert expr[i] == '['
        parts = []
        while expr[i] != ']':
            part_len = 0
            i += 1
            part, part_len = parseNestedExpr(expr[i:], module)
            parts.append(part)
            i += part_len
        if len(parts) > 1:
            return base[tuple(parts)], i + 1
        else:
            return base[parts[0]], i + 1
    def parseExpr(expr, module):
        try:
            value, len_parsed = parseNestedExpr(expr, module)
            assert len_parsed == len(expr), "whole expression was not parsed, falling back to c++ parser"
            return value
        except Exception as e:
            """
            The python resolver fails in several cases in known unit tests, and is intended
            to fall back gracefully to the c++ resolver in general.  For example, python 2 style
            annotations which are frequent in our unit tests often fail with types e.g. int not
            resolvable from the calling frame.
            """
            return None

    return lambda expr: parseExpr(expr, lookup_base)

def get_closure(fn):
    """
    Get a dictionary of closed over variables from a function
    """
    captures = {}
    captures.update(fn.__globals__)

    for index, captured_name in enumerate(fn.__code__.co_freevars):
        captures[captured_name] = fn.__closure__[index].cell_contents

    return captures

def createResolutionCallbackFromClosure(fn):
    """
    Create a resolutionCallback by introspecting the function instead of
    looking up the stack for the enclosing scope
    """
    closure = get_closure(fn)

    class closure_lookup(object):
        # This is a class since `closure` is a dict and it's easier in
        # `env_helper` if everything just works with `getattr` calls
        def __getattr__(self, key):
            if key in closure:
                return closure[key]
            elif hasattr(builtins, key):
                return getattr(builtins, key)
            return None

    return createResolutionCallbackFromEnv(closure_lookup())
