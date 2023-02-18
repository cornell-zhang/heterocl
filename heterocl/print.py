# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-builtin

from hcl_mlir.exceptions import HCLValueError

from .ast import ast
from .schedule import get_src_loc


def print(vals, format_str=""):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    if isinstance(vals, ast.AllocOp):
        printOp = ast.PrintTensorOp(vals, loc)
    elif isinstance(vals, (int, float, ast.Expr)):
        value = ast.immediate_to_constant(vals, loc)
        printOp = ast.PrintOp([value], format_str, loc)
    elif isinstance(vals, (tuple, list)):
        # When vals is an tuple
        if isinstance(vals, tuple):
            vals = list(vals)
        # Check if all elements are int/float or expression
        for i, val in enumerate(vals):
            if isinstance(val, (int, float)):
                continue
            if isinstance(val, ast.Expr):
                continue
            raise HCLValueError(
                f"Unsupported type of element in tuple: {val} of type {type(val)}"
            )
        # when vals is empty
        if len(vals) == 0:
            value = ast.immediate_to_constant(0, loc)
            vals = [value]
        # when v in vals is int or float
        for i, v in enumerate(vals):
            vals[i] = ast.immediate_to_constant(v, loc)
        printOp = ast.PrintOp(vals, format_str, loc)
    else:
        raise HCLValueError(f"Unsupported type for print: {vals}")
    region = ast.scope.get()
    region.append(printOp)
