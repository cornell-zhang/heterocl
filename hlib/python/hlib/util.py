from numbers import Integral

import heterocl.tvm as tvm

def equal_const_int(expr, value):
    if isinstance(expr, Integral):
        return expr == value
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        expr = tvm.ir_pass.Simplify(expr)
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        return False
    return expr.value == value