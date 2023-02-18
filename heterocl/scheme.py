# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import APIError, HCLValueError

from . import types
from .schedule import _reset_builder, _build_schedule, _build_ast, Stage
from .ast import ast


def create_scheme(inputs, func):
    """Create a quantization scheme."""
    try:
        if not isinstance(inputs, list):
            inputs = [inputs]
        return Scheme(inputs, func)
    except Exception as e:
        raise e
    finally:
        _reset_builder()


def create_schedule_from_scheme(scheme, name=""):
    """Create a schedule from a scheme."""
    return _build_schedule(scheme._ast, scheme.inputs, scheme.func, name=name)


class Scheme:
    """A quantization scheme."""

    def __init__(self, inputs, func):
        self.inputs = inputs
        self.func = func
        self._ast = _build_ast(inputs, func)

    def downsize(self, inputs, dtype):
        """Downsize a (list of) tensor to the specified integer type."""
        if not isinstance(dtype, (types.Int, types.UInt)):
            raise APIError("Downsize to non-integer type is not allowed")
        if not isinstance(inputs, list):
            inputs = [inputs]
        for tensor in inputs:
            if isinstance(tensor, ast.AllocOp):
                tensor.dtype = dtype
            elif isinstance(tensor, ast.ComputeOp):
                op = Stage.lookup(tensor.name)
                op.tensor.dtype = dtype
                op.aux_tensor.dtype = dtype
                op.dtype = dtype
            else:
                raise HCLValueError(
                    f"Unexpected op type: {type(tensor)}, input tensor is: {tensor}"
                )

    def quantize(self, inputs, dtype):
        """Quantize a (list of) tensor to the specified fixed-point type."""
        if not isinstance(dtype, (types.Fixed, types.UFixed)):
            raise APIError("Quantize to integer type is not allowed")
        if not isinstance(inputs, list):
            inputs = [inputs]
        for tensor in inputs:
            if isinstance(tensor, ast.AllocOp):
                tensor.dtype = dtype
            elif isinstance(tensor, ast.ComputeOp):
                op = Stage.lookup(tensor.name)
                op.tensor.dtype = dtype
                op.aux_tensor.dtype = dtype
                op.dtype = dtype
            else:
                raise HCLValueError(
                    f"Unexpected op type: {type(tensor)}, input tensor is: {tensor.name}"
                )
