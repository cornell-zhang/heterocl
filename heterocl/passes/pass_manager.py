# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ..ast import ast
from hcl_mlir.exceptions import *
from hcl_mlir.ir import *


class Pass(object):
    """Base class for all intermediate pass.

    A pass is a visitor that can mutate the Intermediate Layer.
    """

    def __init__(self, name):
        self.name = name  # name of the pass

    def apply(self, _ast):
        """Apply the pass to the AST."""
        raise HCLNotImplementedError(
            "Pass.apply() is not implemented for pass: " + self.name
        )

    def update_level(self, op):
        """Update the level of an operation and its children.

        Parameters
        ----------
        op : intermediate.Operation
            the operation to be updated
        """
        if hasattr(op, "body") and op.body is not None:
            for body_op in op.body:
                body_op.level = op.level + 1
                self.update_level(body_op)


class PassManager(object):
    """A pass manager that manages a pipeline of passes."""

    def __init__(self):
        self.pipeline = []

    def add_pass(self, pass_class):
        """Add a pass to the pass pipeline."""
        self.pipeline.append(pass_class)

    def run(self, _ast):
        for pass_class in self.pipeline:
            pass_obj = pass_class()
            _ast = pass_obj.apply(_ast)
        return _ast
