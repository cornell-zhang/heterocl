# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

from . import ast
from hcl_mlir.exceptions import *
from hcl_mlir.ir import *

class Pass(object):
    """Base class for all intermediate pass.

    A pass is a visitor that can mutate the Intermediate Layer.
    """
    def __init__(self, name):
        self.name = name # name of the pass

    def apply(self, _ast):
        """Apply the pass to the AST."""
        raise HCLNotImplementedError("Pass.apply() is not implemented for pass: " + self.name)


class PassManager(object):
    """A pass manager that manages a pipeline of passes.
    """
    def __init__(self):
        self.pipeline = []
    
    def add_pass(self, pass_class):
        """Add a pass to the pass pipeline.
        """
        self.pipeline.append(pass_class)

    def run(self, _ast):
        for pass_class in self.pipeline:
            pass_obj = pass_class()
            _ast = pass_obj.apply(_ast)
        return _ast


class NestElseIf(Pass):
    """Convert all elif into nested if-else statements.

    We need this pass to convert all elif into nested if-else statements
    because MLIR does not support elif.
    """

    def __init__(self):
        super().__init__("nest_else_if")

    def visit(self, op):
        if hasattr(op, "body") and op.body is not None:
            self.nest_elif(op)
            for op in op.body:
                # recursively visit the body operations
                self.visit(op)

    def apply(self, _ast):
        """ Pass entry point
        """
        for op in _ast.region:
            self.visit(op)

        return _ast

    def nest_elif(self, scope):
        """ Convert all elif into nested if-else statements in a given scope.

        Parameters
        ----------
        scope : intermediate.Operation
            an operation that has a body, the body is the scope to be converted.
        """

        if not hasattr(scope, "body"):
            raise APIError("The scope passed to nest_elif must have a body")

        if scope.body is None:
            return

        # if-elif-else chains
        chains = list()
        for op in scope.body:
            if isinstance(op, ast.IfOp):
                chains.append([op])
            elif isinstance(op, ast.ElseIfOp):
                if not chains:
                    raise APIError("elif must follow an if")
                chains[-1].append(op)
            elif isinstance(op, ast.ElseOp):
                if not chains:
                    raise APIError("else must follow an if or elif")
                chains[-1].append(op)
            else:
                continue

        # convert if-elif-else chains into nested if-else statements
        for chain in chains:
            if len(chain) == 1:
                continue
            for i in range(len(chain) - 1):
                # convert elseif to if
                if isinstance(chain[i+1], ast.ElseIfOp):
                    if_op = ast.IfOp(chain[i+1].cond, chain[i+1].loc)
                    if_op.body.extend(chain[i+1].body)
                    self.update_level(if_op)
                    chain[i].else_body.append(if_op)
                    chain[i].else_branch_valid = True
                    chain[i+1] = if_op
                elif isinstance(chain[i+1], ast.ElseOp):
                    chain[i].else_body.extend(chain[i+1].body)
                    chain[i].else_branch_valid = True
                    for op in chain[i].else_body:
                        self.update_level(op)
                    chain.remove(chain[i+1])
                else:
                    raise APIError("Invalid if-elif-else chain: " + str(chain))

        # remove ElseIfOp and ElseOp
        new_body = list()
        for op in scope.body:
            if isinstance(op, ast.ElseIfOp) or isinstance(op, ast.ElseOp):
                pass
            else:
                new_body.append(op)
        scope.body = new_body

    def update_level(self, op):
        """ Update the level of an operation and its children.

        Parameters
        ----------
        op : intermediate.Operation
            the operation to be updated
        """
        if hasattr(op, "body"):
            op.level += 1
            for op in op.body:
                self.update_level(op)