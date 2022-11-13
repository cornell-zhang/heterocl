# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

from . import intermediate as itmd
from hcl_mlir.exceptions import *

class Pass(object):
    """Base class for all intermediate pass.

    A pass is a visitor that can mutate the Intermediate Layer.
    """
    def __init__(self, name, intermediate):
        self.name = name # name of the mutator
        self.itmd = intermediate # the IR to be mutated, must be intermediate layer IR

    def apply(self):
        """Apply the pass to the intermediate layer IR."""
        raise HCLNotImplementedError("Pass.apply() is not implemented for pass: " + self.name)



class NestElseIf(Pass):
    """Convert all elif into nested if-else statements.

    We need this pass to convert all elif into nested if-else statements
    because MLIR does not support elif.
    """

    def __init__(self, intermediate):
        super().__init__("NestElseIf", intermediate)

    def visit(self, op):
        if hasattr(op, "body"):
            self.nest_elif(op)
            for op in op.body:
                # recursively visit the body
                self.visit(op)

    def apply(self):
        """ Pass entry point
        """
        top_func = self.itmd.top_func
        self.nest_elif(top_func)
        for op in top_func.body:
            self.visit(op)

    def nest_elif(self, scope):
        """ Convert all elif into nested if-else statements in a given scope.

        Parameters
        ----------
        scope : intermediate.Operation
            an operation that has a body, the body is the scope to be converted.
        """

        if not hasattr(scope, "body"):
            raise APIError("The scope passed to nest_elif must have a body")

        # if-elif-else chains
        chains = list()
        for op in scope.body:
            if isinstance(op, itmd.IfOp):
                chains.append([op])
            elif isinstance(op, itmd.ElseIfOp):
                if not chains:
                    raise APIError("elif must follow an if")
                chains[-1].append(op)
            elif isinstance(op, itmd.ElseOp):
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
                if isinstance(chain[i+1], itmd.ElseIfOp):
                    if_op = itmd.IfOp(chain[i+1].cond, chain[i+1].loc)
                    if_op.body.extend(chain[i+1].body)
                    self.update_level(if_op)
                    chain[i].else_body.append(if_op)
                    chain[i].else_branch_valid = True
                    chain[i+1] = if_op
                elif isinstance(chain[i+1], itmd.ElseOp):
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
            if isinstance(op, itmd.ElseIfOp) or isinstance(op, itmd.ElseOp):
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