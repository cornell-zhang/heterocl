from ..ast import ast
from .pass_manager import Pass
from hcl_mlir.exceptions import *


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
        """Pass entry point"""
        for op in _ast.region:
            self.visit(op)

        return _ast

    def nest_elif(self, scope):
        """Convert all elif into nested if-else statements in a given scope.

        Parameters
        ----------
        scope : ast.Operation
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
                if len(chains) == 0 or not isinstance(
                    chains[-1][-1], (ast.IfOp, ast.ElseIfOp)
                ):
                    raise APIError("elif must follow an if or elif")
                chains[-1].append(op)
            elif isinstance(op, ast.ElseOp):
                if len(chains) == 0 or not isinstance(
                    chains[-1][0], (ast.IfOp, ast.ElseIfOp)
                ):
                    raise APIError("else must follow an if or elif")
                chains[-1].append(op)
            else:
                continue

        # convert if-elif-else chains into nested if-else statements
        for chain in chains:
            # convert the first if
            self.nest_elif(chain[0])

            if len(chain) == 1:
                continue

            for i in range(len(chain) - 1):
                # convert elseif to if
                if isinstance(chain[i + 1], ast.ElseIfOp):
                    # The body of ElseIfOp is also a scope, we need to
                    # recursively convert all elif in the body
                    self.nest_elif(chain[i + 1])
                    if_op = ast.IfOp(chain[i + 1].cond, chain[i + 1].loc)
                    if_op.body.extend(chain[i + 1].body)
                    if_op.level += 1
                    self.update_level(if_op)
                    chain[i].else_body.append(if_op)
                    chain[i].else_branch_valid = True
                    chain[i + 1] = if_op
                elif isinstance(chain[i + 1], ast.ElseOp):
                    # The body of ElseOp is also a scope, we need to
                    # recursively convert all elif in the body
                    self.nest_elif(chain[i + 1])
                    chain[i].else_body.extend(chain[i + 1].body)
                    chain[i].else_branch_valid = True
                    for op in chain[i].else_body:
                        op.level += 1
                        self.update_level(op)
                    chain.remove(chain[i + 1])
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
