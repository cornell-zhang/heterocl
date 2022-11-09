# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #
from ir import intermediate as itmd
from type_rules import *

class TypeInfer(object):
    """A type inference engine for HeteroCL programs.
    """
    def __init__(self):
        self._rules = []
        self._rules.append(add_sub_rule())
        # ...
        self.build_rule_dict()

    def build_rule_dict(self):
        """Build a dictionary of rules, where the key is the operation type
        """
        pass

    def infer(self, expr):
        """Infer the type of an expression
        """
        if isinstance(expr, itmd.AddOp):
            self.infer_add(expr)


    def infer_add(self, expr):
        """Infer the type of an add operation
        """
        lhs_type = self.infer(expr.lhs)    
        rhs_type = self.infer(expr.rhs)
        # apply a rule to infer the type of the add operation
        # if it is not defined, throw an error
        type_rule = self._rules[type(expr)]
        itypes = [lhs_type, rhs_type]
        res_type = type_rule(itypes)
        return res_type