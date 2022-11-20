# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #
from .ir import intermediate as itmd
from .types import *
from .type_rules import get_type_rules, TypeRule

class TypeInfer(object):
    """A type inference engine for HeteroCL programs.
    """
    def __init__(self):
        self._rules = get_type_rules()
        self.build_rule_dict()

    def build_rule_dict(self):
        """Build a dictionary of rules, where the key is the operation type
        """
        self._rule_dict = dict()
        for type_rule in self._rules:
            if not isinstance(type_rule, TypeRule):
                raise TypeError(f"type_rule must be a TypeRule, not {type(type_rule)}")
            for op_type in type_rule.OpClass:
                self._rule_dict[op_type] = type_rule

    def infer(self, expr):
        """Infer the type of an expression
        """
        if isinstance(expr, itmd.LoadOp):
            return self.infer_load(expr)
        elif isinstance(expr, itmd.BinaryOp):
            return self.infer_binary(expr)
        elif isinstance(expr, itmd.SelectOp):
            return self.infer_select(expr)
        elif isinstance(expr, itmd.ConstantOp):
            return self.infer_const(expr)
        elif isinstance(expr, itmd.IterVar):
            return Index()
        elif isinstance(expr, itmd.CastOp):
            return expr.dtype
        elif isinstance(expr, itmd.SumOp):
            # TODO: infer the type of the reduction
            return expr.dtype
        elif isinstance(expr, itmd.BitCastOp):
            return expr.dtype
        elif isinstance(expr, itmd.GetBitOp):
            return UInt(1)
        elif isinstance(expr, itmd.GetSliceOp):
            return expr.dtype
        elif isinstance(expr, itmd.BitReverseOp):
            return self.infer(expr.expr)
        else:
            raise APIError(f"Type inference not defined for expression of type: {type(expr)}")

    def infer_binary(self, expr):
        lhs_type = self.infer(expr.lhs)    
        rhs_type = self.infer(expr.rhs)
        # find the rule set based on the operation type
        if type(expr) not in self._rule_dict:
            raise APIError(f"Typing rules not defined for operation type: {type(expr)}")
        type_rule = self._rule_dict[type(expr)]
        res_type = type_rule(lhs_type, rhs_type)
        return res_type

    def infer_select(self, expr):
        true_type = self.infer(expr.true_value)
        false_type = self.infer(expr.false_value)
        if type(expr) not in self._rule_dict:
            raise APIError(f"Typing rules not defined for operation type: {type(expr)}")
        type_rule = self._rule_dict[type(expr)]
        res_type = type_rule(true_type, false_type)
        return res_type

    """
        Operations that do not require type inference
    """
    def infer_load(self, expr):
        return expr.tensor.dtype

    def infer_const(self, expr):
        return expr.dtype