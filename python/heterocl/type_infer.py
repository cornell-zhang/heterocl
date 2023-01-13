# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #
from .ast import ast
from .types import *
from .type_rules import get_type_rules, TypeRule


class TypeInfer(object):
    """A type inference engine for HeteroCL programs."""

    def __init__(self):
        self._rules = get_type_rules()
        self.build_rule_dict()

    def build_rule_dict(self):
        """Build a dictionary of rules, where the key is the operation type"""
        self._rule_dict = dict()
        for type_rule in self._rules:
            if not isinstance(type_rule, TypeRule):
                raise TypeError(f"type_rule must be a TypeRule, not {type(type_rule)}")
            for op_type in type_rule.OpClass:
                self._rule_dict[op_type] = type_rule

    def infer(self, expr):
        """Infer the type of an expression"""
        if isinstance(expr, ast.LoadOp):
            return self.infer_load(expr)
        elif isinstance(expr, ast.BinaryOp):
            return self.infer_binary(expr)
        elif isinstance(expr, ast.SelectOp):
            return self.infer_select(expr)
        elif isinstance(expr, ast.ConstantOp):
            return self.infer_const(expr)
        elif isinstance(expr, ast.IterVar):
            return Index()
        elif isinstance(expr, ast.CastOp):
            return expr.dtype
        elif isinstance(expr, ast.SumOp):
            # TODO: infer the type of the reduction
            return expr.dtype
        elif isinstance(expr, ast.BitCastOp):
            return expr.dtype
        elif isinstance(expr, ast.GetBitOp):
            return expr.dtype
        elif isinstance(expr, ast.GetSliceOp):
            return expr.dtype
        elif isinstance(expr, ast.BitReverseOp):
            return self.infer(expr.expr)
        elif isinstance(expr, ast.StructConstructOp):
            return expr.dtype
        elif isinstance(expr, ast.StructGetOp):
            assert isinstance(expr.struct.dtype, Struct)
            struct_t = expr.struct.dtype
            key_list = list(struct_t.dtype_dict.keys())
            key = key_list[expr.field]
            return struct_t.dtype_dict[key]
        elif isinstance(expr, ast.CallOp):
            return self.infer(expr.rets[0])
        elif isinstance(expr, ast.Neg):
            return self.infer(expr.expr)
        elif isinstance(expr, ast.MathTanhOp):
            return Float(64)
        else:
            raise APIError(
                f"Type inference not defined for expression of type: {type(expr)}"
            )

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
