# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir import APIError, DTypeWarning

from .ast import ast
from .types import Index, Float, Int, UInt, Struct
from .type_rules import get_type_rules, TypeRule


class TypeInference:
    """A type inference engine for HeteroCL programs."""

    def __init__(self):
        self._rules = get_type_rules()
        self.build_rule_dict()

    def build_rule_dict(self):
        """Build a dictionary of rules, where the key is the operation type"""
        self._rule_dict = {}
        for type_rule in self._rules:
            if not isinstance(type_rule, TypeRule):
                raise TypeError(f"type_rule must be a TypeRule, not {type(type_rule)}")
            for op_type in type_rule.OpClass:
                self._rule_dict[op_type] = type_rule

    # pylint: disable=too-many-return-statements
    def infer(self, expr):
        """Infer the type of an expression"""
        if isinstance(expr, ast.LoadOp):
            return self.infer_load(expr)
        if isinstance(expr, ast.BinaryOp):
            return self.infer_binary(expr)
        if isinstance(expr, ast.SelectOp):
            return self.infer_select(expr)
        if isinstance(expr, ast.ConstantOp):
            return self.infer_const(expr)
        if isinstance(expr, ast.IterVar):
            return Index()
        if isinstance(expr, ast.CastOp):
            return expr.dtype
        if isinstance(expr, ast.ReduceOp):
            # TODO: infer the type of the reduction
            return expr.dtype
        if isinstance(expr, ast.BitCastOp):
            return expr.dtype
        if isinstance(expr, ast.GetBitOp):
            return expr.dtype
        if isinstance(expr, ast.GetSliceOp):
            return expr.dtype
        if isinstance(expr, ast.SetBitOp):
            return self.infer(expr.expr)
        if isinstance(expr, ast.SetSliceOp):
            return self.infer(expr.expr)
        if isinstance(expr, ast.BitReverseOp):
            return self.infer(expr.expr)
        if isinstance(expr, ast.StructConstructOp):
            return expr.dtype
        if isinstance(expr, ast.StructGetOp):
            assert isinstance(expr.struct.dtype, Struct)
            struct_t = expr.struct.dtype
            key_list = list(struct_t.dtype_dict.keys())
            key = key_list[expr.field]
            return struct_t.dtype_dict[key]
        if isinstance(expr, ast.CallOp):
            return self.infer(expr.rets[0])
        if isinstance(expr, ast.Neg):
            return self.infer(expr.expr)
        if isinstance(expr, ast.MathTanhOp):
            return Float(64)
        raise APIError(
            f"Type inference not defined for expression of type: {type(expr)}"
        )

    def infer_binary(self, expr):
        lhs_type = self.infer(expr.lhs)
        rhs_type = self.infer(expr.rhs)
        if isinstance(lhs_type, tuple):
            lhs_type = lhs_type[-1]
        if isinstance(rhs_type, tuple):
            rhs_type = rhs_type[-1]
        # find the rule set based on the operation type
        if type(expr) not in self._rule_dict:
            raise APIError(f"Typing rules not defined for operation type: {type(expr)}")
        type_rule = self._rule_dict[type(expr)]
        res_type = type_rule(lhs_type, rhs_type)
        # MLIR limitation: modulo only supports integer <= 128 bits
        if isinstance(expr, ast.Mod) and isinstance(res_type, (Int, UInt)):
            if res_type.bits > 128:
                DTypeWarning("Modulo only supports integer <= 128 bits").warn()
                res_type = Int(128) if isinstance(res_type, Int) else UInt(128)
        return res_type

    def infer_select(self, expr):
        true_type = self.infer(expr.true_value)
        false_type = self.infer(expr.false_value)
        if type(expr) not in self._rule_dict:
            raise APIError(f"Typing rules not defined for operation type: {type(expr)}")
        type_rule = self._rule_dict[type(expr)]
        res_type = type_rule(true_type, false_type)
        return res_type

    def infer_load(self, expr):
        return expr.tensor.dtype

    def infer_const(self, expr):
        return expr.dtype
