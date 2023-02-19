# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-public-methods

from . import ast_visitor


class ASTCleaner(ast_visitor.ASTVisitor):
    def __init__(self) -> None:
        super().__init__("cleaner")

    def visit_ast(self, _ast, *args, **kwargs):
        for op in _ast.region:
            self.visit(op, *args, **kwargs)

    def visit_func(self, op, *args, **kwargs):
        op.ir_op = None
        for body_op in op.body:
            self.visit(body_op, *args, **kwargs)
        for ret in op.return_tensors:
            self.visit(ret, *args, **kwargs)

    def visit_call(self, op, *args, **kwargs):
        for arg in op.args:
            self.visit(arg, *args, **kwargs)
        op.ir_op = None
        op.result = None

    def visit_iter_var(self, op, *args, **kwargs):
        op.result = None

    def visit_compute(self, op, *args, **kwargs):
        self.visit(op.tensor, *args, **kwargs)
        self.visit(op.aux_tensor, *args, **kwargs)
        for body_op in op.body:
            self.visit(body_op, *args, **kwargs)
        op.ir_op = None
        op.result = None

    def visit_for(self, op, *args, **kwargs):
        op.iter_var.parent_loop = None
        for body_op in op.body:
            self.visit(body_op, *args, **kwargs)

    def visit_while(self, op, *args, **kwargs):
        self.visit(op.cond, *args, **kwargs)
        for body_op in op.body:
            self.visit(body_op, *args, **kwargs)

    def visit_alloc(self, op, *args, **kwargs):
        op.result = None
        op.ir_op = None

    def visit_binary(self, op, *args, **kwargs):
        self.visit(op.lhs, *args, **kwargs)
        self.visit(op.rhs, *args, **kwargs)
        op.result = None
        op.ir_op = None

    def visit_math_tanh(self, op, *args, **kwargs):
        op.result = None
        op.ir_op = None
        self.visit(op.expr, *args, **kwargs)

    def visit_neg(self, op, *args, **kwargs):
        self.visit(op.expr, *args, **kwargs)
        op.result = None
        op.ir_op = None

    def visit_cmp(self, op, *args, **kwargs):
        self.visit(op.lhs, *args, **kwargs)
        self.visit(op.rhs, *args, **kwargs)
        op.result = None
        op.ir_op = None

    def visit_load(self, op, *args, **kwargs):
        # for index in op.index:
        #   self.visit(index, *args, **kwargs)
        # self.visit(op.tensor, *args, **kwargs)
        op.result = None
        op.ir_op = None

    def visit_store(self, op, *args, **kwargs):
        # for index in op.index:
        #     self.visit(index, *args, **kwargs)
        # self.visit(op.tensor, *args, **kwargs)
        self.visit(op.value, *args, **kwargs)
        op.ir_op = None

    def visit_constant(self, op, *args, **kwargs):
        op.result = None
        op.ir_op = None

    def visit_cast(self, op, *args, **kwargs):
        self.visit(op.expr, *args, **kwargs)
        op.result = None
        op.ir_op = None

    def visit_if(self, op, *args, **kwargs):
        self.visit(op.cond, *args, **kwargs)
        for body_op in op.body:
            self.visit(body_op, *args, **kwargs)
        if op.else_branch_valid:
            for body_op in op.else_body:
                self.visit(body_op, *args, **kwargs)
        op.ir_op = None

    def visit_reduce(self, op, *args, **kwargs):
        self.visit(op.scalar, *args, **kwargs)
        self.visit(op.expr, *args, **kwargs)
        self.ir_op = None
        self.result = None

    def visit_select(self, op, *args, **kwargs):
        self.visit(op.cond, *args, **kwargs)
        self.visit(op.true_value, *args, **kwargs)
        self.visit(op.false_value, *args, **kwargs)
        self.ir_op = None
        self.result = None

    def visit_bitcast(self, op, *args, **kwargs):
        self.visit(op.expr, *args, **kwargs)
        self.ir_op = None
        self.result = None

    def visit_print(self, op, *args, **kwargs):
        for arg in op.args:
            self.visit(arg, *args, **kwargs)
        self.ir_op = None

    def visit_print_tensor(self, op, *args, **kwargs):
        self.visit(op.tensor, *args, **kwargs)
        self.ir_op = None

    def visit_get_bit(self, op, *args, **kwargs):
        self.visit(op.expr, *args, **kwargs)
        self.visit(op.index, *args, **kwargs)
        self.ir_op = None
        self.result = None

    def visit_get_slice(self, op, *args, **kwargs):
        self.visit(op.expr, *args, **kwargs)
        self.visit(op.start, *args, **kwargs)
        self.visit(op.end, *args, **kwargs)
        self.ir_op = None
        self.result = None

    def visit_set_bit(self, op, *args, **kwargs):
        self.visit(op.expr, *args, **kwargs)
        self.visit(op.index, *args, **kwargs)
        self.visit(op.value, *args, **kwargs)
        self.ir_op = None

    def visit_set_slice(self, op, *args, **kwargs):
        self.visit(op.expr, *args, **kwargs)
        self.visit(op.start, *args, **kwargs)
        self.visit(op.end, *args, **kwargs)
        self.visit(op.value, *args, **kwargs)
        self.ir_op = None

    def visit_bit_reverse(self, op, *args, **kwargs):
        self.visit(op.expr, *args, **kwargs)
        self.ir_op = None
        self.result = None

    def visit_constant_tensor(self, op, *args, **kwargs):
        op.ir_op = None
        op.result = None
        op.tensor.ir_op = None
        op.tensor.result = None

    def visit_struct_constuct(self, op, *args, **kwargs):
        for arg in op.args:
            self.visit(arg, *args, **kwargs)
        op.ir_op = None
        op.result = None

    def visit_struct_get(self, op, *args, **kwargs):
        self.visit(op.struct, *args, **kwargs)
        op.ir_op = None
        op.result = None

    def visit_op_handle(self, op, *args, **kwargs):
        op.ir_op = None
        op.result = None

    def visit_loop_handle(self, op, *args, **kwargs):
        self.visit(op.op_hdl, *args, **kwargs)
        op.ir_op = None
        op.result = None

    def visit_partition(self, op, *args, **kwargs):
        self.visit(op.tensor, *args, **kwargs)
        op.ir_op = None

    def visit_replace(self, op, *args, **kwargs):
        self.visit(op.target, *args, **kwargs)
        self.visit(op.src, *args, **kwargs)
        op.ir_op = None

    def visit_reshape(self, op, *args, **kwargs):
        self.visit(op.tensor, *args, **kwargs)
        op.ir_op = None

    def visit_reform(self, op, *args, **kwargs):
        self.visit(op.target, *args, **kwargs)
        op.ir_op = None

    def visit_reuse_at(self, op, *args, **kwargs):
        self.visit(op.target, *args, **kwargs)
        self.visit(op.axis, *args, **kwargs)
        op.ir_op = None
        op.result = None

    def visit_buffer_at(self, op, *args, **kwargs):
        self.visit(op.target, *args, **kwargs)
        self.visit(op.axis, *args, **kwargs)
        op.ir_op = None
        op.result = None

    def visit_inter_kernel_to(self, op, *args, **kwargs):
        self.visit(op.tensor, *args, **kwargs)
        self.visit(op.stage, *args, **kwargs)
        op.ir_op = None

    def visit_outline(self, op, *args, **kwargs):
        for hdl in op.stage_hdls:
            self.visit(hdl, *args, **kwargs)
        op.ir_op = None

    def visit_reorder(self, op, *args, **kwargs):
        for arg in op.args:
            self.visit(arg, *args, **kwargs)
        op.ir_op = None

    def visit_split(self, op, *args, **kwargs):
        self.visit(op.parent, *args, **kwargs)
        op.ir_op = None
        for loop in op.results:
            loop.result = None

    def visit_tile(self, op, *args, **kwargs):
        self.visit(op.x_parent, *args, **kwargs)
        self.visit(op.y_parent, *args, **kwargs)
        op.ir_op = None
        for loop in op.results:
            loop.result = None

    def visit_pipeline(self, op, *args, **kwargs):
        self.visit(op.target, *args, **kwargs)
        op.ir_op = None

    def visit_unroll(self, op, *args, **kwargs):
        self.visit(op.target, *args, **kwargs)
        op.ir_op = None

    def visit_parallel(self, op, *args, **kwargs):
        self.visit(op.target, *args, **kwargs)
        op.ir_op = None

    def visit_fuse(self, op, *args, **kwargs):
        for arg in op.arg_list:
            self.visit(arg, *args, **kwargs)
        op.ir_op = None
        op.result = None

    def visit_compute_at(self, op, *args, **kwargs):
        self.visit(op.stage, *args, **kwargs)
        self.visit(op.parent, *args, **kwargs)
        self.visit(op.axis, *args, **kwargs)
        op.ir_op = None

    def visit_systolic(self, op, *args, **kwargs):
        self.visit(op.target, *args, **kwargs)
