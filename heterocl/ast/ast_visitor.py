# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument, too-many-public-methods, too-many-branches

from hcl_mlir.exceptions import HCLNotImplementedError
from . import ast


class ASTVisitor:
    def __init__(self, name):
        self.name = name

    def visit(self, op, *args, **kwargs):
        if isinstance(op, ast.AST):
            self.visit_ast(op, *args, **kwargs)
        elif isinstance(op, ast.ComputeOp):
            self.visit_compute(op, *args, **kwargs)
        elif isinstance(op, ast.IterVar):
            self.visit_iter_var(op, *args, **kwargs)
        elif isinstance(op, ast.ReduceOp):
            self.visit_reduce(op, *args, **kwargs)
        elif isinstance(op, ast.AllocOp):
            self.visit_alloc(op, *args, **kwargs)
        elif isinstance(op, ast.Cmp):
            self.visit_cmp(op, *args, **kwargs)
        elif isinstance(op, ast.BinaryOp):
            self.visit_binary(op, *args, **kwargs)
        elif isinstance(op, ast.MathTanhOp):
            self.visit_math_tanh(op, *args, **kwargs)
        elif isinstance(op, ast.BitCastOp):
            self.visit_bitcast(op, *args, **kwargs)
        elif isinstance(op, ast.LoadOp):
            self.visit_load(op, *args, **kwargs)
        elif isinstance(op, ast.StoreOp):
            self.visit_store(op, *args, **kwargs)
        elif isinstance(op, ast.ConstantOp):
            self.visit_constant(op, *args, **kwargs)
        elif isinstance(op, ast.CastOp):
            self.visit_cast(op, *args, **kwargs)
        elif isinstance(op, ast.IfOp):
            self.visit_if(op, *args, **kwargs)
        elif isinstance(op, ast.ForOp):
            self.visit_for(op, *args, **kwargs)
        elif isinstance(op, ast.WhileOp):
            self.visit_while(op, *args, **kwargs)
        elif isinstance(op, ast.SelectOp):
            self.visit_select(op, *args, **kwargs)
        elif isinstance(op, ast.PrintOp):
            self.visit_print(op, *args, **kwargs)
        elif isinstance(op, ast.PrintTensorOp):
            self.visit_print_tensor(op, *args, **kwargs)
        elif isinstance(op, ast.GetBitOp):
            self.visit_get_bit(op, *args, **kwargs)
        elif isinstance(op, ast.GetSliceOp):
            self.visit_get_slice(op, *args, **kwargs)
        elif isinstance(op, ast.SetBitOp):
            self.visit_set_bit(op, *args, **kwargs)
        elif isinstance(op, ast.SetSliceOp):
            self.visit_set_slice(op, *args, **kwargs)
        elif isinstance(op, ast.BitReverseOp):
            self.visit_bit_reverse(op, *args, **kwargs)
        elif isinstance(op, ast.ConstantTensorOp):
            self.visit_constant_tensor(op, *args, **kwargs)
        elif isinstance(op, ast.StructConstructOp):
            self.visit_struct_construct(op, *args, **kwargs)
        elif isinstance(op, ast.StructGetOp):
            self.visit_struct_get(op, *args, **kwargs)
        elif isinstance(op, ast.FuncOp):
            self.visit_func(op, *args, **kwargs)
        elif isinstance(op, ast.CallOp):
            self.visit_call(op, *args, **kwargs)
        elif isinstance(op, ast.Neg):
            self.visit_neg(op, *args, **kwargs)
        elif isinstance(op, ast.OpHandle):
            self.visit_op_handle(op, *args, **kwargs)
        elif isinstance(op, ast.LoopHandle):
            self.visit_loop_handle(op, *args, **kwargs)
        elif isinstance(op, ast.ReuseAtOp):
            self.visit_reuse_at(op, *args, **kwargs)
        elif isinstance(op, ast.PartitionOp):
            self.visit_partition(op, *args, **kwargs)
        elif isinstance(op, ast.ReplaceOp):
            self.visit_replace(op, *args, **kwargs)
        elif isinstance(op, ast.ReshapeOp):
            self.visit_reshape(op, *args, **kwargs)
        elif isinstance(op, ast.ReformOp):
            self.visit_reform(op, *args, **kwargs)
        elif isinstance(op, ast.BufferAtOp):
            self.visit_buffer_at(op, *args, **kwargs)
        elif isinstance(op, ast.InterKernelToOp):
            self.visit_inter_kernel_to(op, *args, **kwargs)
        elif isinstance(op, ast.OutlineOp):
            self.visit_outline(op, *args, **kwargs)
        elif isinstance(op, ast.ReorderOp):
            self.visit_reorder(op, *args, **kwargs)
        elif isinstance(op, ast.SplitOp):
            self.visit_split(op, *args, **kwargs)
        elif isinstance(op, ast.TileOp):
            self.visit_tile(op, *args, **kwargs)
        elif isinstance(op, ast.PipelineOp):
            self.visit_pipeline(op, *args, **kwargs)
        elif isinstance(op, ast.UnrollOp):
            self.visit_unroll(op, *args, **kwargs)
        elif isinstance(op, ast.ParallelOp):
            self.visit_parallel(op, *args, **kwargs)
        elif isinstance(op, ast.FuseOp):
            self.visit_fuse(op, *args, **kwargs)
        elif isinstance(op, ast.ComputeAtOp):
            self.visit_compute_at(op, *args, **kwargs)
        elif isinstance(op, ast.SystolicOp):
            self.visit_systolic(op, *args, **kwargs)
        else:
            raise HCLNotImplementedError(
                f"{type(op)}'s {self.name} visitor is not implemented yet."
            )

    def visit_ast(self, _ast, *args, **kwargs):
        return

    def visit_compute(self, op, *args, **kwargs):
        return

    def visit_iter_var(self, op, *args, **kwargs):
        return

    def visit_reduce(self, op, *args, **kwargs):
        return

    def visit_alloc(self, op, *args, **kwargs):
        return

    def visit_cmp(self, op, *args, **kwargs):
        return

    def visit_binary(self, op, *args, **kwargs):
        return

    def visit_math_tanh(self, op, *args, **kwargs):
        return

    def visit_bitcast(self, op, *args, **kwargs):
        return

    def visit_load(self, op, *args, **kwargs):
        return

    def visit_store(self, op, *args, **kwargs):
        return

    def visit_constant(self, op, *args, **kwargs):
        return

    def visit_cast(self, op, *args, **kwargs):
        return

    def visit_if(self, op, *args, **kwargs):
        return

    def visit_for(self, op, *args, **kwargs):
        return

    def visit_while(self, op, *args, **kwargs):
        return

    def visit_select(self, op, *args, **kwargs):
        return

    def visit_print(self, op, *args, **kwargs):
        return

    def visit_print_tensor(self, op, *args, **kwargs):
        return

    def visit_get_bit(self, op, *args, **kwargs):
        return

    def visit_get_slice(self, op, *args, **kwargs):
        return

    def visit_set_bit(self, op, *args, **kwargs):
        return

    def visit_set_slice(self, op, *args, **kwargs):
        return

    def visit_bit_reverse(self, op, *args, **kwargs):
        return

    def visit_constant_tensor(self, op, *args, **kwargs):
        return

    def visit_struct_constuct(self, op, *args, **kwargs):
        return

    def visit_struct_get(self, op, *args, **kwargs):
        return

    def visit_func(self, op, *args, **kwargs):
        return

    def visit_call(self, op, *args, **kwargs):
        return

    def visit_neg(self, op, *args, **kwargs):
        return

    def visit_op_handle(self, op, *args, **kwargs):
        return

    def visit_loop_handle(self, op, *args, **kwargs):
        return

    def visit_reuse_at(self, op, *args, **kwargs):
        return

    def visit_partition(self, op, *args, **kwargs):
        return

    def visit_replace(self, op, *args, **kwargs):
        return

    def visit_reshape(self, op, *args, **kwargs):
        return

    def visit_reform(self, op, *args, **kwargs):
        return

    def visit_buffer_at(self, op, *args, **kwargs):
        return

    def visit_inter_kernel_to(self, op, *args, **kwargs):
        return

    def visit_outline(self, op, *args, **kwargs):
        return

    def visit_reorder(self, op, *args, **kwargs):
        return

    def visit_split(self, op, *args, **kwargs):
        return

    def visit_tile(self, op, *args, **kwargs):
        return

    def visit_pipeline(self, op, *args, **kwargs):
        return

    def visit_unroll(self, op, *args, **kwargs):
        return

    def visit_parallel(self, op, *args, **kwargs):
        return

    def visit_fuse(self, op, *args, **kwargs):
        return

    def visit_compute_at(self, op, *args, **kwargs):
        return

    def visit_systolic(self, op, *args, **kwargs):
        return
