import sys
import ast
import inspect
import string
from textwrap import dedent
from torch._C._jit_tree_views import (
    ClassDef, Ident, Stmt, Decl, Def, Var,
    EmptyTypeAnnotation, Param, ExprStmt, Assign,
    Delete, Return, Raise, Assert, AugAssign, While,
    For, If, Pass, Break, Continue, Apply, Dots, Select,
    TrueLiteral, FalseLiteral, NoneLiteral, Starred,
    ListLiteral, TupleLiteral, DictLiteral, Const,
    StringLiteral, ListComp, Attribute, BinOp, UnaryOp,
    SliceExpr, Subscript, TernaryIf, With, WithItem, Property,
)
from torch._utils_internal import get_source_lines_and_file
from torch._jit_internal import SourceContext, should_drop
import torch.jit.annotations

def build_def(ctx, py_def, type_line, def_name, self_name=None):
    body = py_def.body
    r = ctx.make_range(py_def.lineno + len(py_def.decorator_list),
                       py_def.col_offset,
                       py_def.col_offset + len("def"))
    param_list = build_param_list(ctx, py_def.args, self_name)
    return_type = None
    if getattr(py_def, 'returns', None) is not None:
        return_type = build_expr(ctx, py_def.returns)
    decl = Decl(r, param_list, return_type)
    is_method = self_name is not None
    if type_line is not None:
        type_comment_decl = torch._C.parse_type_comment(type_line)
        decl = torch._C.merge_type_from_type_comment(decl, type_comment_decl, is_method)

    return Def(Ident(r, def_name),
               decl,
               build_stmts(ctx, body))

def get_jit_def(fn, def_name, self_name=None):
    sourcelines, file_lineno, filename = get_source_lines_and_file(fn, torch._C.ErrorReport.call_stack())
    source = ''.join(sourcelines)
    dedent_src = dedent(source)
    py_ast = ast.parse(dedent_src)
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise RuntimeError("Expected a single top-level function")
    leading_whitespace_len = len(source.split('\n', 1)[0]) - len(dedent_src.split('\n', 1)[0])
    type_line = torch.jit.annotations.get_type_line(source)
    ctx = SourceContext(source, filename, file_lineno, leading_whitespace_len, True)
    fn_def = py_ast.body[0]

    if should_drop(fn):
        unused_fn_def = ast.parse("def unused_fn(self: Any):\n\traise RuntimeError(\"Cannot call @unused methods\")")
        if len(unused_fn_def.body) != 1 or not isinstance(unused_fn_def.body[0], ast.FunctionDef):
            raise RuntimeError("Expected a single top-level function")
        unused_def = unused_fn_def.body[0]
        fn_def.body = unused_def.body
        fn_def.args.kwarg = fn_def.args.vararg = None
        for arg in fn_def.args.args + fn_def.args.kwonlyargs:
            arg.annotation = unused_def.args.args[0].annotation

    return build_def(ctx, fn_def, type_line, def_name, self_name=self_name)



